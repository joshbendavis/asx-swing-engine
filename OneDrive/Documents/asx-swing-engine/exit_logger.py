"""
exit_logger.py
--------------
Background process that monitors open bracket trades in TWS and logs exits.

Responsibilities
  - Watch for TP / SL fills via ib_insync execDetailsEvent
  - Detect time-stop conditions (10 trading days open) and submit MKT SELL
  - On any exit: update logs/trades.csv row with exit price, type, P&L, hold days
  - Append to results/equity_curve.csv for live account tracking
  - Reconnect automatically if TWS restarts
  - PID-file guard prevents duplicate instances

trades.csv exit columns added on close
  exit_timestamp  exit_price  exit_type (target/stop/time_stop)
  pnl_aud         pnl_r       hold_days

equity_curve.csv columns
  date  order_ref  ticker  exit_type  pnl_aud  account_balance  running_pnl

Usage
  python exit_logger.py            # run continuously (Ctrl-C to stop)
  python exit_logger.py --once     # check current state once and exit
  python exit_logger.py --dry-run  # show what would be done, no writes
"""

import argparse
import asyncio
import csv
import logging
import os
import sys
from datetime import datetime, date
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
PAPER                  = True
IBKR_ACCOUNT           = os.getenv("IBKR_ACCOUNT", "DUK913437")
TWS_HOST               = "127.0.0.1"
TWS_PORT_PAPER         = 7497
TWS_PORT_LIVE          = 7496
CLIENT_ID              = 12               # distinct from executor (10) and cancel util (11)

STARTING_BALANCE       = float(os.getenv("PAPER_BALANCE_OVERRIDE", "20000"))
TIME_STOP_DAYS         = 10               # trading days before forced exit
TIME_STOP_CHECK_MINS   = 30              # how often to poll for time-stop conditions

TRADES_CSV             = Path("logs/trades.csv")
PENDING_CSV            = Path("logs/pending_orders.csv")
EQUITY_CSV             = Path("results/equity_curve.csv")
PID_FILE               = Path("logs/exit_logger.pid")

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.FileHandler(
            log_dir / f"exit_logger_{datetime.today().strftime('%Y%m%d')}.log",
            encoding="utf-8",
        ),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger("exit-logger")

# ---------------------------------------------------------------------------
# Trades CSV helpers
# ---------------------------------------------------------------------------

EXIT_COLS = [
    "exit_timestamp", "exit_price", "exit_type",
    "pnl_aud", "pnl_r", "hold_days",
]

ALL_COLS = [
    "timestamp", "ticker", "shares", "entry", "stop_loss", "target",
    "risk_per_share", "risk_aud", "order_ref",
    "parent_id", "tp_id", "sl_id",
    "trigger_count", "composite_score", "account", "paper",
] + EXIT_COLS


def _read_trades() -> pd.DataFrame:
    """Read trades.csv; add empty exit columns if missing."""
    if not TRADES_CSV.exists():
        return pd.DataFrame(columns=ALL_COLS)
    df = pd.read_csv(TRADES_CSV, parse_dates=["timestamp"])
    for col in EXIT_COLS:
        if col not in df.columns:
            df[col] = np.nan
    return df


def _write_trades(df: pd.DataFrame) -> None:
    df.to_csv(TRADES_CSV, index=False)


# ---------------------------------------------------------------------------
# Pending orders helpers
# ---------------------------------------------------------------------------

PENDING_COLS = [
    "timestamp", "ticker", "shares", "entry", "stop_loss", "target",
    "risk_per_share", "risk_aud", "order_ref",
    "parent_id", "tp_id", "sl_id",
    "trigger_count", "composite_score", "account", "paper",
]


def _read_pending() -> pd.DataFrame:
    """Read pending_orders.csv; return empty DataFrame if missing."""
    if not PENDING_CSV.exists():
        return pd.DataFrame(columns=PENDING_COLS)
    try:
        return pd.read_csv(PENDING_CSV, parse_dates=["timestamp"])
    except Exception:
        return pd.DataFrame(columns=PENDING_COLS)


def _write_pending(df: pd.DataFrame) -> None:
    PENDING_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(PENDING_CSV, index=False)


def _promote_to_trades(order_ref: str, actual_fill_price: float, actual_shares: int) -> None:
    """
    Move a record from pending_orders.csv to trades.csv on confirmed entry fill.

    The actual_fill_price overwrites the limit-order entry price so that
    all subsequent P&L calculations use the real fill price.
    actual_shares handles the case where a partial fill differs from requested.
    """
    pending = _read_pending()

    match = pending[pending["order_ref"] == order_ref]
    if match.empty:
        log.warning("_promote_to_trades: %s not found in pending_orders.csv", order_ref)
        return

    # Check idempotency — don't double-promote
    existing = _read_trades()
    if not existing.empty and order_ref in existing["order_ref"].values:
        log.info(
            "_promote_to_trades: %s already in trades.csv — skipping duplicate promote",
            order_ref,
        )
        return

    row = match.iloc[0].copy()

    # Overwrite with actual fill data
    original_limit = float(row["entry"])
    row["entry"]  = round(actual_fill_price, 4)
    row["shares"] = actual_shares

    # Recalculate risk_per_share and risk_aud against actual fill price
    stop = float(row["stop_loss"])
    rps  = actual_fill_price - stop
    if rps > 0:
        row["risk_per_share"] = round(rps, 4)
        row["risk_aud"]       = round(rps * actual_shares, 2)

    log.info(
        "ENTRY FILL  %-6s  limit=%.3f  fill=%.3f  qty=%d  risk_aud=$%.2f  → trades.csv",
        str(row["ticker"]).replace(".AX", ""),
        original_limit,
        actual_fill_price,
        actual_shares,
        row["risk_aud"],
    )

    # Append to trades.csv
    TRADES_CSV.parent.mkdir(parents=True, exist_ok=True)
    write_header = not TRADES_CSV.exists()
    with open(TRADES_CSV, "a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=ALL_COLS)
        if write_header:
            writer.writeheader()
        # Write entry columns only (no exit data yet)
        out = {col: row.get(col, "") for col in PENDING_COLS}
        for col in EXIT_COLS:
            out[col] = ""
        writer.writerow(out)

    # Remove from pending_orders.csv
    pending_remaining = pending[pending["order_ref"] != order_ref]
    _write_pending(pending_remaining)
    log.info("  Removed %s from pending_orders.csv", order_ref)


def _canonical_open(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each order_ref, return the single canonical open row:
      - Must have tp_id > 0 (a successfully submitted bracket)
      - Must NOT have an exit_timestamp already set
      - Take the latest timestamp per order_ref
    """
    valid = df[(df["tp_id"] > 0) & (df["exit_timestamp"].isna())].copy()
    valid = valid.sort_values("timestamp")
    return valid.drop_duplicates(subset="order_ref", keep="last")


def _trading_days_elapsed(entry_dt: datetime) -> int:
    """Number of ASX trading days between entry_dt and today (approx: business days)."""
    today = pd.Timestamp(date.today())
    entry = pd.Timestamp(entry_dt.date())
    bdays = pd.bdate_range(start=entry, end=today)
    return max(0, len(bdays) - 1)   # -1: entry day itself doesn't count


# ---------------------------------------------------------------------------
# Equity curve helpers
# ---------------------------------------------------------------------------

EQUITY_COLS = [
    "date", "order_ref", "ticker", "exit_type",
    "pnl_aud", "account_balance", "running_pnl",
]


def _init_equity_csv() -> None:
    EQUITY_CSV.parent.mkdir(parents=True, exist_ok=True)
    if not EQUITY_CSV.exists():
        with open(EQUITY_CSV, "w", newline="", encoding="utf-8") as fh:
            csv.DictWriter(fh, fieldnames=EQUITY_COLS).writeheader()
        log.info("Created equity curve -> %s", EQUITY_CSV)
    else:
        # Backfill running_pnl column if file pre-dates this schema
        try:
            df = pd.read_csv(EQUITY_CSV)
            if "running_pnl" not in df.columns and not df.empty:
                df["running_pnl"] = df["account_balance"] - STARTING_BALANCE
                df.to_csv(EQUITY_CSV, index=False)
                log.info("Backfilled running_pnl column in %s", EQUITY_CSV)
        except Exception:
            pass


def _current_equity_balance() -> float:
    """Return the last recorded account balance, or STARTING_BALANCE."""
    if not EQUITY_CSV.exists():
        return STARTING_BALANCE
    try:
        df = pd.read_csv(EQUITY_CSV)
        if df.empty:
            return STARTING_BALANCE
        return float(df["account_balance"].iloc[-1])
    except Exception:
        return STARTING_BALANCE


def _append_equity(order_ref: str, ticker: str, exit_type: str, pnl_aud: float) -> float:
    """Append one closed trade to the equity curve. Returns new balance."""
    balance = _current_equity_balance() + pnl_aud
    running_pnl = balance - STARTING_BALANCE
    row = {
        "date":            datetime.now().strftime("%Y-%m-%d"),
        "order_ref":       order_ref,
        "ticker":          ticker,
        "exit_type":       exit_type,
        "pnl_aud":         round(pnl_aud, 2),
        "account_balance": round(balance, 2),
        "running_pnl":     round(running_pnl, 2),
    }
    with open(EQUITY_CSV, "a", newline="", encoding="utf-8") as fh:
        csv.DictWriter(fh, fieldnames=EQUITY_COLS).writerow(row)
    return balance


# ---------------------------------------------------------------------------
# Exit processing
# ---------------------------------------------------------------------------

def _record_exit(
    order_ref: str,
    exit_price: float,
    exit_type: str,      # "target" | "stop" | "time_stop"
    dry_run: bool = False,
) -> None:
    """
    Update the canonical open row for order_ref with exit data,
    then append to the equity curve.
    """
    df = _read_trades()
    open_rows = _canonical_open(df)

    match = open_rows[open_rows["order_ref"] == order_ref]
    if match.empty:
        log.warning("_record_exit: no open row found for %s - already closed?", order_ref)
        return

    row   = match.iloc[0]
    entry = float(row["entry"])
    shares = int(row["shares"])
    risk_per_share = float(row["risk_per_share"])
    entry_dt = pd.to_datetime(row["timestamp"])

    pnl_aud  = round((exit_price - entry) * shares, 2)
    pnl_r    = round((exit_price - entry) / risk_per_share, 3) if risk_per_share > 0 else 0.0
    hold_days = _trading_days_elapsed(entry_dt)

    log.info(
        "EXIT  %-6s  %-10s  exit=%.3f  entry=%.3f  "
        "pnl=$%.2f (%.2fR)  hold=%dd",
        row["ticker"].replace(".AX", ""), exit_type,
        exit_price, entry, pnl_aud, pnl_r, hold_days,
    )

    if dry_run:
        log.info("  [DRY-RUN] would update trades.csv and equity_curve.csv")
        return

    # Update canonical row in trades.csv
    idx = match.index[0]
    df.loc[idx, "exit_timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    df.loc[idx, "exit_price"]     = exit_price
    df.loc[idx, "exit_type"]      = exit_type
    df.loc[idx, "pnl_aud"]        = pnl_aud
    df.loc[idx, "pnl_r"]          = pnl_r
    df.loc[idx, "hold_days"]      = hold_days
    _write_trades(df)

    new_balance = _append_equity(order_ref, str(row["ticker"]), exit_type, pnl_aud)
    log.info("  trades.csv updated  |  equity $%.2f", new_balance)


# ---------------------------------------------------------------------------
# Time-stop handler
# ---------------------------------------------------------------------------

def _check_time_stops(ib, dry_run: bool = False) -> None:
    """
    For every open trade where trading days elapsed >= TIME_STOP_DAYS,
    submit a MKT SELL to close the position immediately.
    """
    from ib_insync import MarketOrder, Stock

    df = _read_trades()
    open_rows = _canonical_open(df)

    if open_rows.empty:
        return

    for _, row in open_rows.iterrows():
        entry_dt  = pd.to_datetime(row["timestamp"])
        days_held = _trading_days_elapsed(entry_dt)

        if days_held < TIME_STOP_DAYS:
            continue

        ticker = str(row["ticker"])
        symbol = ticker.replace(".AX", "")
        shares = int(row["shares"])
        order_ref = str(row["order_ref"])

        log.info(
            "TIME-STOP  %-6s  held %d trading days (>= %d) - closing at market",
            symbol, days_held, TIME_STOP_DAYS,
        )

        if dry_run:
            log.info("  [DRY-RUN] would submit MKT SELL %d %s", shares, symbol)
            _record_exit(order_ref, float(row["entry"]), "time_stop", dry_run=True)
            continue

        try:
            contract = Stock(symbol, "SMART", "AUD")
            ib.qualifyContracts(contract)

            order = MarketOrder(
                action        = "SELL",
                totalQuantity = shares,
                orderRef      = order_ref + "-TS",
                account       = IBKR_ACCOUNT,
                tif           = "DAY",
            )
            order.tif = "DAY"   # time-stop exits must fill same session

            trade = ib.placeOrder(contract, order)
            ib.sleep(1)

            # Cancel the surviving OCA leg (TP or SL that didn't trigger)
            for t in ib.trades():
                ref = t.order.orderRef
                if ref in (order_ref + "-TP", order_ref + "-SL"):
                    ib.cancelOrder(t.order)

            # Price will come via execDetailsEvent; record at market price
            # For now record at last known price (fill event will update if available)
            fills = [f for f in ib.fills()
                     if f.execution.orderRef == order_ref + "-TS"]
            exit_px = float(fills[-1].execution.avgPrice) if fills else float(row["entry"])
            _record_exit(order_ref, exit_px, "time_stop")

        except Exception as exc:
            log.error("Time-stop order failed for %s: %s", symbol, exc, exc_info=True)


# ---------------------------------------------------------------------------
# ib_insync event handlers
# ---------------------------------------------------------------------------

def _make_exec_handler(dry_run: bool):
    """Return an execDetailsEvent callback bound to dry_run flag."""

    # Track cumulative fill quantities per order_ref so we can detect
    # when an ASX multi-tranche entry fill is complete.
    _entry_cum: dict[str, int] = {}

    def on_exec_details(trade, fill):
        """Fired whenever an execution (fill) arrives from TWS."""
        ref       = fill.execution.orderRef   # e.g. "ASX-20260405-SMR" or "-TP"/"-SL"
        action    = fill.execution.side       # "BOT" or "SLD"
        avg_price = float(fill.execution.avgPrice)
        cum_qty   = int(fill.execution.cumQty)
        fill_qty  = int(fill.execution.shares)

        if not ref.startswith("ASX-"):
            return

        # ── Entry fill (BOT, no suffix) ───────────────────────────────────────
        if action == "BOT" and not any(ref.endswith(s) for s in ("-TP", "-SL", "-TS")):
            base_ref = ref
            _entry_cum[base_ref] = cum_qty   # track running cumQty

            log.info(
                "ENTRY FILL  ref=%-28s  cumQty=%d  avgPrice=%.3f",
                ref, cum_qty, avg_price,
            )

            # Look up the requested quantity from pending_orders.csv
            pending = _read_pending()
            match   = pending[pending["order_ref"] == base_ref]
            requested_qty = int(match.iloc[0]["shares"]) if not match.empty else 0

            if requested_qty > 0 and cum_qty < requested_qty:
                log.info(
                    "  Partial fill %d / %d shares — waiting for remainder",
                    cum_qty, requested_qty,
                )
                return  # wait for final tranche

            # Final fill reached (cumQty >= requested, or no pending record found)
            if dry_run:
                log.info(
                    "  [DRY-RUN] Would promote %s to trades.csv  fill=%.3f  qty=%d",
                    base_ref, avg_price, cum_qty,
                )
            else:
                _promote_to_trades(base_ref, avg_price, cum_qty)
            return

        # ── Exit fill (SLD) ───────────────────────────────────────────────────
        if action != "SLD":
            return

        if ref.endswith("-TP"):
            base_ref  = ref[:-3]
            exit_type = "target"
        elif ref.endswith("-SL"):
            base_ref  = ref[:-3]
            exit_type = "stop"
        elif ref.endswith("-TS"):
            base_ref  = ref[:-3]
            exit_type = "time_stop"
        else:
            return   # not one of ours

        log.info(
            "EXIT FILL  ref=%-30s  action=%-3s  price=%.3f  qty=%d",
            ref, action, avg_price, fill_qty,
        )
        _record_exit(base_ref, avg_price, exit_type, dry_run=dry_run)

    return on_exec_details


# ---------------------------------------------------------------------------
# Async time-stop polling loop
# ---------------------------------------------------------------------------

async def _time_stop_loop(ib, dry_run: bool) -> None:
    """Runs forever; checks for time-stopped trades every TIME_STOP_CHECK_MINS."""
    while True:
        await asyncio.sleep(TIME_STOP_CHECK_MINS * 60)
        try:
            _check_time_stops(ib, dry_run=dry_run)
        except Exception as exc:
            log.error("Time-stop check failed: %s", exc, exc_info=True)


# ---------------------------------------------------------------------------
# Startup summary
# ---------------------------------------------------------------------------

def _print_open_summary() -> None:
    """Log the current open positions and pending orders we will be watching."""
    df = _read_trades()
    open_rows = _canonical_open(df)

    if open_rows.empty:
        log.info("No open positions found in trades.csv.")
    else:
        log.info("Watching %d open position(s):", len(open_rows))
        for _, r in open_rows.iterrows():
            days = _trading_days_elapsed(pd.to_datetime(r["timestamp"]))
            log.info(
                "  %-6s  entry=%.3f  stop=%.3f  target=%.3f  "
                "tp_id=%-4d  sl_id=%-4d  held=%dd",
                str(r["ticker"]).replace(".AX",""),
                r["entry"], r["stop_loss"], r["target"],
                int(r["tp_id"]), int(r["sl_id"]), days,
            )

    # Also report pending orders (submitted, awaiting entry fill)
    pending = _read_pending()
    # Filter out any already promoted to trades
    if not pending.empty and not df.empty:
        filled_refs = set(df["order_ref"].dropna())
        pending = pending[~pending["order_ref"].isin(filled_refs)]

    if not pending.empty:
        log.info("Pending orders (awaiting entry fill): %d", len(pending))
        for _, r in pending.iterrows():
            days = _trading_days_elapsed(pd.to_datetime(r["timestamp"]))
            stale = " *** STALE ***" if days >= 3 else ""
            log.info(
                "  %-6s  limit=%.3f  stop=%.3f  target=%.3f  "
                "waiting=%dd%s",
                str(r["ticker"]).replace(".AX",""),
                r["entry"], r["stop_loss"], r["target"],
                days, stale,
            )


# ---------------------------------------------------------------------------
# --once mode: snapshot check without staying connected
# ---------------------------------------------------------------------------

def run_once(dry_run: bool) -> None:
    """Connect, check time stops, print open positions, disconnect."""
    from ib_insync import IB

    port = TWS_PORT_PAPER if PAPER else TWS_PORT_LIVE
    ib = IB()
    log.info("Connecting (once mode) %s:%d ...", TWS_HOST, port)
    ib.connect(TWS_HOST, port, clientId=CLIENT_ID, timeout=15)

    _print_open_summary()
    _check_time_stops(ib, dry_run=dry_run)

    ib.disconnect()
    log.info("Done.")


# ---------------------------------------------------------------------------
# Main async loop with auto-reconnect
# ---------------------------------------------------------------------------

async def _run(dry_run: bool) -> None:
    from ib_insync import IB, util

    port = TWS_PORT_PAPER if PAPER else TWS_PORT_LIVE
    mode = "PAPER" if PAPER else "LIVE"

    ib = IB()
    exec_handler = _make_exec_handler(dry_run)

    reconnect_delay = 30   # seconds between reconnect attempts

    while True:
        try:
            log.info("Connecting to TWS %s  %s:%d  clientId=%d ...",
                     mode, TWS_HOST, port, CLIENT_ID)
            await ib.connectAsync(TWS_HOST, port,
                                  clientId=CLIENT_ID, timeout=15)

            log.info("Connected  |  account=%s  |  balance_base=$%.0f",
                     IBKR_ACCOUNT, STARTING_BALANCE)

            # Subscribe to fill events
            ib.execDetailsEvent += exec_handler

            _print_open_summary()

            # Run initial time-stop check immediately on connect
            _check_time_stops(ib, dry_run=dry_run)

            # Start periodic time-stop polling as a concurrent task
            ts_task = asyncio.ensure_future(_time_stop_loop(ib, dry_run))

            # Block until TWS disconnects
            await ib.disconnectedEvent

            ts_task.cancel()
            ib.execDetailsEvent -= exec_handler
            log.warning("TWS disconnected. Reconnecting in %ds ...", reconnect_delay)

        except ConnectionRefusedError:
            log.warning("TWS not reachable on port %d. "
                        "Retrying in %ds ...", port, reconnect_delay)
        except Exception as exc:
            log.error("Unexpected error: %s. Retrying in %ds ...",
                      exc, reconnect_delay, exc_info=True)

        await asyncio.sleep(reconnect_delay)


# ---------------------------------------------------------------------------
# PID-file guard — prevent duplicate instances
# ---------------------------------------------------------------------------

def _write_pid() -> None:
    """Write current PID to PID_FILE."""
    PID_FILE.parent.mkdir(exist_ok=True)
    PID_FILE.write_text(str(os.getpid()))
    log.info("PID %d written -> %s", os.getpid(), PID_FILE)


def _clear_pid() -> None:
    try:
        PID_FILE.unlink(missing_ok=True)
    except Exception:
        pass


def _already_running() -> bool:
    """Return True if a previous instance is still alive."""
    if not PID_FILE.exists():
        return False
    try:
        pid = int(PID_FILE.read_text().strip())
    except (ValueError, OSError):
        return False

    # Check if that PID is alive (works on Windows + Unix)
    try:
        import signal
        if sys.platform == "win32":
            import ctypes
            handle = ctypes.windll.kernel32.OpenProcess(0x0400, False, pid)  # PROCESS_QUERY_INFO
            if handle:
                ctypes.windll.kernel32.CloseHandle(handle)
                return True
            return False
        else:
            os.kill(pid, 0)   # signal 0 = existence check
            return True
    except (ProcessLookupError, PermissionError, OSError):
        return False


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="ASX Swing Engine - exit logger")
    p.add_argument("--once",    action="store_true",
                   help="Check current state once and exit (no persistent loop)")
    p.add_argument("--dry-run", action="store_true",
                   help="Log what would be written without modifying any files")
    p.add_argument("--force",   action="store_true",
                   help="Start even if another instance appears to be running")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Guard: refuse to start a second continuous instance
    if not args.once and not args.force and _already_running():
        log.warning(
            "exit_logger already running (PID %s). "
            "Use --force to override.", PID_FILE.read_text().strip()
        )
        sys.exit(0)

    log.info("=" * 62)
    log.info("  EXIT LOGGER  |  %s  |  account=%s  |  time-stop=%dd",
             "PAPER" if PAPER else "LIVE", IBKR_ACCOUNT, TIME_STOP_DAYS)
    log.info("  Base balance: $%.0f  |  dry-run: %s",
             STARTING_BALANCE, args.dry_run)
    log.info("=" * 62)

    _init_equity_csv()

    if args.once:
        run_once(dry_run=args.dry_run)
    else:
        _write_pid()
        try:
            from ib_insync import util
            util.run(_run(dry_run=args.dry_run))
        except KeyboardInterrupt:
            log.info("Stopped by user.")
        finally:
            _clear_pid()
            log.info("PID file removed.")
