"""
ibkr_executor.py
----------------
Connect to TWS / IB Gateway and submit bracket orders for every signal
produced by signals.py.

Bracket structure
    Parent  : LMT BUY  @ entry     (next-day open; use LMT near close price)
    Target  : LMT SELL @ target    (OCA group with stop)
    Stop    : STP SELL @ stop_loss (OCA group with target)

Position sizing
    shares = floor(account_balance * RISK_PCT / (entry - stop_loss))
    Capped so total open positions <= MAX_POSITIONS.

Paper vs live
    Set PAPER = True  -> connects to TWS paper port 7497
    Set PAPER = False -> connects to TWS live  port 7496
    Account is always validated against IBKR_ACCOUNT at the top.

Order log
    Every submitted bracket is appended to logs/trades.csv with:
    timestamp, ticker, shares, entry, stop_loss, target,
    order_ref, parent_id, tp_id, sl_id, trigger_count, composite_score

Usage
    python ibkr_executor.py                  # reads results/signals_output.csv
    python ibkr_executor.py --dry-run        # prints orders, does NOT submit
    python ibkr_executor.py --signals path/to/file.csv
"""

import argparse
import csv
import math
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Config - edit these, or override via .env
# ---------------------------------------------------------------------------
PAPER            = True                 # flip to False for live trading
IBKR_ACCOUNT     = os.getenv("IBKR_ACCOUNT", "DUK913437")
TWS_HOST         = "127.0.0.1"
TWS_PORT_PAPER   = 7497
TWS_PORT_LIVE    = 7496
CLIENT_ID        = 10                   # unique clientId - change if clashes

RISK_PCT         = 0.015                # 1.5% of account per trade
MAX_POSITIONS    = 6                    # max concurrent open positions
MIN_SHARES       = 1                    # minimum viable parcel
MIN_TRIGGERS     = 2                    # only trade signals with >= N triggers

# Optional: set in .env to override live NAV for sizing (paper mode only).
# Useful when paper account is funded at $1M but you want to size as $20k.
#   PAPER_BALANCE_OVERRIDE=20000
PAPER_BALANCE_OVERRIDE = os.getenv("PAPER_BALANCE_OVERRIDE", "").strip()

SIGNALS_CSV      = Path("results/signals_final.csv")    # written by risk_engine.py
TRADE_LOG        = Path("logs/trades.csv")
PENDING_LOG      = Path("logs/pending_orders.csv")      # submitted but not yet filled

CONNECT_TIMEOUT  = 10                   # seconds to wait for TWS connection
ORDER_PAUSE      = 0.5                  # seconds between order submissions

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
import logging

log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.FileHandler(
            log_dir / f"executor_{datetime.today().strftime('%Y%m%d_%H%M%S')}.log",
            encoding="utf-8",
        ),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger("ibkr-executor")

# ---------------------------------------------------------------------------
# CSV schemas
# ---------------------------------------------------------------------------
TRADE_LOG_COLS = [
    "timestamp", "ticker", "shares", "entry", "stop_loss", "target",
    "risk_per_share", "risk_aud", "order_ref",
    "parent_id", "tp_id", "sl_id",
    "trigger_count", "composite_score", "account", "paper",
]

# pending_orders.csv uses the same schema as trades.csv so the dashboard
# can render both tables identically.
PENDING_LOG_COLS = TRADE_LOG_COLS


def _init_csv(path: Path, cols: list[str]) -> None:
    """Create a CSV file with header row if it does not already exist."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        with open(path, "w", newline="", encoding="utf-8") as fh:
            csv.DictWriter(fh, fieldnames=cols).writeheader()
        log.info("Created %s", path)


def _init_trade_log() -> None:
    _init_csv(TRADE_LOG, TRADE_LOG_COLS)


def _init_pending_log() -> None:
    _init_csv(PENDING_LOG, PENDING_LOG_COLS)


def _append_to_csv(path: Path, cols: list[str], row: dict) -> None:
    """Append one row to a CSV file."""
    with open(path, "a", newline="", encoding="utf-8") as fh:
        csv.DictWriter(fh, fieldnames=cols).writerow(
            {col: row.get(col, "") for col in cols}
        )


def _append_trade_log(row: dict) -> None:
    """Append one confirmed-fill record to logs/trades.csv."""
    _append_to_csv(TRADE_LOG, TRADE_LOG_COLS, row)


def _append_pending_log(row: dict) -> None:
    """Append one submitted-but-unfilled bracket to logs/pending_orders.csv."""
    _append_to_csv(PENDING_LOG, PENDING_LOG_COLS, row)


def _count_pending_orders() -> int:
    """
    Count pending orders whose order_ref has NOT yet appeared in trades.csv.
    Used to calculate remaining position headroom before submitting new orders.
    """
    if not PENDING_LOG.exists():
        return 0
    try:
        pending = pd.read_csv(PENDING_LOG)
        if pending.empty:
            return 0
        if TRADE_LOG.exists():
            trades = pd.read_csv(TRADE_LOG)
            filled_refs = set(trades["order_ref"].dropna()) if not trades.empty else set()
        else:
            filled_refs = set()
        unfilled = pending[~pending["order_ref"].isin(filled_refs)]
        return len(unfilled.drop_duplicates(subset="order_ref"))
    except Exception:
        return 0


def _already_in_pending(symbol: str) -> bool:
    """
    Return True if an open (unfilled) pending order for this symbol exists.
    Prevents re-submission if executor runs twice on the same day.
    """
    if not PENDING_LOG.exists():
        return False
    try:
        pending = pd.read_csv(PENDING_LOG)
        if pending.empty:
            return False
        today_prefix = f"ASX-{datetime.today().strftime('%Y%m%d')}-{symbol}"
        # Check if already promoted to trades
        filled_refs: set[str] = set()
        if TRADE_LOG.exists():
            trades = pd.read_csv(TRADE_LOG)
            if not trades.empty:
                filled_refs = set(trades["order_ref"].dropna())
        unresolved = pending[~pending["order_ref"].isin(filled_refs)]
        return unresolved["order_ref"].str.startswith(today_prefix).any()
    except Exception:
        return False


# ---------------------------------------------------------------------------
# IBKR helpers
# ---------------------------------------------------------------------------

def _connect(dry_run: bool = False):
    """
    Return a connected ib_insync IB instance, or None if dry_run.
    Raises on connection failure.
    """
    if dry_run:
        return None

    try:
        from ib_insync import IB
    except ImportError:
        log.error("ib_insync is not installed.  Run: pip install ib_insync")
        sys.exit(1)

    port = TWS_PORT_PAPER if PAPER else TWS_PORT_LIVE
    mode = "PAPER" if PAPER else "LIVE"
    log.info("Connecting to TWS %s  %s:%d  clientId=%d ...", mode, TWS_HOST, port, CLIENT_ID)

    ib = IB()
    ib.connect(TWS_HOST, port, clientId=CLIENT_ID, timeout=CONNECT_TIMEOUT)

    if not ib.isConnected():
        raise ConnectionError(f"Could not connect to TWS on port {port}")

    log.info("Connected.  Account: %s", IBKR_ACCOUNT)
    return ib


def _get_account_balance(ib) -> float:
    """Return NetLiquidation for IBKR_ACCOUNT in AUD."""
    from ib_insync import AccountValue
    vals: list[AccountValue] = ib.accountValues(account=IBKR_ACCOUNT)
    for v in vals:
        if v.tag == "NetLiquidation" and v.currency == "AUD":
            return float(v.value)
    # Fallback: TotalCashValue
    for v in vals:
        if v.tag == "TotalCashValue" and v.currency == "AUD":
            return float(v.value)
    raise ValueError(f"Could not retrieve AUD NetLiquidation for account {IBKR_ACCOUNT}")


def _count_open_positions(ib) -> int:
    """Count currently open equity positions for IBKR_ACCOUNT."""
    positions = ib.positions(account=IBKR_ACCOUNT)
    return sum(1 for p in positions if abs(p.position) > 0)


def _already_ordered_today(ib, symbol: str) -> bool:
    """
    Return True if an open or filled order for this symbol already exists today.
    Prevents duplicate submission if executor is run more than once.
    """
    today_str = datetime.today().strftime("%Y%m%d")
    for trade in ib.trades():
        if (trade.contract.symbol == symbol
                and trade.order.orderRef.startswith(f"ASX-{today_str}")):
            return True
    return False


def _round_to_tick(price: float) -> float:
    """
    Snap a price to the nearest valid ASX minimum price variation (tick size).

    ASX tick schedule:
        price <  $0.10  ->  $0.001 ticks
        price <= $2.00  ->  $0.005 ticks
        price >  $2.00  ->  $0.010 ticks

    Uses Decimal arithmetic to avoid floating-point rounding artefacts
    (e.g. 255 * 0.01 = 2.5499999... in IEEE 754).
    """
    from decimal import Decimal, ROUND_HALF_UP

    if price < 0.10:
        tick = Decimal("0.001")
    elif price <= 2.00:
        tick = Decimal("0.005")
    else:
        tick = Decimal("0.010")

    price_d = Decimal(str(price))
    snapped  = (price_d / tick).to_integral_value(ROUND_HALF_UP) * tick
    return float(snapped)


def _make_contract(symbol: str):
    """Return an ASX Stock contract using SMART routing (avoids direct-route block)."""
    from ib_insync import Stock
    return Stock(symbol, "SMART", "AUD")


def _place_bracket(
    ib,
    symbol: str,
    shares: int,
    entry: float,
    stop_loss: float,
    target: float,
    order_ref: str,
) -> tuple:
    """
    Submit a bracket order and return (parent_id, tp_id, sl_id).

    Bracket structure:
        parent  : LMT BUY  @ entry   (transmit=False initially)
        tp      : LMT SELL @ target  (OCA)
        sl      : STP SELL @ stop    (OCA)
    All three are linked via OCA group and transmitted together.
    All prices are snapped to the correct ASX tick size before submission.
    """
    from ib_insync import LimitOrder, StopOrder

    # Snap all prices to valid ASX tick sizes before sending
    entry_p  = _round_to_tick(entry)
    target_p = _round_to_tick(target)
    stop_p   = _round_to_tick(stop_loss)

    oca_group = f"{order_ref}-OCA"

    parent = LimitOrder(
        action        = "BUY",
        totalQuantity = shares,
        lmtPrice      = entry_p,
        tif           = "GTC",          # Good-Till-Cancelled: survives overnight / weekend
        orderRef      = order_ref,
        account       = IBKR_ACCOUNT,
        transmit      = False,          # hold until children are attached
    )

    take_profit = LimitOrder(
        action        = "SELL",
        totalQuantity = shares,
        lmtPrice      = target_p,
        tif           = "GTC",
        orderRef      = order_ref + "-TP",
        ocaGroup      = oca_group,
        ocaType       = 1,              # cancel remaining when one fills
        account       = IBKR_ACCOUNT,
        parentId      = 0,              # filled in after parent placement
        transmit      = False,
    )

    stop_loss_order = StopOrder(
        action        = "SELL",
        totalQuantity = shares,
        stopPrice     = stop_p,
        tif           = "GTC",
        orderRef      = order_ref + "-SL",
        ocaGroup      = oca_group,
        ocaType       = 1,
        account       = IBKR_ACCOUNT,
        parentId      = 0,
        transmit      = True,           # transmit=True on last child sends all three
    )

    contract = _make_contract(symbol)
    ib.qualifyContracts(contract)

    # Force GTC immediately before each placeOrder call so TWS DAY presets
    # cannot override the value that was set in the constructor above.
    parent.tif          = "GTC"
    take_profit.tif     = "GTC"
    stop_loss_order.tif = "GTC"

    # Place parent first to get its orderId
    parent_trade    = ib.placeOrder(contract, parent)
    parent_id       = parent_trade.order.orderId

    # Wire children to parent
    take_profit.parentId      = parent_id
    stop_loss_order.parentId  = parent_id

    take_profit.tif     = "GTC"     # re-assert after parentId mutation
    stop_loss_order.tif = "GTC"

    tp_trade = ib.placeOrder(contract, take_profit)
    sl_trade = ib.placeOrder(contract, stop_loss_order)

    # Allow TWS to process
    ib.sleep(0.5)

    return parent_id, tp_trade.order.orderId, sl_trade.order.orderId


# ---------------------------------------------------------------------------
# Core executor
# ---------------------------------------------------------------------------

def run_executor(signals_path: Path, dry_run: bool = False) -> int:
    """
    Read signals CSV, size positions, submit bracket orders.
    Returns the number of orders submitted (or would-be submitted in dry-run).
    """
    _init_trade_log()
    _init_pending_log()

    # ── Load signals ─────────────────────────────────────────────────────────
    if not signals_path.exists():
        log.error("Signals file not found: %s", signals_path)
        return 0

    # Guard: file may exist but be empty (0 bytes) — pd.read_csv raises
    # "No columns to parse from file" in that case; treat it as no signals.
    if signals_path.stat().st_size == 0:
        log.info("Signals file is empty — nothing to trade.")
        return 0
    try:
        signals = pd.read_csv(signals_path)
    except Exception as exc:
        log.warning("Could not parse signals file (%s): %s — skipping.", signals_path, exc)
        return 0
    if signals.empty:
        log.info("No signals in %s - nothing to trade.", signals_path)
        return 0

    # Filter to minimum trigger count
    eligible = signals[signals["trigger_count"] >= MIN_TRIGGERS].copy()
    log.info(
        "Signals loaded: %d total, %d with trigger_count >= %d",
        len(signals), len(eligible), MIN_TRIGGERS,
    )

    if eligible.empty:
        log.info("No eligible signals after trigger filter.")
        return 0

    # ── Connect to TWS ────────────────────────────────────────────────────────
    ib = _connect(dry_run=dry_run)

    # ── Account balance + position headroom ───────────────────────────────────
    if dry_run:
        # In dry-run, prefer the override; fall back to $20k simulation
        balance        = float(PAPER_BALANCE_OVERRIDE) if PAPER_BALANCE_OVERRIDE else 20_000.0
        open_positions = 0
        log.info("[DRY-RUN] Using balance $%.2f AUD, 0 open positions", balance)
    else:
        open_positions = _count_open_positions(ib)
        if PAPER and PAPER_BALANCE_OVERRIDE:
            balance = float(PAPER_BALANCE_OVERRIDE)
            log.info(
                "PAPER_BALANCE_OVERRIDE: sizing against $%.2f AUD  "
                "|  Open positions: %d / %d",
                balance, open_positions, MAX_POSITIONS,
            )
        else:
            balance = _get_account_balance(ib)
            log.info(
                "Account balance: $%.2f AUD  |  Open positions: %d / %d",
                balance, open_positions, MAX_POSITIONS,
            )

    # Pending orders (submitted but not yet filled) also consume slots —
    # they will become real positions once the entry limit triggers.
    pending_count = _count_pending_orders()
    effective_open = open_positions + pending_count
    headroom = MAX_POSITIONS - effective_open

    log.info(
        "Position slots: %d filled + %d pending = %d effective / %d max  ->  %d slot(s) free",
        open_positions, pending_count, effective_open, MAX_POSITIONS, max(0, headroom),
    )

    if headroom <= 0:
        log.info(
            "No headroom (%d filled + %d pending = %d/%d). No new orders.",
            open_positions, pending_count, effective_open, MAX_POSITIONS,
        )
        if ib:
            ib.disconnect()
        return 0

    # ── Place orders ──────────────────────────────────────────────────────────
    today_str  = datetime.today().strftime("%Y%m%d")
    submitted  = 0

    for _, row in eligible.iterrows():
        if submitted >= headroom:
            log.info("Headroom reached (%d new orders). Stopping.", submitted)
            break

        ticker       = str(row["ticker"])          # e.g. "SMR.AX"
        symbol       = ticker.replace(".AX", "")   # e.g. "SMR"
        entry        = float(row["entry"])
        stop_loss    = float(row["stop_loss"])
        target       = float(row["target"])
        risk_per_shr = entry - stop_loss
        t_count      = int(row["trigger_count"])
        score        = float(row.get("composite_score", 0))

        if risk_per_shr <= 0:
            log.warning("%-6s: invalid risk_per_share (entry=%.3f stop=%.3f) - skipping.", symbol, entry, stop_loss)
            continue

        # Position size — use risk_multiplier from risk_engine if present
        risk_multiplier = float(row.get("risk_multiplier", 1.0)) \
                          if "risk_multiplier" in row.index else 1.0
        risk_aud = balance * RISK_PCT * risk_multiplier
        shares   = int(math.floor(risk_aud / risk_per_shr))

        if shares < MIN_SHARES:
            log.warning(
                "%-6s: sized to %d shares (risk $%.2f / share $%.4f) - below minimum, skipping.",
                symbol, shares, risk_aud, risk_per_shr,
            )
            continue

        order_ref = f"ASX-{today_str}-{symbol}"

        # ── Duplicate guard ───────────────────────────────────────────────────
        # Check both TWS open trades AND pending_orders.csv to survive restarts.
        if not dry_run and _already_ordered_today(ib, symbol):
            log.info("%-6s: order already in TWS today - skipping.", symbol)
            continue
        if _already_in_pending(symbol):
            log.info("%-6s: unfilled pending order exists - skipping.", symbol)
            continue

        log.info(
            "%-6s | shares=%-4d | entry=%.3f | stop=%.3f | target=%.3f | "
            "risk=$%.2f (×%.2f) | triggers=%d | score=%.1f",
            symbol, shares, entry, stop_loss, target,
            risk_aud, risk_multiplier, t_count, score,
        )

        if dry_run:
            parent_id = tp_id = sl_id = 0
            log.info("  [DRY-RUN] Would submit bracket for %s x %d", symbol, shares)
        else:
            try:
                parent_id, tp_id, sl_id = _place_bracket(
                    ib, symbol, shares, entry, stop_loss, target, order_ref,
                )
                log.info(
                    "  Bracket submitted: parentId=%d  tpId=%d  slId=%d",
                    parent_id, tp_id, sl_id,
                )
                time.sleep(ORDER_PAUSE)
            except Exception as exc:
                log.error("  Order FAILED for %s: %s", symbol, exc, exc_info=True)
                continue

        # ── Write to pending_orders.csv (not trades.csv) ─────────────────────
        # exit_logger promotes this record to trades.csv on confirmed entry fill.
        _append_pending_log({
            "timestamp":       datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "ticker":          ticker,
            "shares":          shares,
            "entry":           round(entry, 4),
            "stop_loss":       round(stop_loss, 4),
            "target":          round(target, 4),
            "risk_per_share":  round(risk_per_shr, 4),
            "risk_aud":        round(risk_aud, 2),
            "order_ref":       order_ref,
            "parent_id":       parent_id,
            "tp_id":           tp_id,
            "sl_id":           sl_id,
            "trigger_count":   t_count,
            "composite_score": round(score, 2),
            "account":         IBKR_ACCOUNT,
            "paper":           PAPER,
        })
        log.info(
            "  %-6s written to pending_orders.csv (awaiting entry fill)",
            symbol,
        )

        submitted += 1

    # ── Disconnect ────────────────────────────────────────────────────────────
    if ib:
        ib.disconnect()
        log.info("Disconnected from TWS.")

    log.info(
        "%s %d bracket order(s) submitted for account %s.",
        "[DRY-RUN]" if dry_run else "Done.", submitted, IBKR_ACCOUNT,
    )
    return submitted


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="IBKR bracket order executor for ASX Swing Engine")
    p.add_argument(
        "--signals",
        type=Path,
        default=SIGNALS_CSV,
        help=f"Path to signals CSV (default: {SIGNALS_CSV})",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be submitted without connecting to TWS",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    log.info("=" * 62)
    log.info(
        "  IBKR EXECUTOR  |  %s  |  account=%s  |  %s",
        "PAPER" if PAPER else "LIVE",
        IBKR_ACCOUNT,
        datetime.today().strftime("%d %b %Y %H:%M"),
    )
    log.info("=" * 62)

    if not PAPER:
        # Extra guard — require explicit env confirmation for live trading
        live_confirm = os.getenv("IBKR_LIVE_CONFIRMED", "").strip().lower()
        if live_confirm != "yes":
            log.error(
                "LIVE trading requires IBKR_LIVE_CONFIRMED=yes in .env. "
                "Set PAPER=True to use paper trading."
            )
            sys.exit(1)

    run_executor(args.signals, dry_run=args.dry_run)
