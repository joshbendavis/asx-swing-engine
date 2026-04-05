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

SIGNALS_CSV      = Path("results/signals_output.csv")
TRADE_LOG        = Path("logs/trades.csv")

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
# Trade log CSV
# ---------------------------------------------------------------------------
TRADE_LOG_COLS = [
    "timestamp", "ticker", "shares", "entry", "stop_loss", "target",
    "risk_per_share", "risk_aud", "order_ref",
    "parent_id", "tp_id", "sl_id",
    "trigger_count", "composite_score", "account", "paper",
]


def _init_trade_log() -> None:
    """Create logs/trades.csv with header row if it doesn't already exist."""
    TRADE_LOG.parent.mkdir(parents=True, exist_ok=True)
    if not TRADE_LOG.exists():
        with open(TRADE_LOG, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=TRADE_LOG_COLS)
            writer.writeheader()
        log.info("Created trade log -> %s", TRADE_LOG)


def _append_trade_log(row: dict) -> None:
    """Append one completed bracket submission to logs/trades.csv."""
    with open(TRADE_LOG, "a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=TRADE_LOG_COLS)
        writer.writerow({col: row.get(col, "") for col in TRADE_LOG_COLS})


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


def _make_contract(symbol: str):
    """Return a qualified ASX Stock contract for the given symbol (no .AX suffix)."""
    from ib_insync import Stock
    return Stock(symbol, "ASX", "AUD")


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
    """
    from ib_insync import LimitOrder, StopOrder

    oca_group = f"{order_ref}-OCA"

    parent = LimitOrder(
        action        = "BUY",
        totalQuantity = shares,
        lmtPrice      = round(entry, 3),
        orderRef      = order_ref,
        account       = IBKR_ACCOUNT,
        transmit      = False,          # hold until children are attached
    )

    take_profit = LimitOrder(
        action        = "SELL",
        totalQuantity = shares,
        lmtPrice      = round(target, 3),
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
        stopPrice     = round(stop_loss, 3),
        orderRef      = order_ref + "-SL",
        ocaGroup      = oca_group,
        ocaType       = 1,
        account       = IBKR_ACCOUNT,
        parentId      = 0,
        transmit      = True,           # transmit=True on last child sends all three
    )

    contract = _make_contract(symbol)
    ib.qualifyContracts(contract)

    # Place parent first to get its orderId
    parent_trade    = ib.placeOrder(contract, parent)
    parent_id       = parent_trade.order.orderId

    # Wire children to parent
    take_profit.parentId      = parent_id
    stop_loss_order.parentId  = parent_id

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

    # ── Load signals ─────────────────────────────────────────────────────────
    if not signals_path.exists():
        log.error("Signals file not found: %s", signals_path)
        return 0

    signals = pd.read_csv(signals_path)
    if signals.empty:
        log.info("No signals in %s — nothing to trade.", signals_path)
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
        balance        = 20_000.0
        open_positions = 0
        log.info("[DRY-RUN] Using simulated balance $%.2f, 0 open positions", balance)
    else:
        balance        = _get_account_balance(ib)
        open_positions = _count_open_positions(ib)
        log.info(
            "Account balance: $%.2f AUD  |  Open positions: %d / %d",
            balance, open_positions, MAX_POSITIONS,
        )

    headroom = MAX_POSITIONS - open_positions
    if headroom <= 0:
        log.info("Portfolio is full (%d/%d positions). No new orders.", open_positions, MAX_POSITIONS)
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
            log.warning("%-6s: invalid risk_per_share (entry=%.3f stop=%.3f) — skipping.", symbol, entry, stop_loss)
            continue

        # Position size
        risk_aud = balance * RISK_PCT
        shares   = int(math.floor(risk_aud / risk_per_shr))

        if shares < MIN_SHARES:
            log.warning(
                "%-6s: sized to %d shares (risk $%.2f / share $%.4f) — below minimum, skipping.",
                symbol, shares, risk_aud, risk_per_shr,
            )
            continue

        order_ref = f"ASX-{today_str}-{symbol}"

        # ── Duplicate guard ───────────────────────────────────────────────────
        if not dry_run and _already_ordered_today(ib, symbol):
            log.info("%-6s: order already submitted today — skipping.", symbol)
            continue

        log.info(
            "%-6s | shares=%-4d | entry=%.3f | stop=%.3f | target=%.3f | "
            "risk=$%.2f | triggers=%d | score=%.1f",
            symbol, shares, entry, stop_loss, target, risk_aud, t_count, score,
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

        # ── Append to trade log ───────────────────────────────────────────────
        _append_trade_log({
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
                "LIVE trading requires IBKR_LIVE_CONFIRMED=yes in .env.  "
                "Set PAPER=True to use paper trading."
            )
            sys.exit(1)

    run_executor(args.signals, dry_run=args.dry_run)
