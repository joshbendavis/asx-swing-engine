"""
utils/telegram_bot.py
---------------------
Telegram daily summary for the ASX Swing Engine.

Usage (called automatically by run_daily.py):
    from utils.telegram_bot import send_daily_summary
    send_daily_summary(pipeline_context)

Requires in .env:
    TELEGRAM_BOT_TOKEN   — from @BotFather
    TELEGRAM_CHAT_ID     — your personal chat ID (get via @userinfobot)
"""

import json
import logging
import math
import os
from datetime import date
from pathlib import Path

import requests
from dotenv import load_dotenv

# Load .env so PAPER_BALANCE_OVERRIDE (and Telegram credentials) are available
# even when this module is invoked standalone rather than via run_daily.py.
load_dotenv()

log = logging.getLogger("asx-swing")

# ---------------------------------------------------------------------------
# Paths (relative to project root, same as rest of engine)
# ---------------------------------------------------------------------------
_ROOT        = Path(__file__).parent.parent
TRADES_CSV   = _ROOT / "logs"    / "trades.csv"
PENDING_CSV  = _ROOT / "logs"    / "pending_orders.csv"
REGIME_JSON  = _ROOT / "results" / "regime.json"
EQUITY_CSV   = _ROOT / "results" / "equity_curve.csv"
COOLDOWN_JSON= _ROOT / "results" / "kill_cooldown.json"

STARTING_BALANCE = float(os.getenv("PAPER_BALANCE_OVERRIDE", "20000"))
HEAT_BUDGET      = 9.0   # % — matches dashboard constant
MAX_POSITIONS    = 6

# ---------------------------------------------------------------------------
# Telegram API
# ---------------------------------------------------------------------------
_TG_API = "https://api.telegram.org/bot{token}/sendMessage"


def _send(token: str, chat_id: str, text: str) -> bool:
    """POST a message to Telegram. Returns True on success."""
    url = _TG_API.format(token=token)
    payload = {
        "chat_id":    chat_id,
        "text":       text,
        "parse_mode": "HTML",
        "disable_web_page_preview": True,
    }
    try:
        resp = requests.post(url, json=payload, timeout=15)
        if resp.status_code == 200:
            return True
        log.warning("Telegram API returned %d: %s", resp.status_code, resp.text[:200])
        return False
    except requests.RequestException as exc:
        log.warning("Telegram send failed: %s", exc)
        return False


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def _load_json(path: Path) -> dict:
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return {}


def _load_csv(path: Path):
    """Return list of dicts from CSV, empty list on any error."""
    try:
        import csv
        with open(path, newline="", encoding="utf-8") as f:
            return list(csv.DictReader(f))
    except Exception:
        return []


def _fetch_live_prices(tickers: list[str]) -> dict:
    """yfinance spot prices — returns {} gracefully if unavailable."""
    if not tickers:
        return {}
    try:
        import yfinance as yf
        data = yf.download(tickers, period="1d", progress=False, auto_adjust=True)
        if data.empty:
            return {}
        close = data["Close"] if "Close" in data.columns else data
        out = {}
        if hasattr(close, "columns"):
            for t in tickers:
                if t in close.columns:
                    s = close[t].dropna()
                    if not s.empty:
                        out[t] = float(s.iloc[-1])
        elif len(tickers) == 1:
            s = close.dropna()
            if not s.empty:
                out[tickers[0]] = float(s.iloc[-1])
        return out
    except Exception:
        return {}


def _regime_emoji(regime: str) -> str:
    return {
        "STRONG_BULL": "🟢",
        "WEAK_BULL":   "🟡",
        "CHOPPY_BEAR": "🟠",
        "CHOPPY":      "🟠",
        "BEAR":        "🔴",
        "HIGH_VOL":    "🟣",
    }.get(regime.upper() if regime else "", "⚪")


def _step_icon(status) -> str:
    if status == "ok":   return "✅"
    if status == "warn": return "⚠️"
    if status == "error":return "❌"
    return "⬜"


# ---------------------------------------------------------------------------
# Kill conditions (mirrors dashboard._check_kill_conditions logic)
# ---------------------------------------------------------------------------

def _kill_conditions_clear() -> tuple[bool, list[str]]:
    """
    Quick check: read cooldown.json to see if any kill condition is active.
    Returns (all_clear: bool, triggered_conditions: list[str]).
    """
    cd = _load_json(COOLDOWN_JSON)
    if cd.get("active") and cd.get("conditions"):
        return False, cd["conditions"]
    return True, []


# ---------------------------------------------------------------------------
# Open positions + P&L
# ---------------------------------------------------------------------------

def _open_positions_block() -> tuple[str, float, float]:
    """
    Returns (formatted HTML block, heat_pct, live_pnl_aud).
    """
    trades  = _load_csv(TRADES_CSV)
    equity  = _load_csv(EQUITY_CSV)
    pending = _load_csv(PENDING_CSV)

    # Closed order refs (exits recorded in equity_curve.csv)
    closed_refs = {r["order_ref"] for r in equity if "order_ref" in r}

    # Open = in trades.csv but NOT in equity_curve.csv
    open_trades = [t for t in trades if t.get("order_ref") not in closed_refs]

    if not open_trades:
        return "  <i>No open positions</i>", 0.0, 0.0

    # Fetch live prices
    tickers = [t["ticker"] for t in open_trades if t.get("ticker")]
    prices  = _fetch_live_prices(tickers)

    total_risk = 0.0
    total_unrl = 0.0
    lines      = []

    for t in open_trades:
        ticker = t.get("ticker", "?")
        try:
            shares = int(float(t.get("shares", 0)))
            entry  = float(t.get("entry", 0))
            stop   = float(t.get("stop_loss", 0))
            risk   = float(t.get("risk_aud", 0))
        except (ValueError, TypeError):
            shares = entry = stop = risk = 0

        total_risk += risk

        live = prices.get(ticker, 0.0)
        if live > 0 and shares > 0:
            unrl = (live - entry) * shares
            pct  = (live - entry) / entry * 100 if entry else 0
            sign = "+" if unrl >= 0 else ""
            icon = "🟢" if unrl >= 0 else "🔴"
            pnl_str = f"{sign}${unrl:,.0f} ({sign}{pct:.1f}%)"
        else:
            unrl    = 0.0
            icon    = "⚪"
            pnl_str = "price N/A"

        total_unrl += unrl
        lines.append(f"  {icon} <b>{ticker}</b>  {shares:,} @ ${entry:.3f}  →  {pnl_str}")

    heat_pct = total_risk / STARTING_BALANCE * 100 if STARTING_BALANCE > 0 else 0.0

    # Realised P&L from equity_curve
    realised = 0.0
    for r in equity:
        try:
            realised += float(r.get("pnl_aud", 0))
        except (ValueError, TypeError):
            pass

    total_pnl = realised + total_unrl

    return "\n".join(lines), heat_pct, total_pnl


# ---------------------------------------------------------------------------
# Main public function
# ---------------------------------------------------------------------------

def send_daily_summary(pipeline_context: dict | None = None) -> bool:
    """
    Compose and send the daily summary Telegram message.

    pipeline_context (optional dict) can include:
        regime        str   — e.g. "STRONG_BULL"
        f_regime      str   — Variant F regime label
        confidence    int   — regime confidence %
        psm           float — position size multiplier
        screener_n    int   — stocks passing screener
        signals_n     int   — signals found
        orders_n      int   — orders placed
        steps         dict  — {step_name: "ok"|"warn"|"error"|None}
        run_date      str   — YYYY-MM-DD
        run_time      str   — HH:MM
    """
    token   = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
    chat_id = os.getenv("TELEGRAM_CHAT_ID",   "").strip()

    if not token or not chat_id:
        log.info("Telegram: TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID not set — skipping.")
        return False

    ctx = pipeline_context or {}

    # ── Regime ───────────────────────────────────────────────────────────────
    regime_json = _load_json(REGIME_JSON)
    regime      = ctx.get("f_regime") or regime_json.get("f_regime") or regime_json.get("regime", "UNKNOWN")
    psm         = ctx.get("psm") or regime_json.get("f_regime_size_multiplier", 1.0)
    confidence  = ctx.get("confidence") or regime_json.get("confidence", 0)
    r_emoji     = _regime_emoji(regime)

    # ── Pipeline steps ────────────────────────────────────────────────────────
    steps = ctx.get("steps", {})
    step_names = [
        ("regime",      "Regime"),
        ("screener",    "Screener"),
        ("signals",     "Signals"),
        ("risk",        "Risk engine"),
        ("ibkr",        "IBKR"),
        ("exit_logger", "Exit logger"),
    ]
    steps_lines = "  ".join(
        f"{_step_icon(steps.get(k))} {label}"
        for k, label in step_names
    )

    # ── Signal / order counts ─────────────────────────────────────────────────
    screener_n = ctx.get("screener_n", "—")
    signals_n  = ctx.get("signals_n",  "—")
    orders_n   = ctx.get("orders_n",   "—")

    # ── Open positions ────────────────────────────────────────────────────────
    pos_block, heat_pct, live_pnl = _open_positions_block()

    # ── Heat colour indicator ─────────────────────────────────────────────────
    heat_ratio = heat_pct / HEAT_BUDGET if HEAT_BUDGET > 0 else 0
    heat_icon  = "🔴" if heat_ratio >= 0.8 else "🟡" if heat_ratio >= 0.6 else "🟢"

    # ── Kill conditions ───────────────────────────────────────────────────────
    all_clear, kill_list = _kill_conditions_clear()
    if all_clear:
        kill_line = "✅ All kill conditions clear"
    else:
        kill_line = f"🛑 {len(kill_list)} kill condition(s) triggered"

    # ── Portfolio P&L ─────────────────────────────────────────────────────────
    pnl_sign = "+" if live_pnl >= 0 else ""
    pnl_pct  = live_pnl / STARTING_BALANCE * 100 if STARTING_BALANCE > 0 else 0.0
    pnl_icon = "🟢" if live_pnl >= 0 else "🔴"

    # ── Date/time ─────────────────────────────────────────────────────────────
    run_date = ctx.get("run_date") or date.today().isoformat()
    run_time = ctx.get("run_time", "")

    # ── Compose message ───────────────────────────────────────────────────────
    lines = [
        f"<b>📈 ASX Swing Engine — {run_date}</b>  <i>{run_time}</i>",
        "",
        f"{pnl_icon} <b>P&L  {pnl_sign}${abs(live_pnl):,.0f}  ({pnl_sign}{pnl_pct:.2f}%)</b>",
        "",
        f"<b>REGIME</b>  {r_emoji} <b>{regime.replace('_', ' ')}</b>"
        f"  ·  size ×{psm:.1f}  ·  conf {confidence}%",
        "",
        "<b>PIPELINE</b>",
        f"  {steps_lines}",
        "",
        "<b>ACTIVITY</b>",
        f"  Screener passed: <b>{screener_n}</b>",
        f"  Signals found:   <b>{signals_n}</b>",
        f"  Orders placed:   <b>{orders_n}</b>",
        "",
        f"<b>OPEN POSITIONS</b>  ({len([l for l in pos_block.splitlines() if l.strip().startswith(('🟢','🔴','⚪'))])}/{MAX_POSITIONS} slots)",
        pos_block,
        "",
        f"<b>HEAT</b>  {heat_icon} <b>{heat_pct:.1f}%</b> of {HEAT_BUDGET:.0f}% budget",
        "",
        f"<b>KILL CONDITIONS</b>  {kill_line}",
    ]

    # Append kill condition details if triggered
    if not all_clear:
        for cond in kill_list[:5]:   # cap at 5 to keep message tidy
            # Truncate long condition strings
            short = cond[:120] + "…" if len(cond) > 120 else cond
            lines.append(f"  ⛔ {short}")

    message = "\n".join(lines)

    ok = _send(token, chat_id, message)
    if ok:
        log.info("Telegram daily summary sent.")
    else:
        log.warning("Telegram daily summary failed — check token/chat_id in .env.")
    return ok
