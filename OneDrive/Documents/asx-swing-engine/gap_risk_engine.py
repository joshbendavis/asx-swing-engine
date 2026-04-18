# -*- coding: utf-8 -*-
"""
gap_risk_engine.py  -  ASX Swing Engine  -  Step 3.5
=====================================================
Overnight gap-risk evaluation: classifies open positions by how much of the
initial risk has been 'paid back', scores their overnight exposure, and
recommends exit / reduce / trail-tighten / hold before the next open.

Pipeline position:  AFTER risk_engine.py  -->  BEFORE ibkr_executor.py

Scoring model
-------------
  Days held >= 5          : +15
  Days held >= 8          : +15  (additional; stacks with >=5)
  ATR% > 4%               : +15
  ATR% > 6%               : +15  (additional; stacks with >4%)
  Regime CHOPPY / BEAR    : +20
  ROC-20d < -2%           : +15  (only genuinely negative momentum, not just slow)
  UNPAID position         : +25
  PARTIALLY_PAID position : +10

Action thresholds
-----------------
  UNPAID   score > 80        : EXIT at today's close
  UNPAID   score 70-80       : REDUCE position 50% (once per position)
  PARTIALLY_PAID  score > 60 : REDUCE 50%
  PARTIALLY_PAID  score 40-60: TIGHTEN trail to 1.8 x ATR
  PAID:  no exit; score > 60 : TIGHTEN trail to 1.8 x ATR

  Minimum bars held before scoring actions fire: 5 bars (grace period)
  Day-N rule: N=12 (raised from 7; system has 20-bar time stop)

Non-negotiable overrides
------------------------
  Day-7 Unpaid Exit : days_held >= 7 AND current_r < 1 -> EXIT regardless of score

Portfolio caps
--------------
  Unpaid heat  <= 3% of account
  Total heat   <= 9% of account  (unchanged from risk_engine.py)

Event risk proxy
----------------
  volume > 2.5x 20-day avg AND daily range > 2x ATR:
    -> block new entries for that ticker
    -> add +20 to gap score for any open position in that ticker

Public API used by backtest_gap.py and run_daily.py
----------------------------------------------------
  classify_position(current_r)            -> str
  calc_gap_score(...)                     -> (int, dict)
  get_action(classification, score, days) -> str
  check_event_risk(vol, avg_vol, rng, atr)-> bool
  run(signals_path, trades_path, regime_path, actions_path, balance) -> bool
"""

from __future__ import annotations

import json
import logging
import os
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger("asx-swing")

# ---------------------------------------------------------------------------
# Scoring constants
# ---------------------------------------------------------------------------
SCORE_DAYS_GE5     = 15
SCORE_DAYS_GE8     = 15   # additional (stacks with GE5 when days >= 8)
SCORE_ATR_GT4      = 15
SCORE_ATR_GT6      = 15   # additional (stacks with GT4 when > 6%)
SCORE_CHOPPY_BEAR  = 20
SCORE_ROC_LOW      = 15
SCORE_UNPAID       = 25
SCORE_PART_PAID    = 10
SCORE_EVENT_RISK   = 20   # applied to open positions in event-flagged tickers

# Action thresholds
UNPAID_EXIT_SCORE     = 80   # UNPAID score > this  -> EXIT
UNPAID_REDUCE_SCORE   = 70   # UNPAID score >= this -> REDUCE 50% (once per position)
PART_EXIT_SCORE       = 60   # PARTIALLY_PAID score > this -> REDUCE 50%
PART_TIGHTEN_SCORE    = 40   # PARTIALLY_PAID score >= this -> TIGHTEN trail
PAID_TIGHTEN_SCORE    = 60   # PAID score > this -> slight TIGHTEN

# Trail tightening
TRAIL_TIGHT_MULT      = 1.8  # replaces default 2.0 when tightening
TRAIL_DEFAULT_MULT    = 2.0

# Portfolio caps
UNPAID_HEAT_CAP_PCT   = 3.0   # % of account
TOTAL_HEAT_CAP_PCT    = 9.0   # % of account (unchanged from risk_engine)

# Day-7 override (raised to 12 — this is a 20-bar time-stop system; 7 bars is too early)
DAY7_BARS_THRESHOLD   = 12    # bars_held >= this AND unpaid -> forced exit

# Event risk thresholds
EVENT_VOL_MULT         = 2.5  # volume > N x avg_vol
EVENT_RANGE_ATR_MULT   = 2.0  # daily range > N x ATR

# Regime sets that trigger the CHOPPY/BEAR score bonus
# Covers both Variant-F 3-state and legacy 5-state naming
CHOPPY_BEAR_REGIMES = frozenset({
    "CHOPPY_BEAR", "CHOPPY", "BEAR", "HIGH_VOL"
})

# ---------------------------------------------------------------------------
# Pure logic (no I/O — importable by backtest)
# ---------------------------------------------------------------------------

def classify_position(current_r: float) -> str:
    """
    UNPAID         : current R < 1   (position has not yet recouped its initial risk)
    PARTIALLY_PAID : 1R <= current R < 2R
    PAID           : current R >= 2R
    """
    if current_r >= 2.0:
        return "PAID"
    elif current_r >= 1.0:
        return "PARTIALLY_PAID"
    return "UNPAID"


def calc_gap_score(
    bars_held:      int,
    atr_pct:        float,
    regime:         str,
    roc_20_pct:     float,   # 20-day ROC expressed as a PERCENTAGE (e.g. 1.5, not 0.015)
    classification: str,
) -> tuple[int, dict]:
    """
    Calculate overnight gap-risk score for a single position.

    Returns (score, breakdown_dict).
    Higher score = higher overnight risk = more aggressive action required.
    """
    score = 0
    breakdown: dict[str, int] = {}

    # Days held (additive tiers)
    if bars_held >= 5:
        score += SCORE_DAYS_GE5
        breakdown["days_ge5"] = SCORE_DAYS_GE5
    if bars_held >= 8:
        score += SCORE_DAYS_GE8
        breakdown["days_ge8"] = SCORE_DAYS_GE8

    # ATR% (additive tiers)
    if atr_pct > 4.0:
        score += SCORE_ATR_GT4
        breakdown["atr_gt4pct"] = SCORE_ATR_GT4
    if atr_pct > 6.0:
        score += SCORE_ATR_GT6
        breakdown["atr_gt6pct"] = SCORE_ATR_GT6

    # Regime
    if regime in CHOPPY_BEAR_REGIMES:
        score += SCORE_CHOPPY_BEAR
        breakdown["choppy_bear"] = SCORE_CHOPPY_BEAR

    # Genuinely negative momentum (ROC-20 in % terms)
    # Threshold is -2.0% (not +2.0%) — only fires when momentum has turned negative,
    # not merely slow. Nearly every new entry has ROC-20 < 2%; that's not gap risk.
    if roc_20_pct < -2.0:
        score += SCORE_ROC_LOW
        breakdown["roc_low"] = SCORE_ROC_LOW

    # Position classification bonus
    if classification == "UNPAID":
        score += SCORE_UNPAID
        breakdown["unpaid"] = SCORE_UNPAID
    elif classification == "PARTIALLY_PAID":
        score += SCORE_PART_PAID
        breakdown["part_paid"] = SCORE_PART_PAID

    return score, breakdown


def get_action(classification: str, score: int, bars_held: int) -> str:
    """
    Determine the recommended action for a single position.

    Returns one of:
        "DAY7_EXIT"   - non-negotiable day-7 forced exit
        "EXIT"        - score-triggered exit at today's close
        "REDUCE"      - sell 50% of position at today's close
        "TIGHTEN"     - tighten trailing stop to 1.8 x ATR
        "HOLD"        - no action required
    """
    # Non-negotiable Day-7 override: trumps everything
    if bars_held >= DAY7_BARS_THRESHOLD and classification == "UNPAID":
        return "DAY7_EXIT"

    if classification == "UNPAID":
        if score > UNPAID_EXIT_SCORE:
            return "EXIT"
        elif score >= UNPAID_REDUCE_SCORE:
            return "REDUCE"
        return "HOLD"

    elif classification == "PARTIALLY_PAID":
        if score > PART_EXIT_SCORE:
            return "REDUCE"
        elif score >= PART_TIGHTEN_SCORE:
            return "TIGHTEN"
        return "HOLD"

    else:  # PAID
        # Never exit on score alone; only tighten trail if score is high
        if score > PAID_TIGHTEN_SCORE:
            return "TIGHTEN"
        return "HOLD"


def check_event_risk(
    volume:    float,
    avg_vol_20: float,
    price_range: float,
    atr:        float,
) -> bool:
    """
    Returns True if today's bar shows event-risk characteristics:
    abnormal volume (>2.5x) AND abnormal range (>2x ATR).
    """
    if avg_vol_20 <= 0 or atr <= 0:
        return False
    return (volume > EVENT_VOL_MULT * avg_vol_20 and
            price_range > EVENT_RANGE_ATR_MULT * atr)


# ---------------------------------------------------------------------------
# Production runner  (reads files, fetches live data, writes actions)
# ---------------------------------------------------------------------------

_ROOT = Path(__file__).parent
TRADES_CSV      = _ROOT / "logs"    / "trades.csv"
SIGNALS_FINAL   = _ROOT / "results" / "signals_final.csv"
REGIME_JSON     = _ROOT / "results" / "regime.json"
ACTIONS_JSON    = _ROOT / "results" / "gap_risk_actions.json"
EQUITY_CSV      = _ROOT / "results" / "equity_curve.csv"


def _load_json(path: Path) -> dict:
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return {}


def _ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False).mean()


def _atr_series(high: pd.Series, low: pd.Series, close: pd.Series, n: int = 14) -> pd.Series:
    pc = close.shift(1)
    tr = pd.concat([high - low, (high - pc).abs(), (low - pc).abs()], axis=1).max(axis=1)
    return tr.ewm(span=n, adjust=False).mean()


def _fetch_indicators(tickers: list[str], lookback_days: int = 65) -> dict[str, dict]:
    """
    Fetch last `lookback_days` calendar days of OHLCV for each ticker.
    Returns {ticker: {close, atr, atr_pct, roc_20_pct, avg_vol_20, last_vol, last_high, last_low}}.
    """
    try:
        import yfinance as yf
    except ImportError:
        log.error("yfinance not available; gap risk engine cannot fetch live data.")
        return {}

    import warnings
    warnings.filterwarnings("ignore")

    out: dict[str, dict] = {}
    start = (date.today() - timedelta(days=lookback_days)).isoformat()

    BATCH = 20
    batches = [tickers[i:i+BATCH] for i in range(0, len(tickers), BATCH)]

    for batch in batches:
        try:
            import sys, contextlib
            with contextlib.redirect_stderr(open(os.devnull, "w")):
                raw = yf.download(batch, start=start, interval="1d",
                                  auto_adjust=True, progress=False, threads=True)
        except Exception as exc:
            log.warning("yfinance download error: %s", exc)
            continue

        if raw.empty:
            continue

        if isinstance(raw.columns, pd.MultiIndex):
            available = raw.columns.get_level_values(1).unique()
        else:
            available = batch[:1] if len(batch) == 1 else []

        for tkr in batch:
            try:
                if isinstance(raw.columns, pd.MultiIndex):
                    if tkr not in available:
                        continue
                    df = raw.xs(tkr, axis=1, level=1).dropna(how="all")
                else:
                    df = raw.dropna(how="all")

                if len(df) < 25:
                    continue

                close  = df["Close"]
                high   = df["High"]
                low    = df["Low"]
                volume = df["Volume"]

                atr_s    = _atr_series(high, low, close)
                avg_vol  = volume.rolling(20).mean()
                roc_20   = close.pct_change(20) * 100

                last = df.index[-1]
                out[tkr] = {
                    "close":       float(close.iloc[-1]),
                    "atr":         float(atr_s.iloc[-1]),
                    "atr_pct":     float(atr_s.iloc[-1] / close.iloc[-1] * 100)
                                   if close.iloc[-1] > 0 else 4.0,
                    "roc_20_pct":  float(roc_20.iloc[-1])
                                   if not pd.isna(roc_20.iloc[-1]) else 0.0,
                    "avg_vol_20":  float(avg_vol.iloc[-1])
                                   if not pd.isna(avg_vol.iloc[-1]) else 0.0,
                    "last_vol":    float(volume.iloc[-1]),
                    "last_high":   float(high.iloc[-1]),
                    "last_low":    float(low.iloc[-1]),
                    "last_date":   str(last.date()),
                }
            except Exception as exc:
                log.debug("Indicator computation failed for %s: %s", tkr, exc)

    return out


def _open_trades(trades_path: Path, equity_path: Path) -> pd.DataFrame:
    """Return trades that have not yet been closed (not in equity_curve)."""
    try:
        trades = pd.read_csv(trades_path)
    except Exception:
        return pd.DataFrame()

    closed_refs: set[str] = set()
    if equity_path.exists():
        try:
            eq = pd.read_csv(equity_path)
            if "order_ref" in eq.columns:
                closed_refs = set(eq["order_ref"].dropna())
        except Exception:
            pass

    open_mask = ~trades["order_ref"].isin(closed_refs) if "order_ref" in trades.columns else pd.Series(True, index=trades.index)
    return trades[open_mask].copy()


def run(
    signals_path: Path = SIGNALS_FINAL,
    trades_path:  Path = TRADES_CSV,
    regime_path:  Path = REGIME_JSON,
    actions_path: Path = ACTIONS_JSON,
    equity_path:  Path = EQUITY_CSV,
    balance:      float = 20_000.0,
) -> bool:
    """
    Production entry point.  Called by run_daily.py as Step 3.5.

    1. Evaluate every open position -> gap_risk_actions.json
    2. Filter signals_final.csv:
       - Remove entries in event-risk tickers
       - Enforce unpaid heat cap (no new entries if unpaid heat >= cap)
    3. Write updated signals_final.csv and gap_risk_actions.json

    Returns True on success, False if no open positions or missing data.
    """
    log.info("=" * 60)
    log.info("  GAP RISK ENGINE  |  Step 3.5")
    log.info("=" * 60)

    # ── Load regime ───────────────────────────────────────────────────────────
    regime_data  = _load_json(regime_path)
    f_regime     = regime_data.get("f_regime", regime_data.get("regime", "STRONG_BULL"))
    log.info("Regime: %s", f_regime)

    # ── Load open positions ───────────────────────────────────────────────────
    open_df = _open_trades(trades_path, equity_path)
    if open_df.empty:
        log.info("No open positions found — gap risk engine skipping position evaluation.")
        open_tickers = []
    else:
        open_tickers = open_df["ticker"].str.replace(".AX", ".AX", regex=False).tolist()
        log.info("Open positions: %s", ", ".join(open_tickers))

    # ── Load new-entry signals ────────────────────────────────────────────────
    try:
        signals_df = pd.read_csv(signals_path)
    except Exception:
        signals_df = pd.DataFrame()

    new_tickers = signals_df["ticker"].tolist() if not signals_df.empty and "ticker" in signals_df.columns else []

    # ── Fetch indicators for all relevant tickers ─────────────────────────────
    all_tickers = list(dict.fromkeys(open_tickers + new_tickers))
    if not all_tickers:
        log.info("No tickers to evaluate.")
        return False

    log.info("Fetching indicators for %d tickers ...", len(all_tickers))
    indicators = _fetch_indicators(all_tickers)
    log.info("Indicator fetch complete (%d/%d tickers).", len(indicators), len(all_tickers))

    # ── Evaluate open positions ───────────────────────────────────────────────
    actions: list[dict] = []
    event_risk_tickers: set[str] = set()
    total_heat    = 0.0
    unpaid_heat   = 0.0

    # First pass: event risk flags (used to add score bonus)
    for tkr, ind in indicators.items():
        if check_event_risk(ind["last_vol"], ind["avg_vol_20"],
                            ind["last_high"] - ind["last_low"], ind["atr"]):
            event_risk_tickers.add(tkr)
            log.info("EVENT RISK flagged: %s  (vol=%.0f  avg=%.0f  rng=%.3f  atr=%.3f)",
                     tkr, ind["last_vol"], ind["avg_vol_20"],
                     ind["last_high"] - ind["last_low"], ind["atr"])

    # Second pass: score each open position
    for _, row in open_df.iterrows():
        tkr = row["ticker"]
        ind = indicators.get(tkr)
        if ind is None:
            log.warning("%s: no indicator data — HOLD by default", tkr)
            actions.append({"ticker": tkr, "action": "HOLD", "score": 0,
                            "classification": "UNKNOWN", "breakdown": {}})
            continue

        # Current R
        try:
            entry_p  = float(row["entry"])
            init_r   = float(row.get("risk_per_share", row.get("initial_risk", 0)))
            if init_r <= 0:
                # Derive from stop_loss
                stop_p = float(row["stop_loss"])
                init_r = abs(entry_p - stop_p)
        except Exception:
            log.warning("%s: cannot determine initial_risk — HOLD", tkr)
            actions.append({"ticker": tkr, "action": "HOLD", "score": 0,
                            "classification": "UNKNOWN", "breakdown": {}})
            continue

        current_r = (ind["close"] - entry_p) / init_r if init_r > 0 else 0.0

        # Days held (trading days since entry timestamp)
        try:
            entry_ts   = pd.Timestamp(row["timestamp"])
            bars_held  = int(np.busday_count(entry_ts.date(), date.today()))
        except Exception:
            bars_held = 0

        classification = classify_position(current_r)
        score, breakdown = calc_gap_score(
            bars_held, ind["atr_pct"], f_regime, ind["roc_20_pct"], classification
        )

        # Event risk bonus
        if tkr in event_risk_tickers:
            score     += SCORE_EVENT_RISK
            breakdown["event_risk"] = SCORE_EVENT_RISK

        action = get_action(classification, score, bars_held)

        # Heat tracking
        try:
            risk_aud = float(row.get("risk_aud", init_r * 100))
        except Exception:
            risk_aud = 300.0

        total_heat  += risk_aud
        if classification == "UNPAID":
            unpaid_heat += risk_aud

        log.info(
            "%s  close=%.3f  curr_R=%.2f  %-15s  days=%d  "
            "atr%%=%.1f  score=%d  -> %s",
            tkr, ind["close"], current_r, classification,
            bars_held, ind["atr_pct"], score, action
        )

        actions.append({
            "ticker":         tkr,
            "action":         action,
            "score":          score,
            "classification": classification,
            "current_r":      round(current_r, 3),
            "bars_held":      bars_held,
            "atr_pct":        round(ind["atr_pct"], 2),
            "roc_20_pct":     round(ind["roc_20_pct"], 2),
            "regime":         f_regime,
            "breakdown":      breakdown,
            "current_price":  ind["close"],
            "event_risk":     tkr in event_risk_tickers,
        })

    # ── Portfolio heat cap enforcement ────────────────────────────────────────
    unpaid_heat_pct = unpaid_heat / balance * 100 if balance > 0 else 0.0
    total_heat_pct  = total_heat  / balance * 100 if balance > 0 else 0.0

    log.info("Portfolio heat:  unpaid=%.1f%%  total=%.1f%%  "
             "(caps: %.0f%% / %.0f%%)",
             unpaid_heat_pct, total_heat_pct,
             UNPAID_HEAT_CAP_PCT, TOTAL_HEAT_CAP_PCT)

    if unpaid_heat_pct > UNPAID_HEAT_CAP_PCT:
        log.warning(
            "Unpaid heat %.1f%% exceeds %.0f%% cap — "
            "escalating highest-scored UNPAID positions to EXIT.",
            unpaid_heat_pct, UNPAID_HEAT_CAP_PCT
        )
        # Sort UNPAID by score desc; escalate until under cap
        unpaid_actions = sorted(
            [a for a in actions if a["classification"] == "UNPAID"],
            key=lambda x: x["score"], reverse=True
        )
        for a in unpaid_actions:
            if unpaid_heat_pct <= UNPAID_HEAT_CAP_PCT:
                break
            if a["action"] not in ("EXIT", "DAY7_EXIT"):
                a["action"]    = "EXIT"
                a["escalated"] = True
                log.warning("  %s escalated to EXIT (heat cap)", a["ticker"])
                # Approximate heat reduction
                row2 = open_df[open_df["ticker"] == a["ticker"]]
                if not row2.empty:
                    try:
                        ra = float(row2.iloc[0].get("risk_aud", 300))
                    except Exception:
                        ra = 300.0
                    unpaid_heat     -= ra
                    unpaid_heat_pct  = unpaid_heat / balance * 100

    # ── Filter new-entry signals ──────────────────────────────────────────────
    n_before = len(signals_df)
    blocked_new: list[str] = []

    if not signals_df.empty and "ticker" in signals_df.columns:
        # 1. Block entries in event-risk tickers
        event_mask = signals_df["ticker"].isin(event_risk_tickers)
        if event_mask.any():
            blocked_new.extend(signals_df.loc[event_mask, "ticker"].tolist())
            signals_df = signals_df[~event_mask].copy()
            log.info("Blocked %d new entry candidate(s) due to event risk: %s",
                     event_mask.sum(), list(signals_df.loc[event_mask, "ticker"]) if not event_mask.all() else blocked_new)

        # 2. Block new entries if unpaid heat is already at/near cap
        if unpaid_heat_pct >= UNPAID_HEAT_CAP_PCT and not signals_df.empty:
            log.warning(
                "Unpaid heat %.1f%% at/above cap — blocking ALL new entries.",
                unpaid_heat_pct
            )
            blocked_new.extend(signals_df["ticker"].tolist())
            signals_df = signals_df.iloc[0:0].copy()   # empty but keep columns

    log.info("New entries: %d -> %d after gap risk filter (%d blocked: %s)",
             n_before, len(signals_df), len(blocked_new), blocked_new)

    # ── Write outputs ─────────────────────────────────────────────────────────
    try:
        actions_path.parent.mkdir(parents=True, exist_ok=True)
        with open(actions_path, "w") as fh:
            json.dump({
                "run_date":            date.today().isoformat(),
                "regime":              f_regime,
                "unpaid_heat_pct":     round(unpaid_heat_pct, 2),
                "total_heat_pct":      round(total_heat_pct, 2),
                "event_risk_tickers":  list(event_risk_tickers),
                "blocked_new_entries": blocked_new,
                "positions":           actions,
            }, fh, indent=2)
        log.info("Gap risk actions -> %s", actions_path)
    except Exception as exc:
        log.warning("Could not write actions JSON: %s", exc)

    try:
        signals_df.to_csv(signals_path, index=False)
        log.info("Signals updated -> %s  (%d entries remain)", signals_path, len(signals_df))
    except Exception as exc:
        log.warning("Could not update signals CSV: %s", exc)

    # ── Summary log ──────────────────────────────────────────────────────────
    action_summary = {}
    for a in actions:
        action_summary[a["action"]] = action_summary.get(a["action"], 0) + 1
    log.info("Position actions: %s", action_summary)

    exits_needed = [a["ticker"] for a in actions if a["action"] in ("EXIT", "DAY7_EXIT")]
    reduces_needed = [a["ticker"] for a in actions if a["action"] == "REDUCE"]
    tightens_needed = [a["ticker"] for a in actions if a["action"] == "TIGHTEN"]

    if exits_needed:
        log.warning("EXIT before close: %s", exits_needed)
    if reduces_needed:
        log.info("REDUCE 50%% at close: %s", reduces_needed)
    if tightens_needed:
        log.info("TIGHTEN trail to 1.8xATR: %s", tightens_needed)

    return True


# ---------------------------------------------------------------------------
# CLI convenience
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%H:%M:%S",
    )
    # Optional: pass --balance XXXX
    bal = 20_000.0
    for i, arg in enumerate(sys.argv[1:], 1):
        if arg == "--balance" and i < len(sys.argv) - 1:
            try:
                bal = float(sys.argv[i + 1])
            except ValueError:
                pass

    success = run(balance=bal)
    sys.exit(0 if success else 1)
