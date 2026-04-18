# -*- coding: utf-8 -*-
"""
backtest_gap.py  -  ASX Swing Engine
=====================================
Adds the gap_risk_engine (Step 3.5) layer to the Variant-F trailing-stop
backtest and compares two scenarios side-by-side:

  F-Trail          : Variant-F 2xATR trailing stop (baseline, no gap risk)
  F-Trail + GapRisk: Same trail logic + gap risk engine running each bar

Gap risk engine logic (per bar, end-of-day):
  - Classify each open position: UNPAID / PARTIALLY_PAID / PAID
  - Score overnight exposure (days held, ATR%, regime, ROC-20d, classification)
  - Apply event-risk bonus for abnormal volume+range bars
  - Take action: EXIT / DAY7_EXIT / REDUCE / TIGHTEN / HOLD
  - Enforce unpaid-heat portfolio cap (3% of account)
  - Block new entries in event-risk tickers

Verification metrics to check:
  - Gap stops before and after (must decrease)
  - Profit factor before and after (must stay > 1.6)
  - Avg R before and after (must not collapse)
  - Best trade (LYC +4.576R) must survive

Usage:
    python backtest_gap.py
    python backtest_gap.py --period 9y
"""

from __future__ import annotations

import argparse
import contextlib
import os
import sys
import time
import warnings
from collections import defaultdict
from datetime import date
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")

# Import pure gap-risk scoring functions (no I/O)
from gap_risk_engine import (
    classify_position,
    calc_gap_score,
    get_action,
    check_event_risk,
    TRAIL_TIGHT_MULT,
    TRAIL_DEFAULT_MULT,
    UNPAID_HEAT_CAP_PCT,
    CHOPPY_BEAR_REGIMES,
    SCORE_EVENT_RISK,
)

# ---------------------------------------------------------------------------
# Constants (mirror live engine — identical to backtest_trail.py)
# ---------------------------------------------------------------------------
STARTING_BALANCE  = 20_000.0
RISK_PCT          = 0.015
MAX_POSITIONS     = 6
ATR_STOP_MULT     = 2.0
TARGET_R          = 2.0
BREAKEVEN_R       = 1.0
TRAIL_ATR_MULT    = 2.0
TRAIL_ACTIVATION_R = 1.0
TIME_STOP_DAYS    = 20
COMMISSION_ENTRY  = 6.0
COMMISSION_EXIT   = 6.0
SLIPPAGE_PCT      = 0.002
MIN_RISK_AUD      = 75.0
MIN_ATR_PCT       = 3.5
EMA_FAST, EMA_SLOW = 50, 200
RS_PERIOD          = 63
WARMUP             = EMA_SLOW + RS_PERIOD + 10
BATCH_SIZE         = 20
BATCH_DELAY        = 1.0
BENCHMARK          = "^AXJO"
RESULTS_DIR        = Path("results/backtest")

ASX200_SAMPLE = [
    "BHP.AX","CBA.AX","CSL.AX","NAB.AX","WBC.AX","ANZ.AX","WES.AX","MQG.AX",
    "RIO.AX","WOW.AX","GMG.AX","TLS.AX","FMG.AX","REA.AX","ALL.AX","S32.AX",
    "STO.AX","ORI.AX","AMC.AX","TCL.AX","QBE.AX","IAG.AX","SHL.AX","COL.AX",
    "ASX.AX","MPL.AX","APA.AX","JHX.AX","SGP.AX","MIN.AX","IGO.AX","PLS.AX",
    "LYC.AX","WHC.AX","NXT.AX","ALX.AX","ALD.AX","NCM.AX","AGL.AX","SUL.AX",
]
ASX_PRESET = [
    "BHP.AX","CBA.AX","CSL.AX","NAB.AX","WBC.AX","ANZ.AX","WES.AX",
    "MQG.AX","RIO.AX","TLS.AX","WOW.AX","GMG.AX","FMG.AX",
    "REA.AX","TCL.AX","COL.AX","IAG.AX","QBE.AX","ORG.AX",
    "APA.AX","AMC.AX","AGL.AX","MPL.AX","RHC.AX","JHX.AX",
    "SEK.AX","TWE.AX","BXB.AX","IPL.AX","ORI.AX",
    "SHL.AX","CPU.AX","MIN.AX","ILU.AX","DXS.AX","GPT.AX",
    "SGP.AX","CHC.AX","MGR.AX","CWY.AX","BEN.AX","BOQ.AX",
    "SUN.AX","HVN.AX","JBH.AX","SUL.AX","ARB.AX",
    "NXT.AX","PME.AX","WTC.AX","XRO.AX","ALU.AX","CAR.AX","REH.AX",
    "SFR.AX","EVN.AX","NST.AX","RRL.AX","LYC.AX","PLS.AX","IGO.AX",
    "HUB.AX","IEL.AX","TNE.AX","GNC.AX","ELD.AX","NUF.AX",
    "MTS.AX","PTM.AX","MFG.AX","GQG.AX","PPT.AX",
    "WHC.AX","BPT.AX","AZJ.AX","WDS.AX","S32.AX","AWC.AX","NHF.AX",
    "ALX.AX","ABP.AX","NSR.AX","BWP.AX","VCX.AX","SCG.AX",
    "TLC.AX","A2M.AX","KGN.AX","TAH.AX","WEB.AX",
    "DRO.AX","EGH.AX","CBO.AX","DBI.AX","MAH.AX","LTR.AX",
    "SRL.AX","OBL.AX","PXA.AX","SMR.AX","STO.AX",
    "ALL.AX","ASX.AX","ALD.AX","NCM.AX",
]

# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

@contextlib.contextmanager
def _quiet():
    with open(os.devnull, "w") as devnull:
        old = sys.stderr
        sys.stderr = devnull
        try:
            yield
        finally:
            sys.stderr = old


def _ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False).mean()


def _rsi(close: pd.Series, n: int = 14) -> pd.Series:
    d = close.diff()
    g = d.clip(lower=0).ewm(alpha=1/n, adjust=False).mean()
    l = (-d).clip(lower=0).ewm(alpha=1/n, adjust=False).mean()
    return 100 - 100 / (1 + g / l.replace(0, np.nan))


def _atr(high: pd.Series, low: pd.Series, close: pd.Series, n: int = 14) -> pd.Series:
    pc = close.shift(1)
    tr = pd.concat([high - low, (high - pc).abs(), (low - pc).abs()], axis=1).max(axis=1)
    return tr.ewm(span=n, adjust=False).mean()


# ---------------------------------------------------------------------------
# Data download
# ---------------------------------------------------------------------------

def _download_batch(tickers: list[str], period: str,
                    min_rows: int = 0) -> dict[str, pd.DataFrame]:
    result: dict[str, pd.DataFrame] = {}
    with _quiet():
        raw = yf.download(tickers, period=period, interval="1d",
                          auto_adjust=True, progress=False, threads=True)
    if raw.empty:
        return result
    if isinstance(raw.columns, pd.MultiIndex):
        available = raw.columns.get_level_values(1).unique()
        for t in tickers:
            if t in available:
                df = raw.xs(t, axis=1, level=1).dropna(how="all")
                if len(df) >= max(min_rows, 1):
                    result[t] = df
    else:
        if len(tickers) == 1:
            df = raw.dropna(how="all")
            if len(df) >= max(min_rows, 1):
                result[tickers[0]] = df
    return result


def download_universe(tickers: list[str], period: str) -> dict[str, pd.DataFrame]:
    data: dict[str, pd.DataFrame] = {}
    batches = [tickers[i:i+BATCH_SIZE] for i in range(0, len(tickers), BATCH_SIZE)]
    for idx, batch in enumerate(batches, 1):
        if idx % 3 == 0 or idx == len(batches):
            print(f"  Batch {idx}/{len(batches)} ...")
        data.update(_download_batch(batch, period, min_rows=WARMUP))
        if idx < len(batches):
            time.sleep(BATCH_DELAY)
    return data


# ---------------------------------------------------------------------------
# Historical Variant-F regime series
# ---------------------------------------------------------------------------

def build_regime_series(xjo_df: pd.DataFrame, period: str) -> pd.DataFrame:
    print("Building Variant-F regime series ...")
    close = xjo_df["Close"]
    high  = xjo_df["High"]
    low   = xjo_df["Low"]
    ema200      = _ema(close, 200)
    ema50       = _ema(close, 50)
    atr_s       = _atr(high, low, close, 14)
    above_200   = close > ema200
    atr_pct     = atr_s / close * 100
    roc_20      = close.pct_change(20) * 100
    ema50_slope = (ema50 - ema50.shift(5)) / ema50.shift(5) * 100

    print(f"  Downloading {len(ASX200_SAMPLE)} breadth tickers ...")
    breadth_data: dict[str, pd.DataFrame] = {}
    b_batches = [ASX200_SAMPLE[i:i+BATCH_SIZE] for i in range(0, len(ASX200_SAMPLE), BATCH_SIZE)]
    for b_idx, batch in enumerate(b_batches, 1):
        breadth_data.update(_download_batch(batch, period, min_rows=50))
        if b_idx < len(b_batches):
            time.sleep(BATCH_DELAY)

    all_dates = close.index
    breadth_above = pd.DataFrame(index=all_dates, dtype=float)
    for ticker, bdf in breadth_data.items():
        bc   = bdf["Close"].reindex(all_dates)
        bema = _ema(bc.ffill(), 50)
        breadth_above[ticker] = (bc > bema).astype(float)
        breadth_above.loc[bc.isna(), ticker] = np.nan
    market_breadth = breadth_above.mean(axis=1) * 100

    rows = []
    for dt in close.index:
        a200  = bool(above_200.loc[dt])
        brd   = float(market_breadth.loc[dt]) if not pd.isna(market_breadth.loc[dt]) else 50.0
        roc_v = float(roc_20.loc[dt])          if not pd.isna(roc_20.loc[dt])         else 0.0
        slop  = float(ema50_slope.loc[dt])     if not pd.isna(ema50_slope.loc[dt])    else 0.0
        atp   = float(atr_pct.loc[dt])         if not pd.isna(atr_pct.loc[dt])        else 0.0

        # Legacy 5-state (for gap score regime check)
        if atp > 2.5:
            regime = "HIGH_VOL"
        elif not a200:
            regime = "BEAR"
        elif abs(slop) < 0.1 and brd < 50.0:
            regime = "CHOPPY"
        elif roc_v < 0 or slop < 0:
            regime = "WEAK_BULL"
        else:
            regime = "BULL"

        # Variant-F 3-state
        if a200 and brd > 55:
            f_reg = "STRONG_BULL"
        elif a200 and brd >= 40:
            f_reg = "WEAK_BULL"
        else:
            f_reg = "CHOPPY_BEAR"

        rows.append({"date": dt, "regime": regime, "f_regime": f_reg,
                     "market_breadth": brd})

    df  = pd.DataFrame(rows).set_index("date")
    frc = df["f_regime"].value_counts()
    total_td = len(df)
    print("  F-regime: ", end="")
    for fr in ["STRONG_BULL", "WEAK_BULL", "CHOPPY_BEAR"]:
        n = int(frc.get(fr, 0))
        print(f"{fr} {n} ({n/total_td*100:.0f}%)  ", end="")
    print()
    return df


# ---------------------------------------------------------------------------
# Indicators + signals
# ---------------------------------------------------------------------------

def compute_indicators(df: pd.DataFrame,
                       xjo_close: pd.Series,
                       xjo_uptrend: pd.Series) -> pd.DataFrame:
    d = df.copy()
    d["ema50"]       = _ema(d["Close"], EMA_FAST)
    d["ema200"]      = _ema(d["Close"], EMA_SLOW)
    d["rsi"]         = _rsi(d["Close"])
    d["atr"]         = _atr(d["High"], d["Low"], d["Close"])
    d["avg_vol"]     = d["Volume"].rolling(20).mean()
    d["ema50_dist"]  = (d["Close"] / d["ema50"] - 1) * 100
    d["atr_pct"]     = d["atr"] / d["Close"] * 100
    d["stock_r63"]   = d["Close"].pct_change(RS_PERIOD)
    xjo_r63          = xjo_close.pct_change(RS_PERIOD)
    d["xjo_r63"]     = xjo_r63.reindex(d.index, method="ffill")
    d["rs_vs_xjo"]   = d["stock_r63"] - d["xjo_r63"]
    d["xjo_uptrend"] = xjo_uptrend.reindex(d.index, method="ffill").fillna(True)
    d["roc_20"]      = d["Close"].pct_change(20)
    d["ema50_slope"] = (d["ema50"] - d["ema50"].shift(5)) / d["ema50"].shift(5).replace(0, np.nan)
    rsi_prev  = d["rsi"].shift(1)
    t_rsi     = (rsi_prev >= 40) & (rsi_prev <= 55) & (d["rsi"] > rsi_prev)
    macd_line = _ema(d["Close"], 12) - _ema(d["Close"], 26)
    macd_sig  = _ema(macd_line, 9)
    t_macd    = (macd_line.shift(1) < macd_sig.shift(1)) & (macd_line > macd_sig)
    t_vol     = d["Volume"] >= 1.5 * d["avg_vol"]
    d["trigger_count"] = t_rsi.astype(int) + t_macd.astype(int) + t_vol.astype(int)
    return d


def find_signals(d: pd.DataFrame) -> pd.Series:
    return (
        (d["Close"]      >  0.50)          &
        (d["Close"]      <= 50.0)          &
        (d["avg_vol"]    >  200_000)       &
        (d["Close"]      >  d["ema200"])   &
        (d["Close"]      >  d["ema50"])    &
        (d["ema50_dist"] <= 10.0)          &
        (d["rsi"]        >= 40)            &
        (d["rsi"]        <= 65)            &
        (d["atr_pct"]    >= MIN_ATR_PCT)   &
        (d["rs_vs_xjo"]  >  0)            &
        (d["xjo_uptrend"] == True)
    )


# ---------------------------------------------------------------------------
# Signal pool builder (Variant F)
# ---------------------------------------------------------------------------

def build_signal_pool_f(price_data: dict[str, pd.DataFrame],
                        xjo_close: pd.Series,
                        f_regime_series: pd.Series) -> tuple[dict, dict]:
    xjo_uptrend = pd.Series(True, index=xjo_close.index)
    print("  Computing indicators ...")
    ind = {t: compute_indicators(df, xjo_close, xjo_uptrend)
           for t, df in price_data.items()}

    print("  Computing RS percentiles ...")
    date_rs: dict = defaultdict(list)
    for d_tmp in ind.values():
        for dt, val in d_tmp["rs_vs_xjo"].dropna().items():
            date_rs[dt].append(float(val))
    rs_pct80 = {dt: float(np.percentile(vals, 80))
                for dt, vals in date_rs.items() if len(vals) >= 5}

    by_entry: dict = defaultdict(list)
    skipped = 0
    sig_counts: dict[str, int] = {"STRONG_BULL": 0, "WEAK_BULL": 0, "CHOPPY_BEAR": 0}

    for ticker, d in ind.items():
        sigs = find_signals(d)
        n    = len(d)
        for i in range(WARMUP, n - 1):
            if not sigs.iloc[i]:
                continue
            rv  = float(d["rs_vs_xjo"].values[i])
            thr = rs_pct80.get(d.index[i], -np.inf)
            if np.isnan(rv) or rv < thr:
                continue
            tc = int(d["trigger_count"].values[i])
            if tc < 2:
                continue
            atr = float(d["atr"].values[i])
            if atr <= 0:
                continue

            ep       = float(d["Close"].values[i])
            entry_dt = d.index[i]
            if ep <= 0:
                continue

            sig_dt = d.index[i]
            f_reg  = (f_regime_series.loc[sig_dt]
                      if sig_dt in f_regime_series.index else "STRONG_BULL")
            if f_reg == "STRONG_BULL":
                ms = 1.0
            elif f_reg == "WEAK_BULL":
                ms = 0.8
            else:
                roc20_v = d["roc_20"].values[i]
                slop_v  = d["ema50_slope"].values[i]
                mom_ok  = (not np.isnan(roc20_v) and roc20_v > 0 and
                           not np.isnan(slop_v)  and slop_v  > 0)
                if not mom_ok:
                    skipped += 1
                    continue
                ms = 1.0
            sig_counts[f_reg] = sig_counts.get(f_reg, 0) + 1

            by_entry[entry_dt].append({
                "ticker":    ticker,
                "entry_open": ep,
                "atr":        atr,
                "rs_val":     rv,
                "mom_scale":  ms,
                "f_regime":   f_reg,
            })

    total = sum(len(v) for v in by_entry.values())
    print(f"  {total} signals in pool  ({skipped} CHOPPY_BEAR blocked)")
    sb = sig_counts["STRONG_BULL"]
    wb = sig_counts["WEAK_BULL"]
    cb = sig_counts["CHOPPY_BEAR"]
    tot_f = max(sb + wb + cb, 1)
    print(f"  F-split:  STRONG_BULL {sb} ({sb/tot_f*100:.0f}%)  "
          f"WEAK_BULL {wb} ({wb/tot_f*100:.0f}%)  "
          f"CHOPPY_BEAR {cb} ({cb/tot_f*100:.0f}%)")
    return ind, by_entry


# ---------------------------------------------------------------------------
# Core simulation loop — with optional gap risk engine
# ---------------------------------------------------------------------------

def _record_trade(pos: dict, ticker: str, dt,
                  fill_price: float, exit_type: str,
                  comm_exit: float, slippage: float,
                  account: float, extra: dict | None = None,
                  shares_override: int | None = None) -> tuple[dict, float]:
    """Build a trade record and return (record, new_account)."""
    shares  = shares_override if shares_override is not None else pos["shares"]
    pnl     = (fill_price - pos["entry"]) * shares - comm_exit
    pnl_r   = (fill_price - pos["entry"]) / pos["initial_risk"]
    account += pnl
    rec = {
        "ticker":        ticker,
        "entry_date":    pos["entry_date"],
        "exit_date":     dt,
        "entry_price":   round(pos["entry"], 4),
        "exit_price":    round(fill_price, 4),
        "stop_initial":  round(pos["stop_initial"], 4),
        "target":        round(pos["target"], 4),
        "initial_risk":  round(pos["initial_risk"], 4),
        "shares":        shares,
        "risk_aud":      round(pos["risk_aud"] * shares / max(pos["shares"], 1), 2),
        "psm":           pos["psm"],
        "f_regime":      pos.get("f_regime", "STRONG_BULL"),
        "holding_days":  pos["bars_held"],
        "exit_type":     exit_type,
        "trail_active":  pos.get("trail_active", False),
        "trail_high":    round(pos.get("trail_high", 0.0), 4),
        "pnl_r":         round(pnl_r, 3),
        "pnl_aud":       round(pnl, 2),
        "account_after": round(account, 2),
    }
    if extra:
        rec.update(extra)
    return rec, account


def _simulate(
    ind:              dict[str, pd.DataFrame],
    by_entry:         dict,
    regime_series:    pd.DataFrame | None = None,   # full series with regime + f_regime cols
    comm_entry:       float = 6.0,
    comm_exit:        float = 6.0,
    slippage:         float = 0.002,
    gap_risk:         bool  = True,
    stop_mult:        float = ATR_STOP_MULT,
    target_r:         float = TARGET_R,
    trail_atr_mult:   float = TRAIL_ATR_MULT,
    use_gap_risk_eng: bool  = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Simulation loop for F-Trail with optional gap risk engine layer.

    trail_atr_mult > 0  : trailing stop mode (2xATR, activates at +1R)
    use_gap_risk_eng    : if True, apply gap_risk_engine logic every bar
    """
    all_dates = sorted(set().union(*[set(d.index) for d in ind.values()]))
    date_to_i = {dt: i for i, dt in enumerate(all_dates)}

    account   = STARTING_BALANCE
    open_pos  = {}
    cooldown  = {}
    all_trades: list[dict] = []
    eq_log:    list[dict] = []

    # Track event-risk bars for entry blocking
    event_risk_today: set[str] = set()

    for dt in all_dates:
        dt_i = date_to_i[dt]

        # ── A. Regular exit check ─────────────────────────────────────────────
        to_close = []
        for ticker, pos in open_pos.items():
            d = ind[ticker]
            if dt not in d.index:
                pos["bars_held"] += 1
                continue

            o       = float(d["Open"].loc[dt])
            c       = float(d["Close"].loc[dt])
            raw_atr = d["atr"].loc[dt]
            cur_atr = float(raw_atr) if not pd.isna(raw_atr) else pos["initial_risk"]
            t_mult  = pos.get("trail_mult", trail_atr_mult)   # per-position multiplier

            fill_price: float | None = None
            exit_type:  str | None   = None

            # Effective stop for gap check (trail_stop if active, else hard stop)
            eff_stop = pos["trail_stop"] if pos.get("trail_active") else pos["stop"]

            # 1. Gap risk check (open vs stop from previous close)
            if gap_risk and o <= eff_stop:
                fill_price = o * (1.0 - slippage)
                exit_type  = "stop_gap"
            else:
                if not pos.get("trail_active"):
                    # Phase 1: check activation
                    activation_lvl = pos["entry"] + TRAIL_ACTIVATION_R * pos["initial_risk"]
                    if c >= activation_lvl:
                        pos["trail_active"] = True
                        pos["trail_high"]   = c
                        pos["trail_stop"]   = c - t_mult * cur_atr
                    elif c <= pos["stop"]:
                        fill_price = c * (1.0 - slippage)
                        exit_type  = "stop"
                else:
                    # Phase 2: update trail
                    if c > pos["trail_high"]:
                        pos["trail_high"] = c
                    pos["trail_stop"] = pos["trail_high"] - t_mult * cur_atr
                    if c <= pos["trail_stop"]:
                        fill_price = c * (1.0 - slippage)
                        exit_type  = "trail"

                # Time stop (both phases)
                if exit_type is None and pos["bars_held"] >= TIME_STOP_DAYS:
                    fill_price = c * (1.0 - slippage)
                    exit_type  = "time"

            if exit_type:
                rec, account = _record_trade(pos, ticker, dt, fill_price,
                                             exit_type, comm_exit, slippage, account)
                all_trades.append(rec)
                to_close.append(ticker)
                cooldown[ticker] = dt_i + 3
            else:
                pos["bars_held"] += 1

        # Remove regular exits
        for ticker in to_close:
            del open_pos[ticker]

        # ── A.5 Gap Risk Engine ────────────────────────────────────────────────
        event_risk_today = set()   # reset each bar for entry blocking

        if use_gap_risk_eng and open_pos:
            # Get today's regime for scoring
            if regime_series is not None and dt in regime_series.index:
                regime_dt   = str(regime_series.loc[dt, "regime"])
                f_regime_dt = str(regime_series.loc[dt, "f_regime"])
            else:
                regime_dt   = "BULL"
                f_regime_dt = "STRONG_BULL"

            # Score regime: use f_regime name directly (covers CHOPPY_BEAR)
            # Also include legacy regime if it's CHOPPY/BEAR/HIGH_VOL
            effective_regime = f_regime_dt if f_regime_dt in CHOPPY_BEAR_REGIMES else regime_dt

            # ── Event risk detection for ALL tickers this bar ─────────────────
            for tkr, d_tkr in ind.items():
                if dt not in d_tkr.index:
                    continue
                vol_bar  = float(d_tkr["Volume"].loc[dt])
                avg_vol  = float(d_tkr["avg_vol"].loc[dt]) if not pd.isna(d_tkr["avg_vol"].loc[dt]) else 0.0
                hi_bar   = float(d_tkr["High"].loc[dt])
                lo_bar   = float(d_tkr["Low"].loc[dt])
                atr_bar  = float(d_tkr["atr"].loc[dt])   if not pd.isna(d_tkr["atr"].loc[dt])   else 0.0
                if check_event_risk(vol_bar, avg_vol, hi_bar - lo_bar, atr_bar):
                    event_risk_today.add(tkr)

            # ── Per-position evaluation ───────────────────────────────────────
            gap_risk_exits  = []
            gap_risk_reduces = []

            for ticker, pos in list(open_pos.items()):
                d_pos = ind.get(ticker)
                if d_pos is None or dt not in d_pos.index:
                    continue

                # Grace period: no gap-risk actions for the first 5 bars
                if pos["bars_held"] < 5:
                    continue

                c_bar   = float(d_pos["Close"].loc[dt])
                raw_atr = d_pos["atr"].loc[dt]
                atr_bar = float(raw_atr) if not pd.isna(raw_atr) else pos["initial_risk"]
                roc_raw = d_pos["roc_20"].loc[dt]
                roc_pct = float(roc_raw) * 100 if not pd.isna(roc_raw) else 0.0
                atr_pct = (atr_bar / c_bar * 100) if c_bar > 0 else 4.0

                current_r      = (c_bar - pos["entry"]) / pos["initial_risk"]
                classification = classify_position(current_r)
                score, _       = calc_gap_score(
                    pos["bars_held"], atr_pct, effective_regime, roc_pct, classification
                )

                # Event risk bonus for open positions
                if ticker in event_risk_today:
                    score += SCORE_EVENT_RISK

                action = get_action(classification, score, pos["bars_held"])

                if action in ("EXIT", "DAY7_EXIT"):
                    fill  = c_bar * (1.0 - slippage)
                    e_type = "gap_risk_exit" if action == "EXIT" else "day7_exit"
                    rec, account = _record_trade(
                        pos, ticker, dt, fill, e_type, comm_exit, slippage, account,
                        extra={"gap_score": score, "classification": classification}
                    )
                    all_trades.append(rec)
                    gap_risk_exits.append(ticker)
                    cooldown[ticker] = dt_i + 3

                elif action == "REDUCE" and not pos.get("gap_reduced", False):
                    # Reduce only fires once per position entry
                    half  = max(1, pos["shares"] // 2)
                    fill  = c_bar * (1.0 - slippage)
                    # Partial close: half shares
                    rec, account = _record_trade(
                        pos, ticker, dt, fill, "gap_risk_reduce",
                        comm_exit, slippage, account,
                        shares_override=half,
                        extra={"gap_score": score, "classification": classification}
                    )
                    all_trades.append(rec)
                    # Update remaining position (no cooldown — position continues)
                    remaining = pos["shares"] - half
                    pos["risk_aud"] = pos["risk_aud"] * remaining / max(pos["shares"], 1)
                    pos["shares"]   = remaining
                    pos["gap_reduced"] = True   # prevent repeat reduce on same position

                elif action == "TIGHTEN":
                    # Tighten trailing stop multiplier
                    pos["trail_mult"] = TRAIL_TIGHT_MULT

                # HOLD: no action

            # Remove gap-risk-exited positions
            for ticker in gap_risk_exits:
                if ticker in open_pos:
                    del open_pos[ticker]

            # ── Portfolio unpaid heat cap ─────────────────────────────────────
            unpaid_scored = []
            for ticker, pos in list(open_pos.items()):
                d_pos = ind.get(ticker)
                if d_pos is None or dt not in d_pos.index:
                    continue
                # Same grace period as per-position evaluation
                if pos["bars_held"] < 5:
                    continue
                c_bar     = float(d_pos["Close"].loc[dt])
                current_r = (c_bar - pos["entry"]) / pos["initial_risk"]
                if current_r < 1.0:
                    raw_atr = d_pos["atr"].loc[dt]
                    atr_bar = float(raw_atr) if not pd.isna(raw_atr) else pos["initial_risk"]
                    atr_pct = atr_bar / c_bar * 100 if c_bar > 0 else 4.0
                    roc_raw = d_pos["roc_20"].loc[dt]
                    roc_pct = float(roc_raw) * 100 if not pd.isna(roc_raw) else 0.0
                    cl      = "UNPAID"
                    sc, _   = calc_gap_score(pos["bars_held"], atr_pct,
                                             effective_regime, roc_pct, cl)
                    unpaid_scored.append((ticker, pos, sc))

            unpaid_heat     = sum(p["risk_aud"] for _, p, _ in unpaid_scored)
            unpaid_heat_pct = unpaid_heat / account * 100 if account > 0 else 0

            if unpaid_heat_pct > UNPAID_HEAT_CAP_PCT:
                unpaid_scored.sort(key=lambda x: x[2], reverse=True)
                for ticker, pos, sc in unpaid_scored:
                    if unpaid_heat_pct <= UNPAID_HEAT_CAP_PCT:
                        break
                    if ticker not in open_pos:
                        continue
                    d_pos = ind[ticker]
                    c_bar = float(d_pos["Close"].loc[dt])
                    fill  = c_bar * (1.0 - slippage)
                    rec, account = _record_trade(
                        pos, ticker, dt, fill, "gap_risk_heat_cap",
                        comm_exit, slippage, account,
                        extra={"gap_score": sc, "classification": "UNPAID"}
                    )
                    all_trades.append(rec)
                    del open_pos[ticker]
                    cooldown[ticker] = dt_i + 3
                    unpaid_heat     -= pos["risk_aud"]
                    unpaid_heat_pct  = unpaid_heat / account * 100 if account > 0 else 0

        # ── B. Mark-to-market ─────────────────────────────────────────────────
        unreal = sum(
            (float(ind[t]["Close"].loc[dt]) - pos["entry"]) * pos["shares"]
            for t, pos in open_pos.items()
            if dt in ind[t].index
        )
        eq_log.append({
            "date":           dt,
            "account":        round(account, 2),
            "n_open":         len(open_pos),
            "mark_to_market": round(account + unreal, 2),
        })

        # ── C. Entry decisions ────────────────────────────────────────────────
        if dt not in by_entry or len(open_pos) >= MAX_POSITIONS:
            continue

        base_risk  = account * RISK_PCT
        candidates = sorted(by_entry[dt], key=lambda x: x.get("rs_val", 0), reverse=True)

        for cand in candidates:
            if len(open_pos) >= MAX_POSITIONS:
                break
            ticker = cand["ticker"]
            if ticker in open_pos:
                continue
            if dt_i < cooldown.get(ticker, 0):
                continue

            # Block new entries in event-risk tickers
            if use_gap_risk_eng and ticker in event_risk_today:
                continue

            ms          = cand.get("mom_scale", 1.0)
            risk_scaled = base_risk * ms
            if risk_scaled < MIN_RISK_AUD:
                continue

            ep           = cand["entry_open"]
            atr          = cand["atr"]
            ep_filled    = ep * (1.0 + slippage)
            initial_risk = atr * stop_mult
            shares       = int(risk_scaled / initial_risk)
            if shares == 0:
                continue

            account -= comm_entry
            initial_stop = ep_filled - initial_risk

            open_pos[ticker] = {
                "entry_date":   dt,
                "entry":        ep_filled,
                "shares":       shares,
                "initial_risk": initial_risk,
                "risk_aud":     risk_scaled,
                "psm":          ms,
                "f_regime":     cand.get("f_regime", "STRONG_BULL"),
                "stop":         initial_stop,
                "stop_initial": initial_stop,
                "target":       ep_filled + target_r * initial_risk,
                "be_lvl":       ep_filled + BREAKEVEN_R * initial_risk,
                "be_set":       False,
                "bars_held":    0,
                "trail_active": False,
                "trail_high":   0.0,
                "trail_stop":   initial_stop,
                "trail_mult":   trail_atr_mult,   # per-position, can be tightened
                "gap_reduced":  False,            # prevent repeated gap-risk reduces
            }

    # Force-close remaining at end of data
    for ticker, pos in open_pos.items():
        d         = ind[ticker]
        last_c    = float(d["Close"].iloc[-1])
        fill_last = last_c * (1.0 - slippage)
        last_dt   = d.index[-1]
        rec, account = _record_trade(pos, ticker, last_dt, fill_last,
                                     "end_of_data", comm_exit, slippage, account)
        all_trades.append(rec)

    trades_df = pd.DataFrame(all_trades)
    equity_df = pd.DataFrame(eq_log)
    if not trades_df.empty:
        trades_df["entry_date"] = pd.to_datetime(trades_df["entry_date"])
        trades_df["exit_date"]  = pd.to_datetime(trades_df["exit_date"])
        trades_df.sort_values("exit_date", inplace=True)
        trades_df.reset_index(drop=True, inplace=True)
    if not equity_df.empty:
        equity_df["date"] = pd.to_datetime(equity_df["date"])
    return trades_df, equity_df


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

# Exit types added by gap risk engine
GAP_RISK_EXIT_TYPES = frozenset({
    "gap_risk_exit", "day7_exit", "gap_risk_reduce", "gap_risk_heat_cap"
})


def calc_metrics(trades_df: pd.DataFrame, equity_df: pd.DataFrame,
                 label: str = "") -> dict:
    if trades_df.empty:
        return {}

    closed = trades_df[trades_df["exit_type"] != "end_of_data"].copy()
    if closed.empty:
        closed = trades_df.copy()

    n = len(closed)
    wins      = closed[closed["pnl_r"] >  0]
    losses    = closed[closed["pnl_r"] <= 0]
    win_rate  = len(wins) / n * 100 if n else 0

    gross_win  = wins["pnl_aud"].sum()
    gross_loss = losses["pnl_aud"].sum()
    pf         = gross_win / abs(gross_loss) if gross_loss < 0 else np.nan

    avg_r      = closed["pnl_r"].mean()
    expectancy = closed["pnl_aud"].mean()
    avg_hold   = closed["holding_days"].mean()
    total_pnl  = trades_df["pnl_aud"].sum()

    final_account = (trades_df["account_after"].iloc[-1]
                     if "account_after" in trades_df.columns
                     else STARTING_BALANCE + total_pnl)
    pct_return = (final_account / STARTING_BALANCE - 1) * 100
    cagr_years = 8.0
    cagr       = ((final_account / STARTING_BALANCE) ** (1.0 / cagr_years) - 1) * 100

    if not equity_df.empty:
        mtm  = equity_df["mark_to_market"].values
        peak = np.maximum.accumulate(mtm)
        dd   = mtm - peak
        max_dd_abs = dd.min()
        max_dd_pct = (dd / np.where(peak > 0, peak, 1) * 100).min()
    else:
        cum  = STARTING_BALANCE + closed.sort_values("exit_date")["pnl_aud"].cumsum()
        peak = cum.cummax()
        dd   = cum - peak
        max_dd_abs = dd.min()
        max_dd_pct = (dd / peak * 100).min()

    sorted_pnls = closed.sort_values("exit_date")["pnl_aud"].values
    max_consec = cur = 0
    for pnl in sorted_pnls:
        cur = cur + 1 if pnl < 0 else 0
        max_consec = max(max_consec, cur)

    pnl_r_std = closed["pnl_r"].std()
    sharpe = ((avg_r / pnl_r_std) * np.sqrt(252 / avg_hold)
              if pnl_r_std > 0 and avg_hold > 0 else np.nan)
    calmar = (cagr / abs(max_dd_pct)) if max_dd_pct < 0 else np.nan

    avg_win_r  = float(wins["pnl_r"].mean())  if not wins.empty  else 0.0
    avg_loss_r = float(losses["pnl_r"].mean()) if not losses.empty else 0.0
    pct_ge2r   = (closed["pnl_r"] >= 2.0).mean() * 100
    pct_ge3r   = (closed["pnl_r"] >= 3.0).mean() * 100
    max_r      = float(closed["pnl_r"].max())

    # Exit type breakdown
    def _count(et): return int((closed["exit_type"] == et).sum())
    def _countin(ets): return int(closed["exit_type"].isin(ets).sum())

    # Year-by-year
    yearly = {}
    closed2 = trades_df.copy()
    closed2["year"] = pd.to_datetime(closed2["exit_date"]).dt.year
    year_end_acct = {}
    if not equity_df.empty:
        eq = equity_df.copy()
        eq["year"] = pd.to_datetime(eq["date"]).dt.year
        for yr, grp in eq.groupby("year"):
            year_end_acct[yr] = grp["mark_to_market"].iloc[-1]

    prev_acct = STARTING_BALANCE
    for yr in sorted(closed2["year"].unique()):
        grp   = closed2[closed2["year"] == yr]
        n_y   = len(grp[grp["exit_type"] != "end_of_data"])
        w_y   = len(grp[grp["pnl_r"] > 0])
        wr_y  = w_y / n_y * 100 if n_y else 0
        pnl_y = grp["pnl_aud"].sum()
        gw    = grp[grp["pnl_aud"] > 0]["pnl_aud"].sum()
        gl    = abs(grp[grp["pnl_aud"] < 0]["pnl_aud"].sum())
        pf_y  = round(gw / gl, 2) if gl > 0 else "n/a"
        acct_y = year_end_acct.get(yr, prev_acct + pnl_y)
        ret_y  = (acct_y / prev_acct - 1) * 100 if prev_acct > 0 else 0
        yearly[yr] = {"trades": n_y, "win_pct": round(wr_y, 1),
                      "pnl_aud": round(pnl_y, 2), "pf": pf_y,
                      "acct_end": round(acct_y, 2), "return_pct": round(ret_y, 1)}
        prev_acct = acct_y

    return {
        "label":              label,
        "total_records":      n,                          # includes partial reduces
        "full_exits":         n - _count("gap_risk_reduce"),
        "win_rate_pct":       round(win_rate, 1),
        "wins":               len(wins),
        "losses":             len(losses),
        "trail_exits":        _count("trail"),
        "target_exits":       _count("target"),
        "hard_stops":         _count("stop"),
        "gap_stops":          _count("stop_gap"),         # << key metric
        "time_exits":         _count("time"),
        "gap_risk_exits":     _count("gap_risk_exit"),    # score-triggered
        "day7_exits":         _count("day7_exit"),        # day-7 rule
        "gap_risk_reduces":   _count("gap_risk_reduce"),  # partial closes
        "heat_cap_exits":     _count("gap_risk_heat_cap"),
        "profit_factor":      round(pf, 2) if not np.isnan(pf) else "n/a",
        "avg_r":              round(avg_r, 3),
        "avg_win_r":          round(avg_win_r, 3),
        "avg_loss_r":         round(avg_loss_r, 3),
        "pct_ge2r":           round(pct_ge2r, 1),
        "pct_ge3r":           round(pct_ge3r, 1),
        "best_trade_r":       round(max_r, 3),
        "expectancy_aud":     round(expectancy, 2),
        "avg_hold_days":      round(avg_hold, 1),
        "total_pnl_aud":      round(total_pnl, 2),
        "final_account":      round(final_account, 2),
        "pct_return":         round(pct_return, 1),
        "cagr_pct":           round(cagr, 1),
        "calmar":             round(calmar, 2) if not np.isnan(calmar) else "n/a",
        "max_drawdown_aud":   round(max_dd_abs, 2),
        "max_drawdown_pct":   round(max_dd_pct, 1),
        "max_consec_losses":  max_consec,
        "sharpe":             round(sharpe, 2) if not np.isnan(sharpe) else "n/a",
        "yearly":             yearly,
    }


# ---------------------------------------------------------------------------
# Comparison table
# ---------------------------------------------------------------------------

def print_comparison(results: list[tuple[str, dict]]) -> None:
    KEY_METRICS = [
        ("total_records",    "Trade records (incl. partial)"),
        ("full_exits",       "  Full position exits"),
        ("win_rate_pct",     "Win rate %"),
        ("wins",             "  Profitable records"),
        ("losses",           "  Losing records"),
        ("trail_exits",      "  Trail stop exits"),
        ("hard_stops",       "  Hard stop exits"),
        ("gap_stops",        "  Gap stop exits  <<"),       # << key metric
        ("time_exits",       "  Time stop exits"),
        ("gap_risk_exits",   "  Gap risk exits (score)"),
        ("day7_exits",       "  Day-7 unpaid exits"),
        ("gap_risk_reduces", "  Partial reduces (50%)"),
        ("heat_cap_exits",   "  Heat cap force exits"),
        ("profit_factor",    "Profit factor  (must >1.6)"),  # << key metric
        ("avg_r",            "Avg R / record  (must hold)"), # << key metric
        ("avg_win_r",        "  Avg R (winners)"),
        ("avg_loss_r",       "  Avg R (losers)"),
        ("pct_ge2r",         "  % records >= 2R"),
        ("pct_ge3r",         "  % records >= 3R"),
        ("best_trade_r",     "  Best trade R  (LYC must survive)"),  # << key
        ("expectancy_aud",   "Expectancy $"),
        ("avg_hold_days",    "Avg hold days"),
        ("total_pnl_aud",    "Total P&L $"),
        ("final_account",    "Final account $"),
        ("pct_return",       "Total return %"),
        ("cagr_pct",         "CAGR %"),
        ("max_drawdown_aud", "Max drawdown $"),
        ("max_drawdown_pct", "Max drawdown %"),
        ("calmar",           "Calmar"),
        ("max_consec_losses","Max consec. losses"),
        ("sharpe",           "Sharpe"),
    ]
    LOWER_IS_BETTER = {"losses", "hard_stops", "gap_stops", "time_exits",
                       "gap_risk_exits", "day7_exits", "heat_cap_exits",
                       "max_drawdown_aud", "max_drawdown_pct",
                       "max_consec_losses", "avg_loss_r"}

    labels  = [lbl for lbl, _ in results]
    col_w   = max(34, max(len(l) for l in labels) + 4)
    met_w   = 36
    sep     = "=" * (met_w + col_w * len(labels) + 4)

    print(f"\n{sep}")
    print(f"  GAP RISK ENGINE  -  BACKTEST IMPACT  (Variant F)  --  {date.today()}")
    print(f"  ${STARTING_BALANCE:,} start | {RISK_PCT*100:.1f}% risk/trade | IBKR $6/leg | 0.2% slip")
    print(f"  F-Trail baseline : 2xATR trail @1R, no gap risk layer")
    print(f"  F-Trail+GapRisk  : same trail + gap risk engine every bar")
    print(f"  Gap risk engine  : Day-7 unpaid exit | score thresholds | "
          f"unpaid heat cap {UNPAID_HEAT_CAP_PCT:.0f}% | event risk block")
    print(sep)

    hdr = f"  {'Metric':<{met_w}}"
    for lbl in labels:
        hdr += f"  {lbl:^{col_w - 2}}"
    print(hdr)
    print("-" * (met_w + col_w * len(labels) + 4))

    for key, display in KEY_METRICS:
        vals = [m.get(key, None) for _, m in results]

        def fmt(v, k=key):
            if v is None: return "-"
            if k in ("expectancy_aud", "total_pnl_aud", "final_account", "max_drawdown_aud"):
                return f"${v:,.0f}"
            if k in ("pct_return", "cagr_pct"): return f"{v:+.1f}%"
            if k == "max_drawdown_pct": return f"{v:.1f}%"
            if k in ("win_rate_pct", "pct_ge2r", "pct_ge3r"): return f"{v:.1f}%"
            return str(v)

        numeric = [(i, v) for i, v in enumerate(vals) if isinstance(v, (int, float))]
        best_i  = -1
        if numeric:
            best_i = (min(numeric, key=lambda x: x[1]) if key in LOWER_IS_BETTER
                      else max(numeric, key=lambda x: x[1]))[0]

        row = f"  {display:<{met_w}}"
        for i, v in enumerate(vals):
            cell   = fmt(v)
            marker = " *" if i == best_i and len(numeric) > 1 else "  "
            row   += f"  {cell:^{col_w - 4}}{marker}"
        print(row)

    print(sep)
    print("  * = best value in row")
    print("  << = key validation metric\n")

    # Year-by-year
    print(f"  YEAR-BY-YEAR BREAKDOWN")
    print("-" * (met_w + col_w * len(labels) + 4))
    yr_hdr = f"  {'Year':<{met_w}}"
    for lbl in labels:
        yr_hdr += f"  {lbl:^{col_w - 2}}"
    print(yr_hdr)
    print("-" * (met_w + col_w * len(labels) + 4))
    all_years = sorted(set(yr for _, m in results for yr in m.get("yearly", {})))
    for yr in all_years:
        cells = []
        yr_vals = []
        for _, m in results:
            y = m.get("yearly", {}).get(yr)
            if y:
                s = "+" if y["pnl_aud"] >= 0 else ""
                cells.append(f"{s}${y['pnl_aud']:,.0f} ({y['return_pct']:+.1f}%)")
                yr_vals.append(y["pnl_aud"])
            else:
                cells.append("-"); yr_vals.append(None)
        num_yr  = [(i, v) for i, v in enumerate(yr_vals) if v is not None]
        best_yi = max(num_yr, key=lambda x: x[1])[0] if num_yr else -1
        row = f"  {str(yr):<{met_w}}"
        for i, cell in enumerate(cells):
            marker = " *" if i == best_yi and len(num_yr) > 1 else "  "
            row   += f"  {cell:^{col_w - 4}}{marker}"
        print(row)
    print("=" * (met_w + col_w * len(labels) + 4))
    print("  * = better year\n")

    # Specific validation checks
    print("  VALIDATION CHECKS")
    print("-" * 60)
    for lbl, m in results:
        pf  = m.get("profit_factor", 0)
        avgr = m.get("avg_r", 0)
        bestr = m.get("best_trade_r", 0)
        gs  = m.get("gap_stops", 0)
        pf_ok   = isinstance(pf, float) and pf > 1.6
        best_ok = bestr >= 4.0   # LYC +4.576R (allow small rounding)
        label_safe = lbl[:35]
        print(f"  {label_safe:<35}  PF={pf}  {'OK' if pf_ok else 'FAIL <<'}  |  "
              f"AvgR={avgr}  |  GapStops={gs}  |  BestR={bestr}  "
              f"{'(LYC OK)' if best_ok else '(LYC MISSING!)'}")
    print()


# ---------------------------------------------------------------------------
# Chart
# ---------------------------------------------------------------------------

def save_chart(results: list[tuple[str, dict]],
               equity_dfs: list[pd.DataFrame],
               out_path: Path) -> None:
    BG, PANEL  = "#0d1117", "#161b22"
    PALETTE    = ["#58a6ff", "#26a641"]
    GREY, RED  = "#8b949e", "#da3633"

    fig = plt.figure(figsize=(20, 14), facecolor=BG)
    gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.40, wspace=0.28,
                            height_ratios=[3, 2])
    ax_eq   = fig.add_subplot(gs[0, :])
    ax_yr   = [fig.add_subplot(gs[1, i]) for i in range(2)]

    for ax in [ax_eq] + ax_yr:
        ax.set_facecolor(PANEL)
        ax.tick_params(colors=GREY, labelsize=8)
        for sp in ax.spines.values():
            sp.set_edgecolor("#21262d")

    # Equity curves
    for (label, m), eq_df, colour in zip(results, equity_dfs, PALETTE):
        if not eq_df.empty:
            xs  = np.arange(len(eq_df))
            mtm = eq_df["mark_to_market"].values
        else:
            xs  = np.array([0, 1])
            mtm = np.array([STARTING_BALANCE, STARTING_BALANCE])
        final      = m.get("final_account", STARTING_BALANCE)
        ret        = m.get("pct_return", 0)
        label_safe = label.replace("$", r"\$")
        lbl        = f"{label_safe}  (A\\${final:,.0f} | {ret:+.1f}%)"
        ax_eq.plot(xs, mtm, color=colour, lw=1.6, label=lbl, zorder=3)

    ax_eq.axhline(STARTING_BALANCE, color=GREY, lw=0.8, ls="--", alpha=0.5)
    ax_eq.set_title(
        f"Equity Curves | F-Trail vs F-Trail+GapRisk  |  "
        f"$20k start  |  {date.today()}",
        color="white", fontsize=11, pad=8)
    ax_eq.set_ylabel("AUD", color=GREY, fontsize=9)
    ax_eq.grid(color="#21262d", lw=0.5, alpha=0.6)
    ax_eq.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"${v:,.0f}"))
    ax_eq.legend(fontsize=9, facecolor=PANEL, edgecolor=GREY, labelcolor="white",
                 loc="upper left")
    if equity_dfs and not equity_dfs[0].empty:
        eq_dates = pd.to_datetime(equity_dfs[0]["date"])
        step = max(1, len(eq_dates) // 10)
        tpos = list(range(0, len(eq_dates), step))
        tlbl = [pd.Timestamp(eq_dates.iloc[i]).strftime("%b %y") for i in tpos if i < len(eq_dates)]
        ax_eq.set_xticks(tpos[:len(tlbl)])
        ax_eq.set_xticklabels(tlbl, color=GREY, fontsize=8)

    # Year-by-year
    all_years = sorted(set(yr for _, m in results for yr in m.get("yearly", {})))
    for ax, (label, m), colour in zip(ax_yr, results, PALETTE):
        yearly = m.get("yearly", {})
        years  = [yr for yr in all_years if yr in yearly]
        pnls   = [yearly[yr]["pnl_aud"] for yr in years]
        cols   = ["#26a641" if p >= 0 else RED for p in pnls]
        ax.bar(years, pnls, color=cols, alpha=0.85, width=0.6)
        ax.axhline(0, color=GREY, lw=0.8, ls="--")
        label_safe = label.replace("$", r"\$")
        ax.set_title(f"Year-by-Year  |  {label_safe}", color="white", fontsize=9, pad=5)
        ax.set_ylabel("AUD", color=GREY, fontsize=9)
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"${v:,.0f}"))
        ax.grid(color="#21262d", lw=0.5, alpha=0.6, axis="y")
        ax.set_xticks(years)
        ax.set_xticklabels([str(y)[2:] for y in years], color=GREY, fontsize=8)
        rng = max(abs(p) for p in pnls) * 0.15 if pnls else 1
        for yr, pnl in zip(years, pnls):
            off = rng if pnl >= 0 else -rng
            va  = "bottom" if pnl >= 0 else "top"
            ax.text(yr, pnl + off, f"${pnl:,.0f}", ha="center", va=va,
                    color=GREY, fontsize=6.5)

    footer = (
        f"Variant F | 2xATR trail @1R | IBKR \\$6/leg | 0.2% slip | gap risk | "
        f"Day-7 unpaid exit | score thresholds | "
        f"unpaid heat cap {UNPAID_HEAT_CAP_PCT:.0f}% | event risk block"
    )
    fig.text(0.5, 0.002, footer, color=GREY, fontsize=7, ha="center")
    plt.savefig(out_path, dpi=130, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"  Chart -> {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Variant F: trailing stop baseline vs + gap risk engine")
    p.add_argument("--period", default="9y")
    return p.parse_args()


def main():
    args = parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    today_str = date.today().strftime("%Y%m%d")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("  ASX SWING ENGINE  --  GAP RISK ENGINE BACKTEST IMPACT")
    print(f"  Variant F: F-Trail  vs  F-Trail + Gap Risk Engine")
    print(f"  {date.today()}  |  period={args.period}  |  ${STARTING_BALANCE:,} start")
    print("=" * 72)
    print()

    # ── 1. XJO ───────────────────────────────────────────────────────────────
    print("Step 1/5  Downloading XJO ...")
    with _quiet():
        xjo_raw = yf.download(BENCHMARK, period=args.period, interval="1d",
                              auto_adjust=True, progress=False)
    if isinstance(xjo_raw.columns, pd.MultiIndex):
        xjo_raw.columns = [c[0] for c in xjo_raw.columns]
    xjo_df    = xjo_raw[["Open","High","Low","Close","Volume"]].dropna()
    xjo_close = xjo_df["Close"]
    print(f"  XJO: {len(xjo_close)} bars "
          f"({xjo_close.index[0].date()} to {xjo_close.index[-1].date()})\n")

    # ── 2. Regime series ──────────────────────────────────────────────────────
    print("Step 2/5  Building regime series ...")
    regime_series   = build_regime_series(xjo_df, args.period)
    f_regime_series = regime_series["f_regime"]
    print()

    # ── 3. Universe ───────────────────────────────────────────────────────────
    tickers = list(dict.fromkeys(ASX_PRESET))
    print(f"Step 3/5  Downloading {len(tickers)}-stock universe ...")
    price_data = download_universe(tickers, args.period)
    print(f"  Downloaded {len(price_data)}/{len(tickers)} tickers.\n")
    if not price_data:
        print("ERROR: no price data."); sys.exit(1)

    # ── 4. Signal pool (shared by both scenarios) ─────────────────────────────
    print("Step 4/5  Building Variant-F signal pool ...")
    ind, by_entry = build_signal_pool_f(price_data, xjo_close, f_regime_series)
    print()

    # ── 5. Run simulations ────────────────────────────────────────────────────
    realistic = dict(comm_entry=6.0, comm_exit=6.0, slippage=0.002,
                     gap_risk=True, stop_mult=2.0, target_r=2.0,
                     trail_atr_mult=TRAIL_ATR_MULT)

    scenarios = [
        {
            "label":  "F-Trail  (baseline, no gap risk)",
            "params": dict(use_gap_risk_eng=False, regime_series=None),
            "slug":   "gap_baseline",
        },
        {
            "label":  "F-Trail + Gap Risk Engine",
            "params": dict(use_gap_risk_eng=True, regime_series=regime_series),
            "slug":   "gap_with_engine",
        },
    ]

    all_results: list[tuple[str, dict]] = []
    all_equity:  list[pd.DataFrame]     = []
    print("Step 5/5  Simulating ...")

    for sc in scenarios:
        lbl = sc["label"]
        print(f"\n  -- {lbl} --")
        trades_df, equity_df = _simulate(ind, by_entry, **realistic, **sc["params"])

        if trades_df.empty:
            print("  WARNING: no trades."); continue

        m = calc_metrics(trades_df, equity_df, label=lbl)
        all_results.append((lbl, m))
        all_equity.append(equity_df)

        trades_df.to_csv(RESULTS_DIR / f"trades_gap_{sc['slug']}_{today_str}.csv", index=False)
        equity_df.to_csv(RESULTS_DIR / f"equity_gap_{sc['slug']}_{today_str}.csv", index=False)

        print(f"  Records: {m['total_records']}  |  "
              f"Win: {m['win_rate_pct']}%  |  "
              f"PF: {m['profit_factor']}  |  "
              f"AvgR: {m['avg_r']}  |  "
              f"GapStops: {m['gap_stops']}  |  "
              f"GapRiskExits: {m['gap_risk_exits']}  |  "
              f"Day7: {m['day7_exits']}  |  "
              f"Reduces: {m['gap_risk_reduces']}  |  "
              f"BestR: {m['best_trade_r']}")

    print()
    if all_results:
        print_comparison(all_results)
        save_chart(all_results, all_equity,
                   RESULTS_DIR / f"backtest_gap_impact_{today_str}.png")
        print(f"Results saved to {RESULTS_DIR}/")
    else:
        print("No results to compare.")


if __name__ == "__main__":
    main()
