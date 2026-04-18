# -*- coding: utf-8 -*-
"""
backtest_trail.py  -  ASX Swing Engine
=======================================
Compares two Variant-F exit strategies over the same 8-year signal pool:

  F-Fixed   : original 2:1 fixed target  (entry + 2 x initial_risk)
  F-Trail   : 2xATR trailing stop that activates once the trade reaches +1R
              - Before 1R: original 2xATR hard stop (entry - 2xATR)
              - After 1R:  trail = highest_close_since - 2 x current_ATR
              - No fixed upside cap -- rides the trend until trailed out

Both scenarios:
  - Identical Variant-F signal pool  (STRONG_BULL 1.0x / WEAK_BULL 0.8x /
    CHOPPY_BEAR hard momentum filter)
  - IBKR costs: $6/leg entry + $6/leg exit
  - 0.2% adverse slippage on every fill
  - Gap risk: if open <= effective stop, exit at open * (1 - slippage)
  - Time stop: 20 bars
  - Breakeven not applied in trail mode (trail activation supersedes it)

Usage:
    python backtest_trail.py
    python backtest_trail.py --period 9y
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

# ---------------------------------------------------------------------------
# Constants (mirror live engine)
# ---------------------------------------------------------------------------
STARTING_BALANCE  = 20_000.0
RISK_PCT          = 0.015
MAX_POSITIONS     = 6
ATR_STOP_MULT     = 2.0
TARGET_R          = 2.0
BREAKEVEN_R       = 1.0
TRAIL_ACTIVATION_R = 1.0   # trail activates once close >= entry + 1R
TRAIL_ATR_MULT    = 2.0    # trailing stop width: highest_close - 2xATR
TIME_STOP_DAYS    = 20
COMMISSION_ENTRY  = 6.0
COMMISSION_EXIT   = 6.0
SLIPPAGE_PCT      = 0.002
MIN_RISK_AUD      = 75.0
MIN_ATR_PCT       = 3.5    # match backtest_full.py production run
EMA_FAST, EMA_SLOW = 50, 200
RS_PERIOD         = 63
WARMUP            = EMA_SLOW + RS_PERIOD + 10
BATCH_SIZE        = 20
BATCH_DELAY       = 1.0
BENCHMARK         = "^AXJO"

RESULTS_DIR       = Path("results/backtest")

# ---------- universe (identical to backtest_full.py) -----------------------
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
# Historical regime series (Variant F)
# ---------------------------------------------------------------------------

HIGH_VOL_ATR    = 2.5
CHOPPY_SLOPE    = 0.1
CHOPPY_BREADTH  = 50.0


def build_regime_series(xjo_df: pd.DataFrame, period: str) -> pd.DataFrame:
    print("Building historical Variant-F regime series ...")
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

        # Legacy 5-state (stored for completeness, not used in F-variant logic)
        atp = float(atr_pct.loc[dt]) if not pd.isna(atr_pct.loc[dt]) else 0.0
        if atp > HIGH_VOL_ATR:
            regime = "HIGH_VOL"
        elif not a200:
            regime = "BEAR"
        elif abs(slop) < CHOPPY_SLOPE and brd < CHOPPY_BREADTH:
            regime = "CHOPPY"
        elif roc_v < 0 or slop < 0:
            regime = "WEAK_BULL"
        else:
            regime = "BULL"

        psm = {"BULL":1.00,"WEAK_BULL":0.75,"CHOPPY":0.60,"BEAR":0.50,"HIGH_VOL":0.25}.get(regime,1.0)

        # Variant-F 3-state
        if a200 and brd > 55:
            f_reg = "STRONG_BULL"
        elif a200 and brd >= 40:
            f_reg = "WEAK_BULL"
        else:
            f_reg = "CHOPPY_BEAR"

        dual_penalty = (roc_v < 0) and (slop < 0)
        require_all  = dual_penalty or (regime == "CHOPPY")
        min_trig     = 3 if require_all else 2

        rows.append({
            "date":                 dt,
            "regime":               regime,
            "psm":                  psm,
            "require_all_triggers": require_all,
            "min_triggers":         min_trig,
            "market_breadth":       brd,
            "blocked":              False,
            "f_regime":             f_reg,
        })

    df  = pd.DataFrame(rows).set_index("date")
    frc = df["f_regime"].value_counts()
    total_td = len(df)
    print(f"  F-regime split:  ", end="")
    for fr in ["STRONG_BULL", "WEAK_BULL", "CHOPPY_BEAR"]:
        n = int(frc.get(fr, 0))
        print(f"{fr} {n} ({n/total_td*100:.0f}%)  ", end="")
    print()
    return df


# ---------------------------------------------------------------------------
# Indicators + signal detection
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
    """
    Build the Variant-F signal pool:
      STRONG_BULL -> 1.0x size, no filter
      WEAK_BULL   -> 0.8x size, no filter
      CHOPPY_BEAR -> hard momentum filter (ROC20>0 AND EMA50_slope>0), full size
    Entry = signal-bar close (same as backtest_full.py production run).
    """
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
    skipped_filter = 0
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

            # Entry at signal-day close
            ep       = float(d["Close"].values[i])
            entry_dt = d.index[i]
            if ep <= 0:
                continue

            # F-regime sizing / filter
            sig_dt = d.index[i]
            f_reg  = (f_regime_series.loc[sig_dt]
                      if sig_dt in f_regime_series.index else "STRONG_BULL")
            if f_reg == "STRONG_BULL":
                ms = 1.0
            elif f_reg == "WEAK_BULL":
                ms = 0.8
            else:   # CHOPPY_BEAR
                roc20_v = d["roc_20"].values[i]
                slop_v  = d["ema50_slope"].values[i]
                mom_ok  = (not np.isnan(roc20_v) and roc20_v > 0 and
                           not np.isnan(slop_v)  and slop_v  > 0)
                if not mom_ok:
                    skipped_filter += 1
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
    print(f"  {total} signals in pool  ({skipped_filter} CHOPPY_BEAR blocked)")
    sb = sig_counts["STRONG_BULL"]
    wb = sig_counts["WEAK_BULL"]
    cb = sig_counts["CHOPPY_BEAR"]
    tot_f = max(sb + wb + cb, 1)
    print(f"  F-split:  STRONG_BULL {sb} ({sb/tot_f*100:.0f}%)  "
          f"WEAK_BULL {wb} ({wb/tot_f*100:.0f}%)  "
          f"CHOPPY_BEAR {cb} ({cb/tot_f*100:.0f}%)")
    return ind, by_entry


# ---------------------------------------------------------------------------
# Core simulation loop
# ---------------------------------------------------------------------------

def _simulate(ind: dict[str, pd.DataFrame],
              by_entry: dict,
              comm_entry: float = 6.0,
              comm_exit:  float = 6.0,
              slippage:   float = 0.002,
              gap_risk:   bool  = True,
              stop_mult:  float = ATR_STOP_MULT,
              target_r:   float = TARGET_R,
              trail_atr_mult: float = 0.0) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Core simulation loop.

    trail_atr_mult = 0.0  ->  fixed 2:1 target (original mode)
    trail_atr_mult > 0.0  ->  activate 2xATR trailing stop once trade
                               reaches +1R.  Fixed target is removed.

    Trail logic:
      Phase 1 (before 1R):  stop = entry - 2xATR  (hard stop, no change)
      Phase 2 (after 1R):   trail_stop = max_close_since_activation - trail_mult * cur_ATR
                             Moves up only.  Exit when close <= trail_stop.
    """
    all_dates = sorted(set().union(*[set(d.index) for d in ind.values()]))
    date_to_i = {dt: i for i, dt in enumerate(all_dates)}

    account   = STARTING_BALANCE
    open_pos  = {}
    cooldown  = {}
    all_trades: list[dict] = []
    eq_log:    list[dict] = []

    for dt in all_dates:
        dt_i = date_to_i[dt]

        # ── A. Exit check ────────────────────────────────────────────────────
        to_close = []
        for ticker, pos in open_pos.items():
            d = ind[ticker]
            if dt not in d.index:
                pos["bars_held"] += 1
                continue

            o = float(d["Open"].loc[dt])
            c = float(d["Close"].loc[dt])

            # Current bar's ATR (used by trailing stop)
            raw_atr = d["atr"].loc[dt]
            cur_atr = float(raw_atr) if not pd.isna(raw_atr) else pos["initial_risk"]

            fill_price: float | None = None
            exit_type:  str | None   = None

            if trail_atr_mult > 0:
                # ── Trailing stop mode ────────────────────────────────────────
                # Effective stop for gap-risk depends on whether trail is active
                eff_stop = pos["trail_stop"] if pos["trail_active"] else pos["stop"]

                # 1. Gap risk (vs end-of-previous-bar stop)
                if gap_risk and o <= eff_stop:
                    fill_price = o * (1.0 - slippage)
                    exit_type  = "stop_gap"

                else:
                    if not pos["trail_active"]:
                        # Phase 1: check for trail activation
                        activation_level = pos["entry"] + TRAIL_ACTIVATION_R * pos["initial_risk"]
                        if c >= activation_level:
                            pos["trail_active"] = True
                            pos["trail_high"]   = c
                            # Compute trail stop from activation close
                            pos["trail_stop"]   = c - trail_atr_mult * cur_atr
                            # Don't exit on activation bar (trail_stop < c by definition)
                        elif c <= pos["stop"]:
                            fill_price = c * (1.0 - slippage)
                            exit_type  = "stop"
                    else:
                        # Phase 2: update trailing stop
                        if c > pos["trail_high"]:
                            pos["trail_high"] = c
                        pos["trail_stop"] = pos["trail_high"] - trail_atr_mult * cur_atr

                        if c <= pos["trail_stop"]:
                            fill_price = c * (1.0 - slippage)
                            exit_type  = "trail"

                    # Time stop applies in both phases
                    if exit_type is None and pos["bars_held"] >= TIME_STOP_DAYS:
                        fill_price = c * (1.0 - slippage)
                        exit_type  = "time"

            else:
                # ── Fixed 2:1 target mode (original) ─────────────────────────
                # 1. Gap risk
                if gap_risk and o <= pos["stop"]:
                    fill_price = o * (1.0 - slippage)
                    exit_type  = "stop_gap"
                else:
                    # Breakeven ratchet
                    if not pos["be_set"] and c >= pos["be_lvl"]:
                        pos["stop"]   = pos["entry"]
                        pos["be_set"] = True

                    if c <= pos["stop"]:
                        fill_price = c * (1.0 - slippage)
                        exit_type  = "stop"
                    elif c >= pos["target"]:
                        fill_price = pos["target"] * (1.0 - slippage)
                        exit_type  = "target"
                    elif pos["bars_held"] >= TIME_STOP_DAYS:
                        fill_price = c * (1.0 - slippage)
                        exit_type  = "time"

            if exit_type:
                pnl   = (fill_price - pos["entry"]) * pos["shares"] - comm_exit
                pnl_r = (fill_price - pos["entry"]) / pos["initial_risk"]
                account += pnl
                all_trades.append({
                    "ticker":        ticker,
                    "entry_date":    pos["entry_date"],
                    "exit_date":     dt,
                    "entry_price":   round(pos["entry"], 4),
                    "exit_price":    round(fill_price, 4),
                    "stop_initial":  round(pos["stop_initial"], 4),
                    "target":        round(pos["target"], 4),
                    "initial_risk":  round(pos["initial_risk"], 4),
                    "shares":        pos["shares"],
                    "risk_aud":      round(pos["risk_aud"], 2),
                    "psm":           pos["psm"],
                    "f_regime":      pos["f_regime"],
                    "holding_days":  pos["bars_held"],
                    "exit_type":     exit_type,
                    "trail_active":  pos.get("trail_active", False),
                    "trail_high":    round(pos.get("trail_high", 0.0), 4),
                    "pnl_r":         round(pnl_r, 3),
                    "pnl_aud":       round(pnl, 2),
                    "account_after": round(account, 2),
                })
                to_close.append(ticker)
                cooldown[ticker] = dt_i + 3
            else:
                pos["bars_held"] += 1

        for ticker in to_close:
            del open_pos[ticker]

        # ── B. Mark-to-market ────────────────────────────────────────────────
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

        base_risk = account * RISK_PCT
        candidates = sorted(by_entry[dt], key=lambda x: x.get("rs_val", 0), reverse=True)

        for cand in candidates:
            if len(open_pos) >= MAX_POSITIONS:
                break
            ticker = cand["ticker"]
            if ticker in open_pos:
                continue
            if dt_i < cooldown.get(ticker, 0):
                continue

            ms           = cand.get("mom_scale", 1.0)
            risk_scaled  = base_risk * ms
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
                # Trail fields
                "trail_active": False,
                "trail_high":   0.0,
                "trail_stop":   initial_stop,   # starts same as hard stop
            }

    # Force-close remaining positions at end of data
    for ticker, pos in open_pos.items():
        d         = ind[ticker]
        last_c    = float(d["Close"].iloc[-1])
        fill_last = last_c * (1.0 - slippage)
        last_dt   = d.index[-1]
        pnl       = (fill_last - pos["entry"]) * pos["shares"] - comm_exit
        pnl_r     = (fill_last - pos["entry"]) / pos["initial_risk"]
        account  += pnl
        all_trades.append({
            "ticker":        ticker,
            "entry_date":    pos["entry_date"],
            "exit_date":     last_dt,
            "entry_price":   round(pos["entry"], 4),
            "exit_price":    round(fill_last, 4),
            "stop_initial":  round(pos["stop_initial"], 4),
            "target":        round(pos["target"], 4),
            "initial_risk":  round(pos["initial_risk"], 4),
            "shares":        pos["shares"],
            "risk_aud":      round(pos["risk_aud"], 2),
            "psm":           pos["psm"],
            "f_regime":      pos.get("f_regime", "STRONG_BULL"),
            "holding_days":  pos["bars_held"],
            "exit_type":     "end_of_data",
            "trail_active":  pos.get("trail_active", False),
            "trail_high":    round(pos.get("trail_high", 0.0), 4),
            "pnl_r":         round(pnl_r, 3),
            "pnl_aud":       round(pnl, 2),
            "account_after": round(account, 2),
        })

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
# Metrics  (win = pnl_r > 0 for consistency across both exit modes)
# ---------------------------------------------------------------------------

def calc_metrics(trades_df: pd.DataFrame, equity_df: pd.DataFrame,
                 label: str = "") -> dict:
    if trades_df.empty:
        return {}

    closed = trades_df[trades_df["exit_type"] != "end_of_data"].copy()
    if closed.empty:
        closed = trades_df.copy()

    n = len(closed)

    # Win / loss definitions unified: profit-based (works for both target and trail exits)
    wins      = closed[closed["pnl_r"] >  0]
    losses    = closed[closed["pnl_r"] <= 0]
    stop_exits = closed[closed["exit_type"].isin(["stop", "stop_gap"])]
    trail_exits = closed[closed["exit_type"] == "trail"]
    target_exits = closed[closed["exit_type"] == "target"]
    time_exits  = closed[closed["exit_type"] == "time"]
    gap_exits   = closed[closed["exit_type"] == "stop_gap"]

    win_rate   = len(wins) / n * 100 if n else 0

    # Profit factor: gross_win / |gross_loss|  (all closed trades)
    gross_win  = wins["pnl_aud"].sum()
    gross_loss = losses["pnl_aud"].sum()
    pf         = gross_win / abs(gross_loss) if gross_loss < 0 else np.nan

    avg_r       = closed["pnl_r"].mean()
    expectancy  = closed["pnl_aud"].mean()
    avg_hold    = closed["holding_days"].mean()
    total_pnl   = trades_df["pnl_aud"].sum()

    final_account = (trades_df["account_after"].iloc[-1]
                     if "account_after" in trades_df.columns
                     else STARTING_BALANCE + total_pnl)
    pct_return = (final_account / STARTING_BALANCE - 1) * 100
    cagr_years = 8.0
    cagr = ((final_account / STARTING_BALANCE) ** (1.0 / cagr_years) - 1) * 100

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

    # Avg R of winning trades (to see if trail is capturing bigger winners)
    avg_win_r  = float(wins["pnl_r"].mean())  if not wins.empty  else 0.0
    avg_loss_r = float(losses["pnl_r"].mean()) if not losses.empty else 0.0

    # Median R (less skewed by outliers)
    median_r = float(closed["pnl_r"].median())

    # R distribution: % of trades achieving >= 2R, >= 3R, >= 4R
    pct_ge2r = (closed["pnl_r"] >= 2.0).mean() * 100
    pct_ge3r = (closed["pnl_r"] >= 3.0).mean() * 100
    pct_ge4r = (closed["pnl_r"] >= 4.0).mean() * 100
    max_r    = float(closed["pnl_r"].max())

    # F-regime exit breakdown
    freg_breakdown: dict[str, dict] = {}
    if "f_regime" in closed.columns:
        for fr in ["STRONG_BULL", "WEAK_BULL", "CHOPPY_BEAR"]:
            sub  = closed[closed["f_regime"] == fr]
            n_fr = len(sub)
            freg_breakdown[fr] = {
                "n":        n_fr,
                "win_rate": round(len(sub[sub["pnl_r"] > 0]) / n_fr * 100, 1) if n_fr else 0,
                "avg_r":    round(float(sub["pnl_r"].mean()), 3) if n_fr else 0,
            }

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
        grp  = closed2[closed2["year"] == yr]
        n_y  = len(grp[grp["exit_type"] != "end_of_data"])
        w_y  = len(grp[grp["pnl_r"] > 0])
        wr_y = w_y / n_y * 100 if n_y else 0
        pnl_y = grp["pnl_aud"].sum()
        gw    = grp[grp["pnl_aud"] > 0]["pnl_aud"].sum()
        gl    = abs(grp[grp["pnl_aud"] < 0]["pnl_aud"].sum())
        pf_y  = round(gw / gl, 2) if gl > 0 else "n/a"
        acct_y = year_end_acct.get(yr, prev_acct + pnl_y)
        ret_y  = (acct_y / prev_acct - 1) * 100 if prev_acct > 0 else 0
        yearly[yr] = {
            "trades":     n_y,
            "win_pct":    round(wr_y, 1),
            "pnl_aud":    round(pnl_y, 2),
            "pf":         pf_y,
            "acct_end":   round(acct_y, 2),
            "return_pct": round(ret_y, 1),
        }
        prev_acct = acct_y

    return {
        "label":             label,
        "total_trades":      n,
        "win_rate_pct":      round(win_rate, 1),
        "wins":              len(wins),
        "losses":            len(losses),
        "target_exits":      len(target_exits),
        "trail_exits":       len(trail_exits),
        "stop_exits":        len(stop_exits),
        "gap_exits":         len(gap_exits),
        "time_exits":        len(time_exits),
        "profit_factor":     round(pf, 2) if not np.isnan(pf) else "n/a",
        "avg_r":             round(avg_r, 3),
        "avg_win_r":         round(avg_win_r, 3),
        "avg_loss_r":        round(avg_loss_r, 3),
        "median_r":          round(median_r, 3),
        "pct_ge2r":          round(pct_ge2r, 1),
        "pct_ge3r":          round(pct_ge3r, 1),
        "pct_ge4r":          round(pct_ge4r, 1),
        "max_r":             round(max_r, 3),
        "expectancy_aud":    round(expectancy, 2),
        "avg_hold_days":     round(avg_hold, 1),
        "total_pnl_aud":     round(total_pnl, 2),
        "final_account":     round(final_account, 2),
        "pct_return":        round(pct_return, 1),
        "cagr_pct":          round(cagr, 1),
        "calmar":            round(calmar, 2) if not np.isnan(calmar) else "n/a",
        "max_drawdown_aud":  round(max_dd_abs, 2),
        "max_drawdown_pct":  round(max_dd_pct, 1),
        "max_consec_losses": max_consec,
        "sharpe":            round(sharpe, 2) if not np.isnan(sharpe) else "n/a",
        "best_trade_r":      round(float(closed["pnl_r"].max()), 3),
        "worst_trade_r":     round(float(closed["pnl_r"].min()), 3),
        "freg_breakdown":    freg_breakdown,
        "yearly":            yearly,
    }


# ---------------------------------------------------------------------------
# Print comparison table
# ---------------------------------------------------------------------------

def print_comparison(results: list[tuple[str, dict]]) -> None:
    KEY_METRICS = [
        ("total_trades",      "Trades"),
        ("win_rate_pct",      "Win rate %"),
        ("wins",              "  Profitable exits"),
        ("losses",            "  Losing exits"),
        ("target_exits",      "  Target hits (2:1)"),
        ("trail_exits",       "  Trail stop exits"),
        ("stop_exits",        "  Hard stop exits"),
        ("gap_exits",         "  Gap stop exits"),
        ("time_exits",        "  Time stop exits"),
        ("profit_factor",     "Profit factor"),
        ("avg_r",             "Avg R / trade"),
        ("avg_win_r",         "  Avg R (winners)"),
        ("avg_loss_r",        "  Avg R (losers)"),
        ("median_r",          "Median R"),
        ("pct_ge2r",          "  % trades >= 2R"),
        ("pct_ge3r",          "  % trades >= 3R"),
        ("pct_ge4r",          "  % trades >= 4R"),
        ("max_r",             "  Best trade R"),
        ("expectancy_aud",    "Expectancy $"),
        ("avg_hold_days",     "Avg hold days"),
        ("total_pnl_aud",     "Total P&L $"),
        ("final_account",     "Final account $"),
        ("pct_return",        "Total return %"),
        ("cagr_pct",          "CAGR %"),
        ("max_drawdown_aud",  "Max drawdown $"),
        ("max_drawdown_pct",  "Max drawdown %"),
        ("calmar",            "Calmar ratio"),
        ("max_consec_losses", "Max consec. losses"),
        ("sharpe",            "Sharpe"),
    ]
    LOWER_IS_BETTER = {"losses", "stop_exits", "gap_exits", "time_exits",
                       "max_drawdown_aud", "max_drawdown_pct", "max_consec_losses",
                       "avg_loss_r"}

    labels  = [lbl for lbl, _ in results]
    col_w   = max(32, max(len(l) for l in labels) + 4)
    met_w   = 24
    sep     = "=" * (met_w + col_w * len(labels) + 4)

    print(f"\n{sep}")
    print(f"  EXIT STRATEGY COMPARISON  (Variant F)  --  {date.today()}")
    print(f"  ${STARTING_BALANCE:,} start | {RISK_PCT*100:.1f}% risk/trade | "
          f"max {MAX_POSITIONS} positions | IBKR $6/leg | 0.2% slip | gap risk")
    print(f"  F-Fixed : fixed 2:1 target + breakeven ratchet @ 1R")
    print(f"  F-Trail : 2xATR trailing stop activating at 1R (no fixed target)")
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
            if k in ("pct_return", "cagr_pct"):
                return f"{v:+.1f}%"
            if k == "max_drawdown_pct":
                return f"{v:.1f}%"
            if k in ("win_rate_pct", "pct_ge2r", "pct_ge3r", "pct_ge4r"):
                return f"{v:.1f}%"
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
    print("  * = best value in row\n")

    # F-regime breakdown
    print(f"  F-REGIME PERFORMANCE BREAKDOWN")
    print("-" * (met_w + col_w * len(results) + 4))
    hdr2 = f"  {'F-Regime':<{met_w}}"
    for lbl in labels:
        hdr2 += f"  {lbl:^{col_w - 2}}"
    print(hdr2)
    print("-" * (met_w + col_w * len(results) + 4))

    for fr in ["STRONG_BULL", "WEAK_BULL", "CHOPPY_BEAR"]:
        for sub_key, sub_lbl in [("n", "trades"), ("win_rate", "win%"), ("avg_r", "avg R")]:
            display = f"  {fr} - {sub_lbl}"
            row = f"  {display:<{met_w}}"
            for _, m in results:
                bd  = m.get("freg_breakdown", {}).get(fr, {})
                val = bd.get(sub_key, "-")
                sfx = "%" if sub_key == "win_rate" else ""
                cell = f"{val}{sfx}" if isinstance(val, (int, float)) else "-"
                row += f"  {cell:^{col_w - 2}}"
            print(row)

    print("-" * (met_w + col_w * len(results) + 4))
    print()

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
        cells   = []
        yr_vals = []
        for _, m in results:
            y = m.get("yearly", {}).get(yr)
            if y:
                sign = "+" if y["pnl_aud"] >= 0 else ""
                cell = f"{sign}${y['pnl_aud']:,.0f} ({y['return_pct']:+.1f}%)"
                yr_vals.append(y["pnl_aud"])
            else:
                cell = "-"
                yr_vals.append(None)
            cells.append(cell)

        num_yr  = [(i, v) for i, v in enumerate(yr_vals) if v is not None]
        best_yi = max(num_yr, key=lambda x: x[1])[0] if num_yr else -1

        row = f"  {str(yr):<{met_w}}"
        for i, cell in enumerate(cells):
            marker = " *" if i == best_yi and len(num_yr) > 1 else "  "
            row   += f"  {cell:^{col_w - 4}}{marker}"
        print(row)

    print("=" * (met_w + col_w * len(labels) + 4))
    print("  * = better year\n")


# ---------------------------------------------------------------------------
# R-distribution histogram + equity chart
# ---------------------------------------------------------------------------

def save_chart(results: list[tuple[str, dict]],
               equity_dfs: list[pd.DataFrame],
               trades_dfs: list[pd.DataFrame],
               out_path: Path) -> None:
    BG, PANEL = "#0d1117", "#161b22"
    PALETTE   = ["#58a6ff", "#26a641"]
    GREY, RED = "#8b949e", "#da3633"

    fig = plt.figure(figsize=(22, 20), facecolor=BG)
    gs  = gridspec.GridSpec(3, 2, figure=fig,
                            hspace=0.44, wspace=0.28,
                            height_ratios=[3, 2.5, 2])
    ax_eq   = fig.add_subplot(gs[0, :])        # full-width equity
    ax_hist = [fig.add_subplot(gs[1, i]) for i in range(2)]   # R distributions
    ax_yr   = [fig.add_subplot(gs[2, i]) for i in range(2)]   # year-by-year bars

    for ax in [ax_eq] + ax_hist + ax_yr:
        ax.set_facecolor(PANEL)
        ax.tick_params(colors=GREY, labelsize=8)
        for sp in ax.spines.values():
            sp.set_edgecolor("#21262d")

    # ── Equity curves ─────────────────────────────────────────────────────────
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
        f"Equity Curves | Variant F  Fixed 2:1  vs  2xATR Trailing Stop  "
        f"|  $20,000 start  |  {date.today()}",
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

    # ── R-distribution histograms ─────────────────────────────────────────────
    for ax, (label, m), tdf, colour in zip(ax_hist, results, trades_dfs, PALETTE):
        if tdf.empty:
            continue
        closed = tdf[tdf["exit_type"] != "end_of_data"]["pnl_r"].values
        bins   = np.arange(-3, 8.5, 0.5)
        # Draw histogram with per-bar colouring via manual patches
        counts, edges = np.histogram(closed, bins=bins)
        for cnt, left, right in zip(counts, edges[:-1], edges[1:]):
            bar_colour = colour if left >= 0 else RED
            ax.bar(left + (right - left) / 2, cnt, width=(right - left) * 0.88,
                   color=bar_colour, edgecolor="#21262d", linewidth=0.4, alpha=0.85)
        ax.axvline(0,   color=GREY, lw=0.8, ls="--", alpha=0.6)
        ax.axvline(m.get("avg_r", 0), color=colour, lw=1.2, ls="-",
                   label=f"Avg R = {m.get('avg_r', 0):.3f}")

        # Annotate key stats
        wr   = m.get("win_rate_pct", 0)
        pf   = m.get("profit_factor", "n/a")
        avgr = m.get("avg_r", 0)
        pf_str = f"{pf:.2f}" if isinstance(pf, float) else str(pf)
        label_safe = label.replace("$", r"\$")
        ax.set_title(
            f"R Distribution  |  {label_safe}\n"
            f"WR {wr:.1f}%  |  PF {pf_str}  |  Avg R {avgr:.3f}  |  n={len(closed)}",
            color="white", fontsize=9, pad=5)
        ax.set_xlabel("R multiple", color=GREY, fontsize=8)
        ax.set_ylabel("Frequency", color=GREY, fontsize=8)
        ax.legend(fontsize=8, facecolor=PANEL, edgecolor=GREY, labelcolor="white")
        ax.grid(color="#21262d", lw=0.5, alpha=0.6, axis="y")

        # Annotate >=2R, >=3R, >=4R percentages
        pct2 = m.get("pct_ge2r", 0)
        pct3 = m.get("pct_ge3r", 0)
        pct4 = m.get("pct_ge4r", 0)
        txt  = f">=2R: {pct2:.0f}%\n>=3R: {pct3:.0f}%\n>=4R: {pct4:.0f}%"
        ax.text(0.97, 0.97, txt, transform=ax.transAxes,
                ha="right", va="top", color=GREY, fontsize=8,
                bbox=dict(facecolor="#0d1117", alpha=0.7, edgecolor="none"))

    # ── Year-by-year bars ─────────────────────────────────────────────────────
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
        f"Variant F: STRONG_BULL 1.0x / WEAK_BULL 0.8x / CHOPPY_BEAR hard-filter  |  "
        f"ATR>={MIN_ATR_PCT}%  |  RS top 20%  |  >=2 triggers  |  close entry  |  "
        f"IBKR \\$6/leg  |  0.2% slip  |  gap risk  |  20-bar time stop  |  "
        f"Trail: 2xATR trailing stop activates at +1R"
    )
    fig.text(0.5, 0.002, footer, color=GREY, fontsize=7, ha="center")

    plt.savefig(out_path, dpi=130, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"  Chart saved -> {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Variant F: fixed 2:1 target vs 2xATR trailing stop")
    p.add_argument("--period", default="9y",
                   help="yfinance period string (default: 9y)")
    return p.parse_args()


def main():
    args = parse_args()

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    today_str = date.today().strftime("%Y%m%d")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("  ASX SWING ENGINE  --  EXIT STRATEGY COMPARISON")
    print(f"  Variant F: Fixed 2:1  vs  2xATR Trailing Stop")
    print(f"  {date.today()}  |  period={args.period}  |  ${STARTING_BALANCE:,} start")
    print("=" * 72)
    print()

    # ── 1. XJO ───────────────────────────────────────────────────────────────
    print("Step 1/5  Downloading XJO benchmark ...")
    with _quiet():
        xjo_raw = yf.download(BENCHMARK, period=args.period, interval="1d",
                              auto_adjust=True, progress=False)
    if isinstance(xjo_raw.columns, pd.MultiIndex):
        xjo_raw.columns = [c[0] for c in xjo_raw.columns]
    xjo_df    = xjo_raw[["Open","High","Low","Close","Volume"]].dropna()
    xjo_close = xjo_df["Close"]
    print(f"  XJO: {len(xjo_close)} bars  "
          f"({xjo_close.index[0].date()} to {xjo_close.index[-1].date()})\n")

    # ── 2. Regime series ──────────────────────────────────────────────────────
    print("Step 2/5  Building Variant-F regime series ...")
    regime_series   = build_regime_series(xjo_df, args.period)
    f_regime_series = regime_series["f_regime"]
    print()

    # ── 3. Universe ───────────────────────────────────────────────────────────
    tickers = list(dict.fromkeys(ASX_PRESET))
    print(f"Step 3/5  Downloading {len(tickers)}-stock universe ({args.period}) ...")
    price_data = download_universe(tickers, args.period)
    print(f"  Downloaded {len(price_data)}/{len(tickers)} tickers.\n")
    if not price_data:
        print("ERROR: no price data. Check network.")
        sys.exit(1)

    # ── 4. Build single Variant-F signal pool (shared by both scenarios) ──────
    print("Step 4/5  Building Variant-F signal pool ...")
    ind, by_entry = build_signal_pool_f(price_data, xjo_close, f_regime_series)
    print()

    # ── 5. Run both simulations on the same pool ──────────────────────────────
    realistic = dict(comm_entry=6.0, comm_exit=6.0, slippage=0.002, gap_risk=True)

    scenarios = [
        {
            "label":  "F-Fixed   (2:1 target + BE@1R)",
            "params": dict(stop_mult=2.0, target_r=2.0, trail_atr_mult=0.0),
        },
        {
            "label":  "F-Trail   (2xATR trail, activates @1R)",
            "params": dict(stop_mult=2.0, target_r=2.0, trail_atr_mult=TRAIL_ATR_MULT),
        },
    ]

    all_results: list[tuple[str, dict]] = []
    all_equity:  list[pd.DataFrame]     = []
    all_trades:  list[pd.DataFrame]     = []

    print("Step 5/5  Simulating scenarios ...")
    for sc in scenarios:
        lbl = sc["label"]
        print(f"\n  -- {lbl} --")
        trades_df, equity_df = _simulate(ind, by_entry, **realistic, **sc["params"])

        if trades_df.empty:
            print("  WARNING: no trades generated.")
            continue

        m = calc_metrics(trades_df, equity_df, label=lbl)
        all_results.append((lbl, m))
        all_equity.append(equity_df)
        all_trades.append(trades_df)

        slug = "fixed" if "Fixed" in lbl else "trail"
        trades_df.to_csv(RESULTS_DIR / f"trades_trail_{slug}_{today_str}.csv", index=False)
        equity_df.to_csv(RESULTS_DIR / f"equity_trail_{slug}_{today_str}.csv", index=False)

        print(f"  Trades: {m['total_trades']}  |  "
              f"Win rate: {m['win_rate_pct']}%  |  "
              f"PF: {m['profit_factor']}  |  "
              f"Avg R: {m['avg_r']}  |  "
              f"CAGR: {m['cagr_pct']}%  |  "
              f"MaxDD: {m['max_drawdown_pct']}%  |  "
              f"Sharpe: {m['sharpe']}")
        if m["trail_exits"] > 0:
            print(f"  Trail exits: {m['trail_exits']}  |  "
                  f">=2R: {m['pct_ge2r']:.1f}%  "
                  f">=3R: {m['pct_ge3r']:.1f}%  "
                  f">=4R: {m['pct_ge4r']:.1f}%  |  "
                  f"Best R: {m['best_trade_r']}")

    print()
    if all_results:
        print_comparison(all_results)
        chart_path = RESULTS_DIR / f"backtest_trail_vs_fixed_{today_str}.png"
        save_chart(all_results, all_equity, all_trades, chart_path)
        print(f"Results saved to {RESULTS_DIR}/")
    else:
        print("No results to compare.")


if __name__ == "__main__":
    main()
