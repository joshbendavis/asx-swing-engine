# -*- coding: utf-8 -*-
"""
backtest_full.py - ASX Swing Engine | Definitive 8-Year Validation Backtest
============================================================================
Compares two scenarios side by side over 2017-2026:

  A. Idealized   — $10 round-trip commission, 0% slippage, no gap risk
                   (previous benchmark — theoretical upper bound)
  B. Realistic   — IBKR $6/leg ($12 round-trip), 0.2% slippage on every fill,
                   gap risk (if open gaps below stop, exit at open)

Both scenarios use identical signal pool:
  • RS top 20% cross-sectional filter (63-day excess return vs XJO)
  • ≥2 entry triggers (RSI bounce / MACD cross / volume breakout ≥1.5×)
  • Flat 1.5% risk per trade (PSM = 1.0 always — regime informational only)
  • Entry filled at NEXT DAY OPEN (not signal-day close)

Portfolio rules (match live engine exactly):
  Starting capital : $20,000 AUD
  Risk per trade   : 1.5% of current account balance
  Max concurrent   : 6 open positions
  Stop             : entry − 2 × ATR(14)   [signal-bar ATR]
  Target           : entry + 2 × initial_risk  (2:1 R:R)
  Breakeven        : stop → entry once close ≥ entry + 1R
  Time stop        : 20 bars

Realistic fill model:
  Entry   : open_next_day × 1.002  (+0.2% slippage, buy at market)
  Exit    : fill × 0.998           (−0.2% slippage, sell at market/stop)
  Target  : target_price × 0.998  (limit order, fill near target)
  Gap     : if open ≤ stop → exit at open × 0.998  (gap through stop)
  Broker  : $6 deducted at entry + $6 at exit (IBKR ASX minimum)

Usage:
  python backtest_full.py               # 2017-2026 (default period 9y)
  python backtest_full.py --period 5y   # shorter run
"""

from __future__ import annotations

import argparse
import contextlib
import json
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
TIME_STOP_DAYS    = 20
# Commission / slippage defaults — overridden per scenario in main()
COMMISSION_ENTRY  = 6.0         # IBKR minimum per ASX order leg
COMMISSION_EXIT   = 6.0         # IBKR minimum per ASX order leg
SLIPPAGE_PCT      = 0.002       # 0.2% adverse slippage on all fills
MIN_RISK_AUD      = 75.0        # mirror risk_engine.py
SECTOR_CAP        = 2           # mirror risk_engine.py

# Screener hard filters
MIN_PRICE         = 0.50
MAX_PRICE         = 50.0
MIN_AVG_VOL_20    = 200_000
RSI_MIN, RSI_MAX  = 40, 65
MAX_EMA50_DIST    = 10.0        # %
MIN_ATR_PCT       = 3.0
EMA_FAST, EMA_SLOW = 50, 200
RS_PERIOD          = 63
WARMUP             = EMA_SLOW + RS_PERIOD + 10

# Regime thresholds (match regime_detector.py)
HIGH_VOL_ATR      = 2.5
CHOPPY_SLOPE      = 0.1
CHOPPY_BREADTH    = 50.0

BATCH_SIZE  = 20
BATCH_DELAY = 1.0
BENCHMARK   = "^AXJO"

RESULTS_DIR      = Path("results/backtest")
SECTOR_CACHE_PATH = Path("results/sector_cache.json")

# 40-stock ASX200 breadth sample (matches regime_detector.py)
ASX200_SAMPLE = [
    "BHP.AX","CBA.AX","CSL.AX","NAB.AX","WBC.AX","ANZ.AX","WES.AX","MQG.AX",
    "RIO.AX","WOW.AX","GMG.AX","TLS.AX","FMG.AX","REA.AX","ALL.AX","S32.AX",
    "STO.AX","ORI.AX","AMC.AX","TCL.AX","QBE.AX","IAG.AX","SHL.AX","COL.AX",
    "ASX.AX","MPL.AX","APA.AX","JHX.AX","SGP.AX","MIN.AX","IGO.AX","PLS.AX",
    "LYC.AX","WHC.AX","NXT.AX","ALX.AX","ALD.AX","NCM.AX","AGL.AX","SUL.AX",
]

# Universe: preset + breadth sample combined (duplicates removed)
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
    "ALL.AX","ASX.AX","ALD.AX","NCM.AX",   # extras from breadth sample
]


# ---------------------------------------------------------------------------
# Helpers
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
    """Download OHLCV for a batch; return {ticker: DataFrame}."""
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
# Regime series — historical classification per bar
# ---------------------------------------------------------------------------

def build_regime_series(xjo_df: pd.DataFrame, period: str) -> pd.DataFrame:
    """
    Compute per-bar regime classification for the full backtest period.

    Downloads ASX200 breadth sample, computes market breadth per bar,
    then classifies each bar using the same logic as regime_detector.py.

    Returns a DataFrame indexed by date with columns:
        regime, confidence, psm, require_all_triggers, min_triggers, blocked
    """
    print("Building historical regime series ...")

    close  = xjo_df["Close"]
    high   = xjo_df["High"]
    low    = xjo_df["Low"]

    ema200 = _ema(close, 200)
    ema50  = _ema(close, 50)
    atr_s  = _atr(high, low, close, 14)

    above_200  = close > ema200          # bool series
    atr_pct    = atr_s / close * 100
    roc_20     = close.pct_change(20) * 100
    ema50_slope = (ema50 - ema50.shift(5)) / ema50.shift(5) * 100   # 5-bar slope %

    # ── Market breadth: download breadth sample ───────────────────────────────
    print(f"  Downloading {len(ASX200_SAMPLE)} ASX200 breadth tickers ...")
    breadth_data: dict[str, pd.DataFrame] = {}
    b_batches = [ASX200_SAMPLE[i:i+BATCH_SIZE] for i in range(0, len(ASX200_SAMPLE), BATCH_SIZE)]
    for b_idx, batch in enumerate(b_batches, 1):
        breadth_data.update(_download_batch(batch, period, min_rows=50))
        if b_idx < len(b_batches):
            time.sleep(BATCH_DELAY)
    print(f"  Breadth data: {len(breadth_data)}/{len(ASX200_SAMPLE)} tickers.")

    # Build per-bar breadth: % of sample above their own 50 EMA
    all_dates = close.index
    breadth_above = pd.DataFrame(index=all_dates, dtype=float)
    for ticker, bdf in breadth_data.items():
        bc   = bdf["Close"].reindex(all_dates)
        bema = _ema(bc.ffill(), 50)
        breadth_above[ticker] = (bc > bema).astype(float)
        breadth_above.loc[bc.isna(), ticker] = np.nan

    # breadth per bar = mean of non-NaN values * 100
    market_breadth = breadth_above.mean(axis=1) * 100   # 0-100 Series

    # ── Per-bar classification ────────────────────────────────────────────────
    rows = []
    dates = close.index

    for dt in dates:
        a200  = bool(above_200.loc[dt])
        atp   = float(atr_pct.loc[dt])     if not pd.isna(atr_pct.loc[dt])     else 0.0
        r20   = float(roc_20.loc[dt])      if not pd.isna(roc_20.loc[dt])      else 0.0
        slope = float(ema50_slope.loc[dt]) if not pd.isna(ema50_slope.loc[dt]) else 0.0
        brd   = float(market_breadth.loc[dt]) if not pd.isna(market_breadth.loc[dt]) else 50.0
        cl    = float(close.loc[dt])
        e200  = float(ema200.loc[dt])

        # Classify
        if atp > HIGH_VOL_ATR:
            regime = "HIGH_VOL"
        elif not a200:
            regime = "BEAR"
        elif abs(slope) < CHOPPY_SLOPE and brd < CHOPPY_BREADTH:
            regime = "CHOPPY"
        elif r20 < 0 or slope < 0:
            regime = "WEAK_BULL"
        else:
            regime = "BULL"

        # Confidence (simplified vectorised version of regime_detector._calc_confidence)
        if regime == "BULL":
            conf = 100
            if brd < 60:  conf -= 20
            if r20 < 2:   conf -= 15
            if slope < 0.1: conf -= 10
        elif regime == "WEAK_BULL":
            conf = 65
            if r20 < 0 and slope < 0:  conf -= 15
            elif r20 < 0 or slope < 0: conf -= 5
            if brd > 60:               conf += 10
            if abs(slope) < 0.05:      conf -= 10
        elif regime == "BEAR":
            conf = 100
            if e200 > 0 and abs(cl - e200) / e200 * 100 < 1.0: conf -= 20
            if r20 > -2:  conf -= 10
        elif regime == "CHOPPY":
            conf = 70
            if brd < 40:        conf += 15
            if abs(slope) < 0.05: conf += 15
        elif regime == "HIGH_VOL":
            conf = 100
            if atp < 3.0: conf -= 20
        else:
            conf = 50

        conf = int(max(0, min(100, conf)))

        # Flat PSM per regime — no hard blocking, BEAR still trades at 0.5×
        psm = {
            "BULL":      1.00,
            "WEAK_BULL": 0.75,
            "CHOPPY":    0.60,
            "BEAR":      0.50,
            "HIGH_VOL":  0.25,
        }.get(regime, 1.0)

        # Dual momentum penalty → require all 3 triggers (quality gate, not a block)
        dual_penalty = (r20 < 0) and (slope < 0)
        require_all  = dual_penalty or (regime == "CHOPPY")
        min_trig     = 3 if require_all else 2

        rows.append({
            "date":                dt,
            "regime":              regime,
            "confidence":          conf,
            "psm":                 psm,
            "require_all_triggers":require_all,
            "min_triggers":        min_trig,
            "market_breadth":      brd,
            "blocked":             False,   # regime never blocks
        })

    df = pd.DataFrame(rows).set_index("date")
    rc = df["regime"].value_counts()
    print(f"  Regime distribution: {dict(rc)}")
    psm_dist = df.groupby("psm").size().sort_index()
    print(f"  PSM distribution: {dict(psm_dist)}")
    print(f"  Avg PSM across period: {df['psm'].mean():.3f}")
    return df


# ---------------------------------------------------------------------------
# Sector map — fetch from yfinance, persist to sector_cache.json
# ---------------------------------------------------------------------------

def fetch_sector_map(tickers: list[str]) -> dict[str, str]:
    """Return {ticker: GICS_sector} for all tickers. Uses/updates sector_cache.json."""
    cache: dict[str, str] = {}
    if SECTOR_CACHE_PATH.exists():
        try:
            with open(SECTOR_CACHE_PATH) as fh:
                cache = json.load(fh)
        except Exception:
            pass

    missing = [t for t in tickers if t not in cache]
    if missing:
        print(f"  Fetching sector for {len(missing)} uncached tickers ...")
        for i, ticker in enumerate(missing, 1):
            try:
                info   = yf.Ticker(ticker).info
                sector = info.get("sector") or "Unknown"
            except Exception:
                sector = "Unknown"
            cache[ticker] = sector
            if i % 10 == 0:
                print(f"    {i}/{len(missing)} ...")
            time.sleep(0.05)   # gentle throttle

        try:
            SECTOR_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
            with open(SECTOR_CACHE_PATH, "w") as fh:
                json.dump(cache, fh, indent=2)
            print(f"  Sector cache updated ({len(cache)} entries).")
        except Exception as exc:
            print(f"  WARNING: could not save sector cache: {exc}")

    return {t: cache.get(t, "Unknown") for t in tickers}


# ---------------------------------------------------------------------------
# Indicators
# ---------------------------------------------------------------------------

def compute_indicators(df: pd.DataFrame, xjo_close: pd.Series,
                       xjo_uptrend: pd.Series) -> pd.DataFrame:
    d = df.copy()
    d["ema50"]      = _ema(d["Close"], EMA_FAST)
    d["ema200"]     = _ema(d["Close"], EMA_SLOW)
    d["rsi"]        = _rsi(d["Close"])
    d["atr"]        = _atr(d["High"], d["Low"], d["Close"])
    d["avg_vol"]    = d["Volume"].rolling(20).mean()
    d["ema50_dist"] = (d["Close"] / d["ema50"] - 1) * 100
    d["atr_pct"]    = d["atr"] / d["Close"] * 100
    d["stock_r63"]  = d["Close"].pct_change(RS_PERIOD)
    xjo_r63        = xjo_close.pct_change(RS_PERIOD)
    d["xjo_r63"]   = xjo_r63.reindex(d.index, method="ffill")
    d["rs_vs_xjo"] = d["stock_r63"] - d["xjo_r63"]
    d["xjo_uptrend"] = xjo_uptrend.reindex(d.index, method="ffill").fillna(True)

    # Entry triggers
    rsi_prev    = d["rsi"].shift(1)
    t_rsi       = (rsi_prev >= 40) & (rsi_prev <= 55) & (d["rsi"] > rsi_prev)
    macd_line   = _ema(d["Close"], 12) - _ema(d["Close"], 26)
    macd_sig    = _ema(macd_line, 9)
    t_macd      = (macd_line.shift(1) < macd_sig.shift(1)) & (macd_line > macd_sig)
    t_vol       = d["Volume"] >= 1.5 * d["avg_vol"]
    d["trigger_count"] = t_rsi.astype(int) + t_macd.astype(int) + t_vol.astype(int)
    return d


def find_signals(d: pd.DataFrame,
                 min_atr_pct: float = MIN_ATR_PCT) -> pd.Series:
    """Hard screener filters — same as live screener.

    min_atr_pct can be overridden per scenario (e.g. 4.0 for the tighter filter).
    """
    return (
        (d["Close"]      >  MIN_PRICE)     &
        (d["Close"]      <= MAX_PRICE)     &
        (d["avg_vol"]    >  MIN_AVG_VOL_20) &
        (d["Close"]      >  d["ema200"])   &
        (d["Close"]      >  d["ema50"])    &
        (d["ema50_dist"] <= MAX_EMA50_DIST) &
        (d["rsi"]        >= RSI_MIN)       &
        (d["rsi"]        <= RSI_MAX)       &
        (d["atr_pct"]    >= min_atr_pct)   &
        (d["rs_vs_xjo"]  >  0)            &
        (d["xjo_uptrend"] == True)
    )


# ---------------------------------------------------------------------------
# Portfolio simulator — scenario A (RS top 20%, no regime)
# ---------------------------------------------------------------------------

def _build_signal_pool(price_data: dict[str, pd.DataFrame],
                       xjo_close: pd.Series,
                       min_triggers: int = 2,
                       min_atr_pct:  float = MIN_ATR_PCT,
                       entry_at_close: bool = False) -> tuple[dict, dict]:
    """
    Pre-scan signal pool for a scenario.

    Parameters
    ----------
    min_atr_pct     : ATR% floor at signal time (default 3.0; tighter = 4.0).
    entry_at_close  : if True, fill at signal-bar Close (3:30pm proxy) instead
                      of next-day Open.  Both still apply slippage in _simulate.
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

    entry_desc = "signal-day close" if entry_at_close else "next-day open"
    print(f"  Scanning (RS top 20% + >={min_triggers} triggers, "
          f"ATR>={min_atr_pct}%, entry={entry_desc}) ...")
    by_entry: dict = defaultdict(list)
    for ticker, d in ind.items():
        sigs = find_signals(d, min_atr_pct=min_atr_pct)
        n    = len(d)
        for i in range(WARMUP, n - 1):   # n-1 keeps a buffer on both paths
            if not sigs.iloc[i]:
                continue
            rv  = float(d["rs_vs_xjo"].values[i])
            thr = rs_pct80.get(d.index[i], -np.inf)
            if np.isnan(rv) or rv < thr:
                continue
            tc  = int(d["trigger_count"].values[i])
            if tc < min_triggers:
                continue
            atr = float(d["atr"].values[i])
            if atr <= 0:
                continue
            if entry_at_close:
                # Fill at today's close — proxy for a 3:30pm market order
                ep       = float(d["Close"].values[i])
                entry_dt = d.index[i]
            else:
                # Fill at next day's open — standard overnight entry
                ep       = float(d["Open"].values[i + 1])
                entry_dt = d.index[i + 1]
            if ep <= 0:
                continue
            by_entry[entry_dt].append({
                "ticker": ticker, "entry_open": ep, "atr": atr,
                "rs_val": rv, "trigger_count": tc,
            })

    total = sum(len(v) for v in by_entry.values())
    print(f"  {total} signals in pool across {len(ind)} tickers.")
    return ind, by_entry


def run_baseline(price_data: dict[str, pd.DataFrame],
                 xjo_close: pd.Series) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Scenario A: RS top 20% + >=2 triggers, full size (PSM=1.0 always)."""
    ind, by_entry = _build_signal_pool(price_data, xjo_close, min_triggers=2)
    return _simulate(ind, by_entry, psm_series=None)


# ---------------------------------------------------------------------------
# Portfolio simulator — scenario B (full system)
# ---------------------------------------------------------------------------

def run_full_system(price_data: dict[str, pd.DataFrame],
                    xjo_close: pd.Series,
                    regime_series: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Scenario B: identical signal pool as baseline, PSM scaled by regime on entry date.
    Same 166 trades as scenario A — only position sizes differ.
    """
    ind, by_entry = _build_signal_pool(price_data, xjo_close, min_triggers=2)
    return _simulate(ind, by_entry, psm_series=regime_series)


# ---------------------------------------------------------------------------
# Core simulation loop (shared)
# ---------------------------------------------------------------------------


def _simulate(ind: dict[str, pd.DataFrame],
              by_entry: dict,
              psm_series: pd.DataFrame | None = None,
              comm_entry: float = 0.0,
              comm_exit:  float = 10.0,
              slippage:   float = 0.0,
              gap_risk:   bool  = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Core simulation loop.

    Parameters
    ----------
    ind         : {ticker: indicator DataFrame} from compute_indicators()
    by_entry    : {entry_date: [candidate dicts]} pre-filtered signal pool
    psm_series  : optional DataFrame indexed by date with 'regime' and 'psm' columns.
                  When None, PSM = 1.0 always (flat sizing).
    comm_entry  : brokerage deducted at entry ($AUD).  Default 0 (idealized).
    comm_exit   : brokerage deducted at exit ($AUD).   Default 10 (old round-trip).
    slippage    : fractional adverse fill adjustment.  Default 0 (idealized).
                  Entry fills at open * (1 + slippage); exits at price * (1 - slippage).
                  For target exits, fill is at target_price * (1 - slippage).
    gap_risk    : if True, check each bar's open against the stop before intraday checks.
                  If open <= stop, exit at open * (1 - slippage) as 'stop_gap'.
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

        # ── A. Exit check ─────────────────────────────────────────────────────
        to_close = []
        for ticker, pos in open_pos.items():
            d = ind[ticker]
            if dt not in d.index:
                pos["bars_held"] += 1
                continue

            o = float(d["Open"].loc[dt])   # for gap-risk check
            c = float(d["Close"].loc[dt])

            fill_price: float | None = None
            exit_type:  str   | None = None

            # 1. Gap risk — open gaps below stop → exit at open (gapped through)
            #    Check this BEFORE breakeven update; if it gapped, we're already out.
            if gap_risk and o <= pos["stop"]:
                fill_price = o * (1.0 - slippage)
                exit_type  = "stop_gap"
            else:
                # Breakeven ratchet (evaluated on close)
                if not pos["be_set"] and c >= pos["be_lvl"]:
                    pos["stop"]   = pos["entry"]
                    pos["be_set"] = True

                # End-of-day exit triggers
                if c <= pos["stop"]:
                    fill_price = c * (1.0 - slippage)
                    exit_type  = "stop"
                elif c >= pos["target"]:
                    # Fill at target price (limit order), not close — more realistic
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
                    "stop":          round(pos["stop"], 4),
                    "target":        round(pos["target"], 4),
                    "initial_risk":  round(pos["initial_risk"], 4),
                    "shares":        pos["shares"],
                    "risk_aud":      round(pos["risk_aud"], 2),
                    "psm":           pos["psm"],
                    "regime":        pos["regime"],
                    "holding_days":  pos["bars_held"],
                    "exit_type":     exit_type,
                    "breakeven_hit": pos["be_set"],
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

        # ── B. Mark-to-market equity ──────────────────────────────────────────
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

        # Regime PSM lookup (PSM-only mode — no blocking, no re-filtering)
        if psm_series is not None and dt in psm_series.index:
            reg        = psm_series.loc[dt]
            regime_now = str(reg["regime"])
            psm_now    = float(reg["psm"])
        else:
            regime_now = "BULL"
            psm_now    = 1.0

        base_risk    = account * RISK_PCT
        risk_aud_bar = base_risk * psm_now

        # Sort candidates by rs_val descending — highest RS gets first slot
        candidates = sorted(by_entry[dt], key=lambda x: x.get("rs_val", 0), reverse=True)

        for cand in candidates:
            if len(open_pos) >= MAX_POSITIONS:
                break
            ticker = cand["ticker"]
            if ticker in open_pos:
                continue
            if dt_i < cooldown.get(ticker, 0):
                continue

            # Min risk floor: skip if scaled risk < $75
            if risk_aud_bar < MIN_RISK_AUD:
                continue

            ep           = cand["entry_open"]
            ep_filled    = ep * (1.0 + slippage)   # adverse slippage on buy
            atr          = cand["atr"]
            initial_risk = atr * ATR_STOP_MULT      # dollar risk/share (ATR-based)
            shares       = int(risk_aud_bar / initial_risk)
            if shares == 0:
                continue

            account -= comm_entry   # deduct $6 entry brokerage immediately

            open_pos[ticker] = {
                "entry_date":   dt,
                "entry":        ep_filled,
                "shares":       shares,
                "initial_risk": initial_risk,
                "risk_aud":     risk_aud_bar,
                "psm":          psm_now,
                "regime":       regime_now,
                "stop":         ep_filled - initial_risk,
                "target":       ep_filled + TARGET_R * initial_risk,
                "be_lvl":       ep_filled + BREAKEVEN_R * initial_risk,
                "be_set":       False,
                "bars_held":    0,
            }

    # Force-close remaining at end of data (apply slippage as if selling at market)
    for ticker, pos in open_pos.items():
        d          = ind[ticker]
        last_c     = float(d["Close"].iloc[-1])
        fill_last  = last_c * (1.0 - slippage)
        last_dt    = d.index[-1]
        pnl        = (fill_last - pos["entry"]) * pos["shares"] - comm_exit
        pnl_r      = (fill_last - pos["entry"]) / pos["initial_risk"]
        account   += pnl
        all_trades.append({
            "ticker":        ticker,
            "entry_date":    pos["entry_date"],
            "exit_date":     last_dt,
            "entry_price":   round(pos["entry"], 4),
            "exit_price":    round(fill_last, 4),
            "stop":          round(pos["stop"], 4),
            "target":        round(pos["target"], 4),
            "initial_risk":  round(pos["initial_risk"], 4),
            "shares":        pos["shares"],
            "risk_aud":      round(pos["risk_aud"], 2),
            "psm":           pos["psm"],
            "regime":        pos["regime"],
            "holding_days":  pos["bars_held"],
            "exit_type":     "end_of_data",
            "breakeven_hit": pos["be_set"],
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
# Metrics
# ---------------------------------------------------------------------------

def calc_metrics(trades_df: pd.DataFrame, equity_df: pd.DataFrame) -> dict:
    if trades_df.empty:
        return {}

    closed = trades_df[trades_df["exit_type"] != "end_of_data"].copy()
    if closed.empty:
        closed = trades_df.copy()

    n         = len(closed)
    wins      = closed[closed["exit_type"] == "target"]
    stops     = closed[closed["exit_type"].isin(["stop", "stop_gap"])]
    stop_gaps = closed[closed["exit_type"] == "stop_gap"]
    timeouts  = closed[closed["exit_type"] == "time"]
    win_rate  = len(wins) / n * 100 if n else 0

    gross_win  = wins["pnl_aud"].sum()
    gross_loss = stops["pnl_aud"].sum()
    pf         = gross_win / abs(gross_loss) if gross_loss < 0 else np.nan

    avg_r       = closed["pnl_r"].mean()
    expectancy  = closed["pnl_aud"].mean()
    avg_hold    = closed["holding_days"].mean()
    total_pnl   = trades_df["pnl_aud"].sum()

    final_account = (trades_df["account_after"].iloc[-1]
                     if "account_after" in trades_df.columns
                     else STARTING_BALANCE + total_pnl)
    pct_return = (final_account / STARTING_BALANCE - 1) * 100
    cagr_years = 8.0    # 2017-2026
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

    # Calmar ratio: CAGR% / |max_dd_pct|
    calmar = (cagr / abs(max_dd_pct)) if max_dd_pct < 0 else np.nan

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
        grp = closed2[closed2["year"] == yr]
        n_y  = len(grp[grp["exit_type"] != "end_of_data"])
        w_y  = (grp["exit_type"] == "target").sum()
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

    # Regime entry breakdown — how many trades entered in each regime
    REGIME_ORDER = ["BULL", "WEAK_BULL", "CHOPPY", "BEAR", "HIGH_VOL"]
    entry_trades = trades_df[trades_df["exit_type"] != "end_of_data"]
    regime_breakdown: dict[str, int] = {}
    if "regime" in entry_trades.columns:
        vc = entry_trades["regime"].value_counts()
        for r in REGIME_ORDER:
            regime_breakdown[r] = int(vc.get(r, 0))

    # Avg PSM across entries
    avg_entry_psm = (float(entry_trades["psm"].mean())
                     if "psm" in entry_trades.columns and not entry_trades.empty
                     else 1.0)

    return {
        "total_trades":      n,
        "win_rate_pct":      round(win_rate, 1),
        "wins":              len(wins),
        "losses":            len(stops),
        "stop_gaps":         len(stop_gaps),
        "time_stops":        len(timeouts),
        "profit_factor":     round(pf, 2) if not np.isnan(pf) else "n/a",
        "avg_r":             round(avg_r, 3),
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
        "best_trade_aud":    round(closed["pnl_aud"].max(), 2),
        "worst_trade_aud":   round(closed["pnl_aud"].min(), 2),
        "avg_entry_psm":     round(avg_entry_psm, 3),
        "regime_breakdown":  regime_breakdown,
        "yearly":            yearly,
    }


# ---------------------------------------------------------------------------
# Comparison table
# ---------------------------------------------------------------------------

def print_comparison(results: list[tuple[str, dict]]) -> None:
    KEY_METRICS = [
        ("total_trades",      "Trades"),
        ("win_rate_pct",      "Win rate %"),
        ("wins",              "  Targets"),
        ("losses",            "  Stops"),
        ("stop_gaps",         "  Gap stops"),
        ("time_stops",        "  Time stops"),
        ("profit_factor",     "Profit factor"),
        ("avg_r",             "Avg R / trade"),
        ("expectancy_aud",    "Expectancy $"),
        ("total_pnl_aud",     "Total P&L $"),
        ("final_account",     "Final account $"),
        ("pct_return",        "Total return %"),
        ("cagr_pct",          "CAGR %"),
        ("max_drawdown_aud",  "Max drawdown $"),
        ("max_drawdown_pct",  "Max drawdown %"),
        ("calmar",            "Calmar ratio"),
        ("max_consec_losses", "Max consec. losses"),
        ("sharpe",            "Sharpe"),
        ("avg_hold_days",     "Avg hold days"),
    ]
    LOWER_IS_BETTER = {"losses", "stop_gaps", "time_stops", "max_drawdown_aud",
                       "max_drawdown_pct", "max_consec_losses"}

    labels = [lbl for lbl, _ in results]
    col_w  = max(28, max(len(l) for l in labels) + 4)
    met_w  = 22
    sep    = "=" * (met_w + col_w * len(labels) + 4)

    print(f"\n{sep}")
    print(f"  8-YEAR PARAMETER COMPARISON  ({date.today()})")
    print(f"  ${STARTING_BALANCE:,} start | {RISK_PCT*100:.1f}% risk/trade | max {MAX_POSITIONS} positions")
    print(f"  All scenarios: RS top 20% + >=2 triggers | flat sizing | IBKR $6/leg | 0.2% slip | gap risk")
    print(sep)

    hdr = f"  {'Metric':<{met_w}}"
    for lbl in labels:
        hdr += f"  {lbl:^{col_w - 2}}"
    print(hdr)
    print("-" * (met_w + col_w * len(labels) + 4))

    for key, display in KEY_METRICS:
        vals = [m.get(key, None) for _, m in results]

        def fmt(v, k=key):
            if v is None: return "—"
            if k in ("expectancy_aud", "total_pnl_aud", "final_account", "max_drawdown_aud"):
                return f"${v:,.0f}"
            if k in ("pct_return", "cagr_pct"):
                return f"{v:+.1f}%"
            if k == "max_drawdown_pct":
                return f"{v:.1f}%"
            if k == "win_rate_pct":
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

    # ── Regime entry breakdown ────────────────────────────────────────────────
    REGIME_ORDER = ["BULL", "WEAK_BULL", "CHOPPY", "BEAR", "HIGH_VOL"]
    PSM_MAP      = {"BULL": 1.0, "WEAK_BULL": 0.75, "CHOPPY": 0.60,
                    "BEAR": 0.50, "HIGH_VOL": 0.25}

    print(f"  REGIME ENTRY BREAKDOWN  (where were the {results[0][1].get('total_trades','?')} trades taken?)")
    print("-" * (met_w + col_w * len(results) + 4))
    hdr2 = f"  {'Regime (PSM)':<{met_w}}"
    for lbl in labels:
        hdr2 += f"  {lbl:^{col_w - 2}}"
    print(hdr2)
    print("-" * (met_w + col_w * len(results) + 4))

    for regime in REGIME_ORDER:
        psm_label = f"{regime} ({PSM_MAP[regime]:.2f}x)"
        row = f"  {psm_label:<{met_w}}"
        for _, m in results:
            bd = m.get("regime_breakdown", {})
            count = bd.get(regime, 0)
            total = m.get("total_trades", 1)
            pct   = count / total * 100 if total else 0
            cell  = f"{count} ({pct:.0f}%)" if count else "—"
            row  += f"  {cell:^{col_w - 2}}"
        print(row)

    # Avg PSM row
    row = f"  {'Avg entry PSM':<{met_w}}"
    for _, m in results:
        row += f"  {m.get('avg_entry_psm', 1.0):^{col_w - 2}.3f}"
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
        yr_vals = []
        cells   = []
        for _, m in results:
            y = m.get("yearly", {}).get(yr)
            if y:
                sign = "+" if y["pnl_aud"] >= 0 else ""
                cell = f"{sign}${y['pnl_aud']:,.0f} ({y['return_pct']:+.1f}%)"
                yr_vals.append(y["pnl_aud"])
            else:
                cell = "—"
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
    print("  * = best year for that scenario\n")


# ---------------------------------------------------------------------------
# Chart
# ---------------------------------------------------------------------------

def save_comparison_chart(results: list[tuple[str, dict]],
                          equity_dfs: list[pd.DataFrame],
                          out_path: Path) -> None:
    BG, PANEL = "#0d1117", "#161b22"
    PALETTE   = ["#58a6ff", "#26a641", "#e3b341", "#f78166"]
    GREY, RED = "#8b949e", "#da3633"

    n_sc = len(results)
    fig = plt.figure(figsize=(22, 17), facecolor=BG)
    gs  = gridspec.GridSpec(3, n_sc, figure=fig,
                            hspace=0.44, wspace=0.30, height_ratios=[3, 2, 2])
    ax_eq   = fig.add_subplot(gs[0, :])      # full-width equity curve
    yr_axes = [fig.add_subplot(gs[1, i]) for i in range(n_sc)]
    ax_stat = fig.add_subplot(gs[2, :])      # full-width stats

    for ax in [ax_eq] + yr_axes + [ax_stat]:
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
        # Escape $ so matplotlib's mathtext parser treats it as a literal character
        label_safe = label.replace("$", r"\$")
        lbl        = f"{label_safe}  (A\\${final:,.0f} | {ret:+.1f}%)"
        ax_eq.plot(xs, mtm, color=colour, lw=1.6, label=lbl, zorder=3)

    ax_eq.axhline(STARTING_BALANCE, color=GREY, lw=0.8, ls="--", alpha=0.5)
    ax_eq.set_title(
        f"Equity Curves — 8-Year Comparison  |  $20,000 start  |  {date.today()}",
        color="white", fontsize=11, pad=8)
    ax_eq.set_ylabel("AUD", color=GREY, fontsize=9)
    ax_eq.grid(color="#21262d", lw=0.5, alpha=0.6)
    ax_eq.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"${v:,.0f}"))
    ax_eq.legend(fontsize=8, facecolor=PANEL, edgecolor=GREY, labelcolor="white",
                 loc="upper left")

    if equity_dfs and not equity_dfs[0].empty:
        eq_dates = pd.to_datetime(equity_dfs[0]["date"])
        step = max(1, len(eq_dates) // 10)
        tpos = list(range(0, len(eq_dates), step))
        tlbl = [pd.Timestamp(eq_dates.iloc[i]).strftime("%b %y") for i in tpos if i < len(eq_dates)]
        ax_eq.set_xticks(tpos[:len(tlbl)])
        ax_eq.set_xticklabels(tlbl, color=GREY, fontsize=8)

    # ── Year-by-year bars per scenario ────────────────────────────────────────
    all_years = sorted(set(yr for _, m in results for yr in m.get("yearly", {})))

    for ax, (label, m), colour in zip(yr_axes, results, PALETTE):
        yearly = m.get("yearly", {})
        years  = [yr for yr in all_years if yr in yearly]
        pnls   = [yearly[yr]["pnl_aud"] for yr in years]
        cols   = ["#26a641" if p >= 0 else RED for p in pnls]

        ax.bar(years, pnls, color=cols, alpha=0.85, width=0.6)
        ax.axhline(0, color=GREY, lw=0.8, ls="--")
        ax.set_title(f"Year-by-Year  |  {label}", color="white", fontsize=9, pad=5)
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
                    color=GREY, fontsize=6)

    # ── Stats table ───────────────────────────────────────────────────────────
    ax_stat.axis("off")

    stat_rows = [
        ("Final account",    "final_account",     lambda v: f"${v:,.0f}"),
        ("Total return",     "pct_return",         lambda v: f"{v:+.1f}%"),
        ("CAGR",             "cagr_pct",           lambda v: f"{v:+.1f}%"),
        ("Trades",           "total_trades",       str),
        ("Win rate",         "win_rate_pct",        lambda v: f"{v:.1f}%"),
        ("Profit factor",    "profit_factor",      str),
        ("Avg R",            "avg_r",              str),
        ("Expectancy",       "expectancy_aud",      lambda v: f"${v:,.0f}"),
        ("Max DD $",         "max_drawdown_aud",   lambda v: f"${v:,.0f}"),
        ("Max DD %",         "max_drawdown_pct",   lambda v: f"{v:.1f}%"),
        ("Calmar",           "calmar",             str),
        ("Sharpe",           "sharpe",             str),
        ("Consec losses",    "max_consec_losses",  str),
    ]

    n_col = len(results)
    col_positions = np.linspace(0.01, 0.99, n_col + 1)[:-1]
    label_x = 0.0
    val_xs   = col_positions + 0.01

    y = 0.95
    dy = 0.073

    # Header
    ax_stat.text(label_x, y, "Metric", transform=ax_stat.transAxes,
                 color=GREY, fontsize=8.5, va="top", fontweight="bold")
    for i, (lbl, _) in enumerate(results):
        ax_stat.text(val_xs[i], y, lbl, transform=ax_stat.transAxes,
                     color=PALETTE[i], fontsize=8.5, va="top", fontweight="bold")
    y -= dy * 0.7
    # Draw separator line using axes-fraction coordinates via axline
    ax_stat.plot([0, 1], [y, y], color="#21262d", lw=0.5, alpha=0.5,
                 transform=ax_stat.transAxes, clip_on=False)
    y -= dy * 0.3

    for row_label, key, fmtfn in stat_rows:
        ax_stat.text(label_x, y, row_label, transform=ax_stat.transAxes,
                     color=GREY, fontsize=8, va="top")
        for i, (_, m) in enumerate(results):
            v = m.get(key)
            txt = fmtfn(v) if v is not None else "—"
            ax_stat.text(val_xs[i], y, txt, transform=ax_stat.transAxes,
                         color="white", fontsize=8, va="top", fontweight="bold")
        y -= dy

    ax_stat.set_title("Summary Statistics", color="white", fontsize=10, pad=6)

    fig.text(0.5, 0.002,
             f"Stop: {ATR_STOP_MULT}x ATR  |  Target: {TARGET_R}:1  |  BE@1R  |  "
             f"Time: {TIME_STOP_DAYS}d  |  "
             f"Realistic: IBKR $6/leg · 0.2% slippage · gap risk (open vs stop)",
             color=GREY, fontsize=7.5, ha="center")

    plt.savefig(out_path, dpi=130, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"  Chart -> {out_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="ASX Swing Engine — full 8-year validation backtest")
    p.add_argument("--period", default="9y",
                   help="yfinance period (default: 9y = 2017-2026)")
    p.add_argument("--no-sectors", action="store_true",
                   help="Skip sector fetch (treat all as Unknown, no sector cap)")
    return p.parse_args()


def main():
    args = parse_args()

    # Force UTF-8 stdout on Windows so Unicode chars don't crash
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    print("=" * 70)
    print("  ASX SWING ENGINE - DEFINITIVE 8-YEAR VALIDATION BACKTEST")
    print(f"  {date.today()}  |  period={args.period}  |  $20,000 start")
    print("=" * 70)
    print()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    today_str = date.today().strftime("%Y%m%d")

    # ── 1. Download XJO (full OHLCV for regime ATR) ───────────────────────────
    print("Step 1/6  Downloading benchmark (XJO) ...")
    with _quiet():
        xjo_raw = yf.download(BENCHMARK, period=args.period, interval="1d",
                              auto_adjust=True, progress=False)
    if isinstance(xjo_raw.columns, pd.MultiIndex):
        xjo_raw.columns = [c[0] for c in xjo_raw.columns]
    xjo_df    = xjo_raw[["Open","High","Low","Close","Volume"]].dropna()
    xjo_close = xjo_df["Close"]
    print(f"  XJO: {len(xjo_close)} bars  "
          f"({xjo_close.index[0].date()} to {xjo_close.index[-1].date()})\n")

    # ── 2. Build historical regime series ─────────────────────────────────────
    print("Step 2/6  Building historical regime series ...")
    regime_series = build_regime_series(xjo_df, args.period)
    print()

    # ── 3. Download universe price data ───────────────────────────────────────
    tickers = list(dict.fromkeys(ASX_PRESET))   # deduplicate, preserve order
    print(f"Step 3/6  Downloading {len(tickers)}-ticker universe ({args.period}) ...")
    price_data = download_universe(tickers, args.period)
    print(f"  Data: {len(price_data)}/{len(tickers)} tickers.\n")

    if not price_data:
        print("ERROR: No price data downloaded. Check network.")
        sys.exit(1)

    # ── 4. Build signal pools (one per scenario — they differ in ATR threshold
    #        and entry timing, so they cannot share a pool) ─────────────────────
    all_results: list[tuple[str, dict]] = []
    all_equity:  list[pd.DataFrame] = []

    import re as _re

    # Both scenarios use the same realistic cost model
    realistic = dict(comm_entry=6.0, comm_exit=6.0, slippage=0.002, gap_risk=True)

    # Scenario A — Realistic baseline (from previous run)
    #   ATR >= 3%  |  entry at next-day open
    # Scenario B — Both improvements combined
    #   ATR >= 4%  |  entry at signal-day close (3:30pm proxy)
    pool_configs = [
        (
            "Realistic baseline  (ATR 3%, next-day open)",
            dict(min_atr_pct=3.0, entry_at_close=False),
        ),
        (
            "Both improvements  (ATR 4%, signal-day close)",
            dict(min_atr_pct=4.0, entry_at_close=True),
        ),
    ]

    for sc_label, pool_params in pool_configs:
        print(f"Step 4/6  Building signal pool: {sc_label} ...")
        ind, by_entry = _build_signal_pool(
            price_data, xjo_close, min_triggers=2, **pool_params
        )
        print()

        print(f"Running simulation: {sc_label} ...")
        trades_df, equity_df = _simulate(ind, by_entry, psm_series=None, **realistic)

        if trades_df.empty:
            print("  No trades generated!")
            continue

        m = calc_metrics(trades_df, equity_df)
        all_results.append((sc_label, m))
        all_equity.append(equity_df)

        print(f"  {m['total_trades']} trades  |  "
              f"win {m['win_rate_pct']}%  |  "
              f"PF {m['profit_factor']}  |  "
              f"return {m.get('pct_return','n/a')}%  |  "
              f"CAGR {m.get('cagr_pct','n/a')}%  |  "
              f"gap stops {m.get('stop_gaps', 0)}")

        slug = _re.sub(r"[^a-z0-9]+", "_", sc_label.lower()).strip("_")[:30].rstrip("_")
        trades_df.to_csv(RESULTS_DIR / f"trades_8y_{slug}_{today_str}.csv", index=False)
        equity_df.to_csv(RESULTS_DIR / f"equity_8y_{slug}_{today_str}.csv", index=False)
        print()

    # ── Output ────────────────────────────────────────────────────────────────
    if all_results:
        print_comparison(all_results)
        save_comparison_chart(
            all_results, all_equity,
            RESULTS_DIR / f"backtest_8y_full_{today_str}.png"
        )
        print(f"Results saved to {RESULTS_DIR}/")
    else:
        print("No results to compare.")


if __name__ == "__main__":
    main()
