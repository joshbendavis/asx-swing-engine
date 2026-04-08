"""
signals.py
----------
Entry trigger scanner for ASX swing trades.

Reads today's screener output (results/screener_output.csv) and identifies
which candidates have a valid entry trigger based on at least 2 of 3
intraday/end-of-day conditions:

  1. RSI bounce   — yesterday's RSI was in 40–55; today's RSI > yesterday's
  2. MACD cross   — MACD(12,26,9) line crosses above signal line today
  3. Vol breakout — today's volume >= 1.5× the 20-day average volume

Trade levels (based on today's close):
  Entry      = today's close
  Stop loss  = entry - 2 × ATR(14)
  Target     = entry + 2 × initial_risk  (2:1 R:R)
  Risk ($)   = 2 × ATR(14)

Output ranked by screener composite score → results/signals_output.csv
"""

import json
import os
import sys
import time
import warnings
import contextlib
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
SCREENER_CSV    = Path("results/screener_output.csv")
OUTPUT_CSV      = Path("results/signals_output.csv")
REGIME_JSON     = Path("results/regime.json")

HISTORY_PERIOD  = "6mo"        # enough for MACD warm-up (26 + 9 + buffer)
BATCH_SIZE      = 20
BATCH_DELAY     = 1.0

# Trigger thresholds
RSI_BOUNCE_LO   = 40.0         # yesterday's RSI lower bound for bounce zone
RSI_BOUNCE_HI   = 55.0         # yesterday's RSI upper bound for bounce zone
VOL_MULT        = 1.5          # today's volume must be >= this × 20d avg
VOL_LONG        = 20           # rolling window for average volume

# Exit levels
ATR_STOP_MULT   = 2.0          # stop = entry - ATR_STOP_MULT × ATR
TARGET_R        = 2.0          # target = entry + TARGET_R × initial_risk

# MACD parameters
MACD_FAST       = 12
MACD_SLOW       = 26
MACD_SIGNAL     = 9

# ATR
ATR_PERIOD      = 14

# Minimum triggers required
MIN_TRIGGERS    = 2


# ---------------------------------------------------------------------------
# Helpers — suppress yfinance noise
# ---------------------------------------------------------------------------

@contextlib.contextmanager
def _suppress_yf_noise():
    with open(os.devnull, "w") as devnull:
        old = sys.stderr
        sys.stderr = devnull
        try:
            yield
        finally:
            sys.stderr = old


# ---------------------------------------------------------------------------
# Regime helper
# ---------------------------------------------------------------------------

def _load_regime() -> dict:
    """
    Load results/regime.json and return the parsed dict.
    Returns a safe BULL default if the file is missing or unreadable.
    """
    default = {"regime": "BULL", "position_size_multiplier": 1.0}
    if not REGIME_JSON.exists():
        return default
    try:
        with open(REGIME_JSON, "r") as f:
            data = json.load(f)
        return data
    except Exception:
        return default


# ---------------------------------------------------------------------------
# Technical indicators
# ---------------------------------------------------------------------------

def calc_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Return full RSI series (same length as close)."""
    delta    = close.diff()
    gain     = delta.clip(lower=0)
    loss     = (-delta).clip(lower=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()
    rs       = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def calc_macd(close: pd.Series,
              fast: int = 12, slow: int = 26, signal: int = 9
              ) -> tuple[pd.Series, pd.Series]:
    """Return (macd_line, signal_line) series."""
    ema_fast   = close.ewm(span=fast,   adjust=False).mean()
    ema_slow   = close.ewm(span=slow,   adjust=False).mean()
    macd_line  = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    return macd_line, signal_line


def calc_atr(high: pd.Series, low: pd.Series,
             close: pd.Series, period: int = 14) -> pd.Series:
    """Return full ATR series."""
    prev_close = close.shift(1)
    tr = pd.concat(
        [high - low,
         (high - prev_close).abs(),
         (low  - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    return tr.ewm(span=period, adjust=False).mean()


# ---------------------------------------------------------------------------
# Batch price download (mirrors screener pattern)
# ---------------------------------------------------------------------------

def fetch_batch(tickers: list[str], period: str) -> dict[str, pd.DataFrame]:
    """Download OHLCV for a batch; return {ticker: DataFrame}."""
    with _suppress_yf_noise():
        raw = yf.download(
            tickers,
            period=period,
            interval="1d",
            auto_adjust=True,
            progress=False,
            threads=True,
        )

    result: dict[str, pd.DataFrame] = {}
    if raw.empty:
        return result

    if isinstance(raw.columns, pd.MultiIndex):
        available = raw.columns.get_level_values(1).unique()
        for ticker in tickers:
            if ticker in available:
                df = raw.xs(ticker, axis=1, level=1).dropna(how="all")
                if not df.empty:
                    result[ticker] = df
    else:
        if len(tickers) == 1:
            result[tickers[0]] = raw.dropna(how="all")

    return result


# ---------------------------------------------------------------------------
# Signal evaluation per ticker
# ---------------------------------------------------------------------------

def evaluate_signals(ticker: str, df: pd.DataFrame,
                     min_triggers: int = MIN_TRIGGERS) -> dict | None:
    """
    Evaluate the 3 entry triggers for a single ticker.
    Returns a dict of signal details or None if fewer than min_triggers fire.
    Requires at least 2 rows of data (today + yesterday).
    """
    if len(df) < max(MACD_SLOW + MACD_SIGNAL + 5, VOL_LONG + 2):
        return None

    close  = df["Close"].dropna()
    high   = df["High"].dropna()
    low    = df["Low"].dropna()
    volume = df["Volume"].dropna()

    if len(close) < 2:
        return None

    # ---- Indicators ----
    rsi              = calc_rsi(close, ATR_PERIOD)
    macd_line, sig_l = calc_macd(close, MACD_FAST, MACD_SLOW, MACD_SIGNAL)
    atr_series       = calc_atr(high, low, close, ATR_PERIOD)

    today_rsi   = float(rsi.iloc[-1])
    yest_rsi    = float(rsi.iloc[-2])

    today_macd  = float(macd_line.iloc[-1])
    yest_macd   = float(macd_line.iloc[-2])
    today_sig   = float(sig_l.iloc[-1])
    yest_sig    = float(sig_l.iloc[-2])

    today_vol   = float(volume.iloc[-1])
    avg_vol_20  = float(volume.tail(VOL_LONG + 1).iloc[:-1].mean())  # exclude today

    last_price  = float(close.iloc[-1])
    atr         = float(atr_series.iloc[-1])

    # ---- Trigger checks ----
    t1_rsi_bounce = (
        RSI_BOUNCE_LO <= yest_rsi <= RSI_BOUNCE_HI
        and today_rsi > yest_rsi
    )
    t2_macd_cross = (
        yest_macd < yest_sig    # was below signal yesterday
        and today_macd > today_sig  # crossed above today
    )
    t3_vol_break  = (
        avg_vol_20 > 0
        and today_vol >= VOL_MULT * avg_vol_20
    )

    triggers      = [t1_rsi_bounce, t2_macd_cross, t3_vol_break]
    trigger_count = sum(triggers)

    if trigger_count < min_triggers:
        return None

    # ---- Trade levels ----
    initial_risk = ATR_STOP_MULT * atr
    entry        = last_price
    stop_loss    = round(entry - initial_risk, 3)
    target       = round(entry + TARGET_R * initial_risk, 3)
    risk_pct     = round(initial_risk / entry * 100, 2)

    return {
        "ticker":          ticker,
        "entry":           round(entry, 3),
        "stop_loss":       stop_loss,
        "target":          target,
        "risk_pct":        risk_pct,
        "atr":             round(atr, 4),
        "trigger_count":   trigger_count,
        "t_rsi_bounce":    t1_rsi_bounce,
        "t_macd_cross":    t2_macd_cross,
        "t_vol_break":     t3_vol_break,
        "rsi_today":       round(today_rsi, 1),
        "rsi_yest":        round(yest_rsi, 1),
        "macd_today":      round(today_macd, 5),
        "signal_today":    round(today_sig, 5),
        "vol_ratio_today": round(today_vol / avg_vol_20, 2) if avg_vol_20 > 0 else np.nan,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_signals() -> pd.DataFrame:

    # 1 — Load screener output
    if not SCREENER_CSV.exists():
        raise FileNotFoundError(
            f"Screener output not found at {SCREENER_CSV}. "
            "Run screener.py first."
        )

    screener = pd.read_csv(SCREENER_CSV, index_col="rank")
    tickers  = screener["ticker"].tolist()
    print(f"Loaded {len(tickers)} candidates from screener output.")

    # Build lookup: ticker -> screener row
    screener_map = screener.set_index("ticker")

    # 1b — Load market regime
    regime_data = _load_regime()
    regime      = regime_data.get("regime", "BULL")
    psm         = regime_data.get("position_size_multiplier", 1.0)
    confidence  = regime_data.get("confidence", None)

    conf_str = f"  confidence={confidence}%" if confidence is not None else ""
    print(f"Market regime: {regime}{conf_str}  |  position_size_multiplier={psm}")

    if regime == "BEAR":
        print("WARNING: BEAR regime detected — skipping signal scan. No trades today.")
        return pd.DataFrame()

    if regime == "CHOPPY":
        min_triggers = 3
        print("CHOPPY regime: raising min_triggers to 3 (all 3 conditions must fire).")
    else:
        min_triggers = MIN_TRIGGERS

    if regime == "HIGH_VOL":
        print(
            "HIGH_VOL regime: position sizes will be halved by ibkr_executor "
            f"(position_size_multiplier={psm} is in regime.json)."
        )

    # 2 — Download recent OHLCV
    print(f"Downloading {HISTORY_PERIOD} price data for {len(tickers)} tickers ...")
    price_data: dict[str, pd.DataFrame] = {}
    batches = [tickers[i: i + BATCH_SIZE] for i in range(0, len(tickers), BATCH_SIZE)]

    for idx, batch in enumerate(batches, 1):
        if idx % 5 == 0 or idx == len(batches):
            print(f"  Batch {idx}/{len(batches)} ...")
        price_data.update(fetch_batch(batch, HISTORY_PERIOD))
        if idx < len(batches):
            time.sleep(BATCH_DELAY)

    print(f"  Price data received for {len(price_data)} tickers.")

    # 3 — Evaluate triggers
    print("Evaluating entry triggers ...")
    signals: list[dict] = []

    for ticker, df in price_data.items():
        result = evaluate_signals(ticker, df, min_triggers=min_triggers)
        if result is None:
            continue

        # Merge screener metrics
        if ticker in screener_map.index:
            row = screener_map.loc[ticker]
            result["composite_score"] = row["composite_score"]
            result["rs_vs_xjo"]       = row["rs_vs_xjo"]
            result["momentum_20d"]    = row["momentum_20d"]
            result["atr_pct"]         = row["atr_pct"]
            result["rsi_14_screener"] = row["rsi_14"]
            result["market_cap_m"]    = row["market_cap_m"]

        signals.append(result)

    if not signals:
        print("No tickers met the entry trigger criteria today.")
        return pd.DataFrame()

    # 4 — Rank: trigger_count desc, then composite_score desc
    df_out = pd.DataFrame(signals)
    df_out.sort_values(
        ["trigger_count", "composite_score"],
        ascending=[False, False],
        inplace=True,
    )
    df_out.reset_index(drop=True, inplace=True)
    df_out.index += 1
    df_out.index.name = "rank"

    # 5 — Column order
    cols = [
        "ticker", "entry", "stop_loss", "target",
        "trigger_count", "t_rsi_bounce", "t_macd_cross", "t_vol_break",
        "composite_score", "risk_pct", "atr_pct",
        "rsi_today", "rsi_yest", "vol_ratio_today",
        "macd_today", "signal_today",
        "rs_vs_xjo", "momentum_20d", "market_cap_m",
    ]
    cols = [c for c in cols if c in df_out.columns]
    return df_out[cols]


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    pd.set_option("display.max_rows",    None)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width",       200)
    pd.set_option("display.float_format", "{:.3f}".format)

    print(f"\n=== ASX Swing Engine - Signal Scanner | {date.today()} ===\n")

    results = run_signals()

    if results.empty:
        print("No signals today.")
    else:
        # Pretty-print trigger columns as checkmarks
        display = results.copy()
        for col in ["t_rsi_bounce", "t_macd_cross", "t_vol_break"]:
            if col in display.columns:
                display[col] = display[col].map({True: "YES", False: "---"})

        print(f"=== {len(results)} signal(s) found ===\n")
        print(display.to_string())

        OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
        results.to_csv(OUTPUT_CSV)
        print(f"\nSaved -> {OUTPUT_CSV}")
