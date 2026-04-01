"""
ASX Swing Trade Universe Screener
----------------------------------
Dynamic universe pulled from the official ASX listed-companies CSV.

Hard filters (pass/fail):
  - Price > $0.50
  - 20-day avg volume > 200K
  - Market cap > $100M
  - Price above 200 EMA  (long-term uptrend)
  - Price above 50 EMA   (medium-term uptrend)
  - RSI(14) in 40–65     (pullback zone — not chasing overbought)
  - Price within 10% above 50 EMA (not too extended)

Composite score (0–100), ranked descending:
  35%  Relative Strength vs XJO (63-day excess return)
  25%  Rate of Change 20-day (momentum)
  20%  ATR%(14) sweet-spot score (1.5–4% = full marks, linear ramp either side)
  20%  Volume ratio (5-day avg / 20-day avg)
"""

import io
import os
import sys
import time
import warnings
import contextlib

import numpy as np
import pandas as pd
import requests
import yfinance as yf

warnings.filterwarnings("ignore")


@contextlib.contextmanager
def _suppress_yf_noise():
    """Suppress yfinance's delisted/failed-download stderr chatter."""
    with open(os.devnull, "w") as devnull:
        old_stderr = sys.stderr
        sys.stderr = devnull
        try:
            yield
        finally:
            sys.stderr = old_stderr

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
BENCHMARK          = "^AXJO"
HISTORY_PERIOD     = "1y"         # ~252 trading days — enough for 200 EMA

MIN_PRICE          = 0.50
MAX_PRICE          = 50.0         # mega-caps swing trade poorly (backtest confirmed)
MIN_AVG_VOLUME     = 200_000
MIN_MARKET_CAP     = 100_000_000  # AUD

EMA_SLOW           = 200
EMA_FAST           = 50
RSI_PERIOD         = 14
RSI_MIN            = 40
RSI_MAX            = 65
MAX_EMA50_DIST_PCT = 10.0         # max % a stock can be above its 50 EMA

ATR_PERIOD         = 14
MIN_ATR_PCT        = 3.0          # hard floor — backtest shows edge only above 3%
ATR_SWEET_MIN      = 3.0          # align scoring sweet spot with hard floor
ATR_SWEET_MAX      = 4.0          # above: too volatile
ATR_SCORE_ZERO     = 8.0          # ATR% at which score hits 0 on the high side

RS_PERIOD          = 63           # ~3 months
MOMENTUM_PERIOD    = 20
VOL_SHORT          = 5
VOL_LONG           = 20

WEIGHTS  = {"rs": 0.35, "momentum": 0.25, "atr_sweet": 0.20, "vol_ratio": 0.20}

BATCH_SIZE   = 20
BATCH_DELAY  = 1.0   # seconds between download batches

ASX_CSV_URL  = "https://www.asx.com.au/asx/research/ASXListedCompanies.csv"


# ---------------------------------------------------------------------------
# Universe
# ---------------------------------------------------------------------------

def fetch_asx_universe() -> list[str]:
    """Download ASX listed companies CSV and return ticker list (e.g. 'BHP.AX')."""
    print("Fetching ASX universe …")
    resp = requests.get(ASX_CSV_URL, timeout=30)
    resp.raise_for_status()

    # Row 0 = title ("ASX listed companies as at …"); row 1 = column headers
    df = pd.read_csv(io.StringIO(resp.text), skiprows=1)
    df.columns = df.columns.str.strip()

    codes = df["ASX code"].dropna().str.strip()
    # Keep standard equity codes: 2–5 uppercase alphanumeric characters
    codes = codes[codes.str.match(r"^[A-Z0-9]{2,5}$")]

    tickers = (codes + ".AX").tolist()
    print(f"  {len(tickers)} tickers in ASX universe.")
    return tickers


# ---------------------------------------------------------------------------
# Technical indicators
# ---------------------------------------------------------------------------

def calc_ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def calc_rsi(close: pd.Series, period: int = 14) -> float:
    delta    = close.diff().dropna()
    gain     = delta.clip(lower=0)
    loss     = (-delta).clip(lower=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()
    rs       = avg_gain / avg_loss.replace(0, np.nan)
    rsi      = 100 - (100 / (1 + rs))
    return float(rsi.iloc[-1]) if len(rsi) else np.nan


def calc_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> float:
    prev_close = close.shift(1)
    tr = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    atr = tr.ewm(span=period, adjust=False).mean()
    return float(atr.iloc[-1]) if len(atr) else np.nan


def atr_sweet_score(atr_pct: float) -> float:
    """
    Trapezoidal score (0–100):
      0 → ATR_SWEET_MIN  : linear ramp  0 → 100
      ATR_SWEET_MIN → ATR_SWEET_MAX : 100
      ATR_SWEET_MAX → ATR_SCORE_ZERO: linear ramp 100 → 0
      > ATR_SCORE_ZERO   : 0
    """
    if np.isnan(atr_pct):
        return 0.0
    if atr_pct < ATR_SWEET_MIN:
        return max(0.0, atr_pct / ATR_SWEET_MIN * 100)
    if atr_pct <= ATR_SWEET_MAX:
        return 100.0
    if atr_pct < ATR_SCORE_ZERO:
        return max(0.0, (1 - (atr_pct - ATR_SWEET_MAX) / (ATR_SCORE_ZERO - ATR_SWEET_MAX)) * 100)
    return 0.0


def percentile_rank(series: pd.Series) -> pd.Series:
    """Rank each value 0–100 within the series (higher = better)."""
    return series.rank(pct=True, na_option="bottom") * 100


# ---------------------------------------------------------------------------
# Data fetching
# ---------------------------------------------------------------------------

def fetch_batch(tickers: list[str], period: str) -> dict[str, pd.DataFrame]:
    """Download OHLCV for a batch of tickers; return {ticker: DataFrame}."""
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
        # yfinance 1.x: columns are (field, ticker) when group_by is not set
        # level=1 selects by ticker name across all fields
        available = raw.columns.get_level_values(1).unique()
        for ticker in tickers:
            if ticker in available:
                df = raw.xs(ticker, axis=1, level=1).dropna(how="all")
                if not df.empty:
                    result[ticker] = df
    else:
        # Flat columns — single ticker
        if len(tickers) == 1:
            result[tickers[0]] = raw.dropna(how="all")

    return result


def fetch_market_cap(ticker: str) -> float | None:
    try:
        fi = yf.Ticker(ticker).fast_info
        mcap = getattr(fi, "market_cap", None)
        if mcap is None:
            # Fallback for older yfinance versions
            info = yf.Ticker(ticker).info
            mcap = info.get("marketCap")
        return mcap
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Screener
# ---------------------------------------------------------------------------

def run_screener() -> pd.DataFrame:

    # 1 — Build universe
    tickers = fetch_asx_universe()

    # 2 — Benchmark (XJO) — one download, used for all RS calcs
    print(f"Downloading benchmark ({BENCHMARK}) …")
    with _suppress_yf_noise():
        xjo_raw = yf.download(BENCHMARK, period=HISTORY_PERIOD, interval="1d",
                              auto_adjust=True, progress=False)
    _xjo_close = xjo_raw["Close"]
    xjo_close = (_xjo_close.iloc[:, 0] if isinstance(_xjo_close, pd.DataFrame) else _xjo_close).dropna()
    if len(xjo_close) < RS_PERIOD + 1:
        raise RuntimeError("Not enough benchmark history. Check connection / yfinance.")
    xjo_63d_return = float(xjo_close.iloc[-1] / xjo_close.iloc[-RS_PERIOD - 1] - 1) * 100

    # 3 — Download OHLCV in batches
    print(f"Downloading OHLCV for {len(tickers)} tickers …")
    price_data: dict[str, pd.DataFrame] = {}
    batches = [tickers[i: i + BATCH_SIZE] for i in range(0, len(tickers), BATCH_SIZE)]

    for idx, batch in enumerate(batches, 1):
        if idx % 10 == 0 or idx == len(batches):
            print(f"  Batch {idx}/{len(batches)} …")
        price_data.update(fetch_batch(batch, HISTORY_PERIOD))
        if idx < len(batches):
            time.sleep(BATCH_DELAY)

    print(f"  Received price data for {len(price_data)} tickers.")

    # 4 — Technical filters + raw metric calculation
    candidates: list[dict] = []

    for ticker, df in price_data.items():
        close  = df["Close"].dropna()
        high   = df["High"].dropna()
        low    = df["Low"].dropna()
        volume = df["Volume"].dropna()

        # Need at least EMA_SLOW bars to be meaningful
        if len(close) < EMA_SLOW + 10:
            continue

        last_price   = float(close.iloc[-1])
        avg_vol_20   = float(volume.tail(VOL_LONG).mean())

        # ---- Hard filters (cheap — no extra API calls) ----
        if last_price < MIN_PRICE:
            continue
        if avg_vol_20 < MIN_AVG_VOLUME:
            continue

        ema200 = float(calc_ema(close, EMA_SLOW).iloc[-1])
        if last_price < ema200:
            continue

        ema50 = float(calc_ema(close, EMA_FAST).iloc[-1])
        if last_price < ema50:
            continue

        ema50_dist = (last_price / ema50 - 1) * 100
        if ema50_dist > MAX_EMA50_DIST_PCT:
            continue

        rsi = calc_rsi(close, RSI_PERIOD)
        if np.isnan(rsi) or not (RSI_MIN <= rsi <= RSI_MAX):
            continue

        # ---- New hard filters (data-driven from backtest) ----
        if last_price > MAX_PRICE:
            continue

        atr     = calc_atr(high, low, close, ATR_PERIOD)
        atr_pct = round(atr / last_price * 100, 2) if not np.isnan(atr) else np.nan
        if np.isnan(atr_pct) or atr_pct < MIN_ATR_PCT:
            continue

        rs_vs_xjo = np.nan
        if len(close) >= RS_PERIOD + 1:
            stock_63d = (close.iloc[-1] / close.iloc[-RS_PERIOD - 1] - 1) * 100
            rs_vs_xjo = round(float(stock_63d) - xjo_63d_return, 2)
        if np.isnan(rs_vs_xjo) or rs_vs_xjo <= 0:
            continue

        # ---- Raw scoring metrics ----
        momentum = np.nan
        if len(close) >= MOMENTUM_PERIOD + 1:
            momentum = round(float((close.iloc[-1] / close.iloc[-MOMENTUM_PERIOD - 1] - 1) * 100), 2)

        avg_vol_5  = float(volume.tail(VOL_SHORT).mean())
        vol_ratio  = round(avg_vol_5 / avg_vol_20, 2) if avg_vol_20 > 0 else np.nan

        candidates.append({
            "ticker":        ticker,
            "last_price":    round(last_price, 3),
            "avg_vol_20d":   int(avg_vol_20),
            "ema_50":        round(ema50, 3),
            "ema_200":       round(ema200, 3),
            "ema50_dist_pct":round(ema50_dist, 1),
            "rsi_14":        round(rsi, 1),
            "atr_pct":       atr_pct,
            "rs_vs_xjo":     rs_vs_xjo,
            "momentum_20d":  momentum,
            "vol_ratio":     vol_ratio,
        })

    print(f"  {len(candidates)} passed technical filters. Fetching market caps …")

    # 5 — Market cap filter (slow: one API call per candidate)
    passing: list[dict] = []
    for row in candidates:
        mcap = fetch_market_cap(row["ticker"])
        if mcap is None or mcap < MIN_MARKET_CAP:
            continue
        row["market_cap_m"] = round(mcap / 1_000_000, 1)
        passing.append(row)

    print(f"  {len(passing)} passed market cap filter.")

    if not passing:
        print("No stocks passed all filters.")
        return pd.DataFrame()

    # 6 — Composite score
    df = pd.DataFrame(passing)

    df["rs_score"]        = percentile_rank(df["rs_vs_xjo"])
    df["momentum_score"]  = percentile_rank(df["momentum_20d"])
    df["atr_sweet_score"] = df["atr_pct"].apply(atr_sweet_score)
    df["vol_ratio_score"] = percentile_rank(df["vol_ratio"])

    df["composite_score"] = (
        df["rs_score"]        * WEIGHTS["rs"]        +
        df["momentum_score"]  * WEIGHTS["momentum"]  +
        df["atr_sweet_score"] * WEIGHTS["atr_sweet"] +
        df["vol_ratio_score"] * WEIGHTS["vol_ratio"]
    ).round(1)

    # 7 — Final sort and column order
    df.sort_values("composite_score", ascending=False, inplace=True)
    df.reset_index(drop=True, inplace=True)
    df.index += 1
    df.index.name = "rank"

    return df[[
        "ticker", "last_price", "market_cap_m", "composite_score",
        "rs_vs_xjo", "momentum_20d", "atr_pct", "vol_ratio",
        "rsi_14", "ema50_dist_pct", "avg_vol_20d",
        "rs_score", "momentum_score", "atr_sweet_score", "vol_ratio_score",
    ]]


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    pd.set_option("display.max_rows",    None)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width",       180)
    pd.set_option("display.float_format", "{:.2f}".format)

    results = run_screener()

    if results.empty:
        print("No results.")
    else:
        print(f"\n=== ASX Swing Universe — {len(results)} stocks | ranked by composite score ===")
        print(results.to_string())
        results.to_csv("results/screener_output.csv")
        print("\nSaved → results/screener_output.csv")
