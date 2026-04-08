"""
regime_detector.py
------------------
Daily market regime classification engine for the ASX swing trading system.

Classifies the current market into one of four regimes:
    BULL     - trending up, breadth supportive
    BEAR     - index below 200 EMA
    CHOPPY   - above 200 EMA but flat/weak breadth
    HIGH_VOL - volatility override (ATR% > 2.5)

Writes results/regime.json and returns a dict for use by run_daily.py.

Usage:
    python regime_detector.py          # run standalone, pretty-print result
    from regime_detector import run_regime_detector
    result = run_regime_detector()
"""

import json
import logging
import os
import sys
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")

# ── logger ────────────────────────────────────────────────────────────────────
log = logging.getLogger("regime")
if not log.handlers:
    _handler = logging.StreamHandler(sys.stdout)
    _handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%H:%M:%S"))
    log.addHandler(_handler)
    log.setLevel(logging.INFO)

# ── constants ─────────────────────────────────────────────────────────────────
XJO_TICKER = "^AXJO"
XJO_PERIOD  = "2y"

ASX200_SAMPLE = [
    "BHP.AX", "CBA.AX", "CSL.AX", "NAB.AX", "WBC.AX", "ANZ.AX", "WES.AX", "MQG.AX",
    "RIO.AX", "WOW.AX", "GMG.AX", "TLS.AX", "FMG.AX", "REA.AX", "ALL.AX", "S32.AX",
    "STO.AX", "ORI.AX", "AMC.AX", "TCL.AX", "QBE.AX", "IAG.AX", "SHL.AX", "COL.AX",
    "ASX.AX", "MPL.AX", "APA.AX", "JHX.AX", "SGP.AX", "MIN.AX", "IGO.AX", "PLS.AX",
    "LYC.AX", "WHC.AX", "NXT.AX", "ALX.AX", "ALD.AX", "NCM.AX", "AGL.AX", "SUL.AX",
]

RESULTS_DIR = Path(__file__).parent / "results"

# ── helpers ───────────────────────────────────────────────────────────────────

def _suppress_stderr():
    """Context manager: redirect stderr to devnull (silences yfinance progress bars)."""
    import contextlib
    devnull = open(os.devnull, "w")
    return contextlib.redirect_stderr(devnull)


def _download_xjo() -> pd.DataFrame:
    """Download XJO OHLCV data. Raises RuntimeError on failure."""
    log.info("Downloading %s (%s) ...", XJO_TICKER, XJO_PERIOD)
    with _suppress_stderr():
        df = yf.download(
            XJO_TICKER,
            period=XJO_PERIOD,
            auto_adjust=True,
            progress=False,
            threads=False,
        )

    if df is None or df.empty:
        raise RuntimeError(
            f"Failed to download {XJO_TICKER}: yfinance returned empty data. "
            "Check network connectivity and that the ticker is valid."
        )

    # Flatten multi-level columns if present (yfinance >=0.2.x)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] for col in df.columns]

    required = {"Close", "High", "Low"}
    missing = required - set(df.columns)
    if missing:
        raise RuntimeError(f"XJO data missing expected columns: {missing}. Got: {list(df.columns)}")

    log.info("XJO: %d rows downloaded (latest: %s)", len(df), df.index[-1].date())
    return df


def _calc_atr(df: pd.DataFrame, span: int = 14) -> pd.Series:
    """
    ATR via EWM.
    True range = max(High-Low, |High-PrevClose|, |Low-PrevClose|)
    """
    high  = df["High"]
    low   = df["Low"]
    close = df["Close"]

    prev_close = close.shift(1)
    tr = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low  - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    atr = tr.ewm(span=span, adjust=False).mean()
    return atr


def _calc_xjo_indicators(df: pd.DataFrame) -> dict:
    """
    Compute all XJO-based indicators. Returns a flat dict of scalar values.
    """
    close = df["Close"]

    ema_200_series = close.ewm(span=200, adjust=False).mean()
    ema_50_series  = close.ewm(span=50,  adjust=False).mean()
    atr_series     = _calc_atr(df, span=14)

    last_close  = float(close.iloc[-1])
    ema_200_val = float(ema_200_series.iloc[-1])
    ema_50_val  = float(ema_50_series.iloc[-1])

    # ema_50_slope: % change over last 5 days
    # requires ema_50[-6] (6 positions back: index -6 is the value 5 days before index -1)
    if len(ema_50_series) < 6:
        raise RuntimeError("Insufficient XJO history to compute ema_50_slope (need >= 6 rows).")
    ema_50_prev6 = float(ema_50_series.iloc[-6])
    ema_50_slope = (ema_50_val - ema_50_prev6) / ema_50_prev6 * 100.0

    above_200 = last_close > ema_200_val
    above_50  = last_close > ema_50_val

    atr_last  = float(atr_series.iloc[-1])
    atr_pct   = atr_last / last_close * 100.0

    # roc_20: (close[-1] / close[-21] - 1) * 100
    if len(close) < 21:
        raise RuntimeError("Insufficient XJO history to compute roc_20 (need >= 21 rows).")
    close_21ago = float(close.iloc[-21])
    roc_20 = (last_close / close_21ago - 1.0) * 100.0

    return {
        "xjo_close":    round(last_close,   2),
        "ema_200":      round(ema_200_val,   2),
        "ema_50":       round(ema_50_val,    2),
        "ema_50_slope": round(ema_50_slope,  4),
        "above_200":    bool(above_200),
        "above_50":     bool(above_50),
        "atr_pct":      round(atr_pct,       4),
        "roc_20":       round(roc_20,         4),
    }


def _calc_market_breadth() -> float:
    """
    Download ASX200 sample stocks (6mo) in one batch call.
    Returns % of stocks whose last close is above their own 50 EMA.
    Handles partial failures gracefully.
    """
    log.info("Downloading %d ASX200 sample stocks for breadth calculation ...", len(ASX200_SAMPLE))

    with _suppress_stderr():
        raw = yf.download(
            ASX200_SAMPLE,
            period="6mo",
            auto_adjust=True,
            progress=False,
            threads=True,
            group_by="ticker",
        )

    if raw is None or raw.empty:
        log.warning("Breadth: yfinance returned empty data for ASX200 sample. Using 50.0 as fallback.")
        return 50.0

    above_count = 0
    checked     = 0
    failed      = []

    for ticker in ASX200_SAMPLE:
        try:
            # Multi-ticker download: columns are (ticker, field) MultiIndex
            if isinstance(raw.columns, pd.MultiIndex):
                if ticker not in raw.columns.get_level_values(0):
                    failed.append(ticker)
                    continue
                close_series = raw[ticker]["Close"].dropna()
            else:
                # Single ticker fallback (shouldn't happen but be safe)
                close_series = raw["Close"].dropna()

            if len(close_series) < 50:
                log.debug("Breadth: %s has only %d rows — skipping.", ticker, len(close_series))
                failed.append(ticker)
                continue

            ema_50 = close_series.ewm(span=50, adjust=False).mean()
            last_close = float(close_series.iloc[-1])
            last_ema50 = float(ema_50.iloc[-1])

            if last_close > last_ema50:
                above_count += 1
            checked += 1

        except Exception as exc:
            log.debug("Breadth: failed for %s — %s", ticker, exc)
            failed.append(ticker)

    if failed:
        log.warning(
            "Breadth: %d/%d stocks failed or had insufficient data: %s",
            len(failed), len(ASX200_SAMPLE), ", ".join(failed),
        )

    if checked == 0:
        log.warning("Breadth: no stocks successfully computed. Using 50.0 as fallback.")
        return 50.0

    breadth = (above_count / checked) * 100.0
    log.info("Breadth: %d/%d stocks above 50 EMA = %.1f%%", above_count, checked, breadth)
    return round(breadth, 2)


def _classify_regime(
    above_200:    bool,
    atr_pct:      float,
    ema_50_slope: float,
    market_breadth: float,
) -> str:
    """Return regime string in priority order."""
    if atr_pct > 2.5:
        return "HIGH_VOL"
    if not above_200:
        return "BEAR"
    if abs(ema_50_slope) < 0.1 and market_breadth < 50:
        return "CHOPPY"
    return "BULL"


def _calc_confidence(
    regime:         str,
    above_200:      bool,
    atr_pct:        float,
    ema_50_slope:   float,
    roc_20:         float,
    market_breadth: float,
    xjo_close:      float,
    ema_200:        float,
) -> int:
    """Compute confidence score (0–100) based on regime."""
    if regime == "BULL":
        score = 100
        if market_breadth < 60:
            score -= 20
        if roc_20 < 2:
            score -= 15
        if ema_50_slope < 0.1:
            score -= 10

    elif regime == "BEAR":
        score = 100
        # penalise if close is within 1% of ema_200 (borderline bear)
        if ema_200 > 0 and abs(xjo_close - ema_200) / ema_200 * 100 < 1.0:
            score -= 20
        if roc_20 > -2:
            score -= 10

    elif regime == "CHOPPY":
        score = 70
        if market_breadth < 40:
            score += 15
        if abs(ema_50_slope) < 0.05:
            score += 15

    elif regime == "HIGH_VOL":
        score = 100
        if atr_pct < 3.0:
            score -= 20

    else:
        score = 50  # unknown fallback

    return int(max(0, min(100, score)))


_POSITION_SIZE_MAP = {
    "BULL":     1.0,
    "CHOPPY":   0.5,
    "BEAR":     0.0,
    "HIGH_VOL": 0.25,
}


# ── main entry point ──────────────────────────────────────────────────────────

def run_regime_detector() -> dict:
    """
    Run full regime detection pipeline.

    Returns a dict with the same structure as results/regime.json:
        {
            "regime": str,
            "confidence": int,
            "position_size_multiplier": float,
            "timestamp": str (ISO 8601),
            "indicators": { ... }
        }

    Also writes results/regime.json.
    Raises RuntimeError if XJO data cannot be downloaded.
    """
    log.info("=" * 60)
    log.info("  Regime Detector — %s", datetime.now().strftime("%d %b %Y %H:%M"))
    log.info("=" * 60)

    # 1. Download XJO
    xjo_df = _download_xjo()

    # 2. Compute XJO indicators
    log.info("Computing XJO indicators ...")
    xjo_ind = _calc_xjo_indicators(xjo_df)

    above_200    = xjo_ind["above_200"]
    above_50     = xjo_ind["above_50"]
    atr_pct      = xjo_ind["atr_pct"]
    ema_50_slope = xjo_ind["ema_50_slope"]
    roc_20       = xjo_ind["roc_20"]
    xjo_close    = xjo_ind["xjo_close"]
    ema_200      = xjo_ind["ema_200"]

    log.info(
        "XJO: close=%.2f | ema200=%.2f | ema50=%.2f | slope=%.4f%% | "
        "above200=%s | above50=%s | atr_pct=%.4f%% | roc20=%.4f%%",
        xjo_close, ema_200, xjo_ind["ema_50"], ema_50_slope,
        above_200, above_50, atr_pct, roc_20,
    )

    # 3. Market breadth
    market_breadth = _calc_market_breadth()

    # 4. Classify regime
    regime = _classify_regime(above_200, atr_pct, ema_50_slope, market_breadth)
    log.info("Regime classified: %s", regime)

    # 5. Confidence score
    confidence = _calc_confidence(
        regime         = regime,
        above_200      = above_200,
        atr_pct        = atr_pct,
        ema_50_slope   = ema_50_slope,
        roc_20         = roc_20,
        market_breadth = market_breadth,
        xjo_close      = xjo_close,
        ema_200        = ema_200,
    )
    log.info("Confidence: %d/100", confidence)

    # 6. Position size multiplier
    psm = _POSITION_SIZE_MAP.get(regime, 0.0)
    log.info("Position size multiplier: %.2f", psm)

    # 7. Build result dict
    result = {
        "regime":                   regime,
        "confidence":               confidence,
        "position_size_multiplier": float(psm),
        "timestamp":                datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
        "indicators": {
            "xjo_close":      float(xjo_ind["xjo_close"]),
            "ema_200":        float(xjo_ind["ema_200"]),
            "ema_50":         float(xjo_ind["ema_50"]),
            "ema_50_slope":   float(xjo_ind["ema_50_slope"]),
            "above_200":      bool(xjo_ind["above_200"]),
            "above_50":       bool(xjo_ind["above_50"]),
            "atr_pct":        float(xjo_ind["atr_pct"]),
            "roc_20":         float(xjo_ind["roc_20"]),
            "market_breadth": float(market_breadth),
        },
    }

    # 8. Write regime.json
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "regime.json"
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2)
    log.info("Regime written -> %s", out_path)

    log.info("=" * 60)
    return result


# ── standalone ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import pprint
    result = run_regime_detector()
    print()
    pprint.pprint(result)
