"""
backtest.py — ASX Swing Engine walk-forward backtest
-----------------------------------------------------
Signal  : all technical screener filters pass at bar D's CLOSE
Entry   : open of bar D+1
Stop    : entry − 2 × ATR(14)  [from signal bar]
Target  : entry + 3 × initial_risk  (3:1)
Breakeven: stop moves to entry once close ≥ entry + 1 × initial_risk
Time    : exit at CLOSE of bar D+1+20 if neither stop nor target hit
All exits: on CLOSE (not intraday)

Usage
-----
  python backtest.py                   # preset ~100-stock ASX universe, 2y
  python backtest.py --period 3y       # longer lookback
  python backtest.py --from-screener   # universe = last screener_output.csv
  python backtest.py --top-n 30        # top 30 from screener
"""

import argparse
import contextlib
import os
import sys
import time
from pathlib import Path
from datetime import date

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import yfinance as yf

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
ATR_STOP_MULTIPLE = 2.0     # stop   = entry − N × ATR
TARGET_R          = 2.0     # target = entry + TARGET_R × initial_risk
BREAKEVEN_R       = 1.0     # move stop to entry once this R multiple hit
TIME_STOP_DAYS    = 20      # bars before forced exit
POSITION_SIZE     = 10_000  # AUD per trade (fixed)
COMMISSION        = 10.0    # AUD round-trip

# Screener filter constants (mirrors screener.py)
MIN_PRICE          = 0.50
MAX_PRICE          = 50.0    # new hard cap
MIN_AVG_VOL_20     = 200_000
EMA_FAST           = 50
EMA_SLOW           = 200
RSI_PERIOD         = 14
RSI_MIN, RSI_MAX   = 40, 65
MAX_EMA50_DIST_PCT = 10.0
ATR_PERIOD         = 14
MIN_ATR_PCT        = 3.0     # new hard floor
RS_PERIOD          = 63      # bars for relative strength vs XJO
BENCHMARK          = "^AXJO"
WARMUP             = EMA_SLOW + RS_PERIOD + 5  # enough for EMA200 + 63d RS

BATCH_SIZE  = 20
BATCH_DELAY = 1.0

RESULTS_DIR = Path("results/backtest")

# ---------------------------------------------------------------------------
# Preset universe — ~100 liquid ASX names with multi-year history
# ---------------------------------------------------------------------------
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

def download_universe(tickers: list[str], period: str) -> dict[str, pd.DataFrame]:
    """Batch-download OHLCV; return {ticker: df}."""
    data: dict[str, pd.DataFrame] = {}
    batches = [tickers[i:i+BATCH_SIZE] for i in range(0, len(tickers), BATCH_SIZE)]

    for idx, batch in enumerate(batches, 1):
        print(f"  Downloading batch {idx}/{len(batches)} …")
        with _quiet():
            raw = yf.download(batch, period=period, interval="1d",
                              auto_adjust=True, progress=False, threads=True)
        if raw.empty:
            continue

        if isinstance(raw.columns, pd.MultiIndex):
            available = raw.columns.get_level_values(1).unique()
            for t in batch:
                if t in available:
                    df = raw.xs(t, axis=1, level=1).dropna(how="all")
                    if len(df) >= WARMUP:
                        data[t] = df
        else:
            if len(batch) == 1:
                df = raw.dropna(how="all")
                if len(df) >= WARMUP:
                    data[batch[0]] = df

        if idx < len(batches):
            time.sleep(BATCH_DELAY)

    return data


# ---------------------------------------------------------------------------
# Signal scanner
# ---------------------------------------------------------------------------

def compute_indicators(df: pd.DataFrame,
                       xjo_close: pd.Series | None = None) -> pd.DataFrame:
    """Add indicator columns to df. Returns copy."""
    d = df.copy()
    d["ema50"]      = _ema(d["Close"], EMA_FAST)
    d["ema200"]     = _ema(d["Close"], EMA_SLOW)
    d["rsi"]        = _rsi(d["Close"], RSI_PERIOD)
    d["atr"]        = _atr(d["High"], d["Low"], d["Close"], ATR_PERIOD)
    d["avg_vol"]    = d["Volume"].rolling(20).mean()
    d["ema50_dist"] = (d["Close"] / d["ema50"] - 1) * 100
    d["atr_pct"]    = d["atr"] / d["Close"] * 100

    # Rolling RS vs XJO: stock 63d return minus XJO 63d return
    d["stock_r63"] = d["Close"].pct_change(RS_PERIOD)
    if xjo_close is not None:
        xjo_r63 = xjo_close.pct_change(RS_PERIOD)
        d["xjo_r63"]   = xjo_r63.reindex(d.index, method="ffill")
        d["rs_vs_xjo"] = d["stock_r63"] - d["xjo_r63"]
    else:
        d["rs_vs_xjo"] = np.nan
    return d


def find_signals(d: pd.DataFrame, use_new_filters: bool = False) -> pd.Series:
    """Boolean Series — True on bars where all screener filters pass."""
    sig = (
        (d["Close"]      >  MIN_PRICE)          &
        (d["avg_vol"]    >  MIN_AVG_VOL_20)     &
        (d["Close"]      >  d["ema200"])         &
        (d["Close"]      >  d["ema50"])          &
        (d["ema50_dist"] <= MAX_EMA50_DIST_PCT)  &
        (d["rsi"]        >= RSI_MIN)             &
        (d["rsi"]        <= RSI_MAX)
    )
    if use_new_filters:
        sig = (sig
               & (d["Close"]      <= MAX_PRICE)
               & (d["atr_pct"]    >= MIN_ATR_PCT)
               & (d["rs_vs_xjo"]  >  0))
    return sig


# ---------------------------------------------------------------------------
# Trade simulator (per ticker)
# ---------------------------------------------------------------------------

def simulate_ticker(
    ticker: str,
    df: pd.DataFrame,
    target_r: float = TARGET_R,
    trail_atr_multiple: float | None = None,
    xjo_close: pd.Series | None = None,
    use_new_filters: bool = False,
) -> list[dict]:
    """
    Simulate all trades for one ticker. Returns list of trade dicts.

    Parameters
    ----------
    target_r           : R-multiple target (e.g. 2.0 or 3.0)
    trail_atr_multiple : if set, trail stop at this × ATR(14) after breakeven;
                         if None, stop stays at breakeven entry price only.

    State machine:
      WAITING  -> signal on bar i -> plan entry at bar i+1 open
      IN_TRADE -> each close: update trailing stop, then check stop / target / time
    """
    d = compute_indicators(df, xjo_close=xjo_close)
    signals = find_signals(d, use_new_filters=use_new_filters)

    closes  = d["Close"].values
    opens   = d["Open"].values
    atrs    = d["atr"].values
    dates   = d.index
    n       = len(d)

    trades: list[dict] = []

    # State
    in_trade       = False
    entry_bar      = -1
    entry_price    = 0.0
    current_stop   = 0.0
    target         = 0.0
    breakeven_lvl  = 0.0
    initial_risk   = 0.0
    shares         = 0
    breakeven_set  = False
    trail_high     = 0.0     # highest close seen since breakeven triggered
    cooldown_until = 0

    for i in range(WARMUP, n):

        # ── IN TRADE ─────────────────────────────────────────────────────────
        if in_trade and i > entry_bar:
            c       = closes[i]
            holding = i - entry_bar

            # 1. Breakeven: once 1R is hit, move stop to entry
            if not breakeven_set and c >= breakeven_lvl:
                current_stop  = entry_price
                breakeven_set = True
                trail_high    = c   # start tracking high from here

            # 2. Trailing stop: after breakeven, ratchet stop up with price
            if breakeven_set and trail_atr_multiple is not None:
                trail_high   = max(trail_high, c)
                trail_stop   = trail_high - trail_atr_multiple * atrs[i]
                current_stop = max(current_stop, trail_stop)  # only move up

            # 3. Check exits (on close)
            exit_type = exit_price = None
            if c <= current_stop:
                exit_type, exit_price = "stop", c
            elif c >= target:
                exit_type, exit_price = "target", c
            elif holding >= TIME_STOP_DAYS:
                exit_type, exit_price = "time", c

            if exit_type:
                pnl_r   = (exit_price - entry_price) / initial_risk
                pnl_aud = (exit_price - entry_price) * shares - COMMISSION
                trades.append({
                    "ticker":        ticker,
                    "entry_date":    dates[entry_bar],
                    "exit_date":     dates[i],
                    "entry_price":   round(entry_price, 4),
                    "exit_price":    round(exit_price, 4),
                    "stop":          round(current_stop, 4),
                    "target":        round(target, 4),
                    "initial_risk":  round(initial_risk, 4),
                    "shares":        shares,
                    "holding_days":  holding,
                    "exit_type":     exit_type,
                    "breakeven_hit": breakeven_set,
                    "pnl_r":         round(pnl_r, 3),
                    "pnl_aud":       round(pnl_aud, 2),
                })
                in_trade       = False
                cooldown_until = i + 3
                continue

        # ── WAITING ──────────────────────────────────────────────────────────
        elif not in_trade and i >= cooldown_until:
            if signals.iloc[i] and i + 1 < n:
                ep   = opens[i + 1]
                risk = atrs[i] * ATR_STOP_MULTIPLE
                if risk <= 0 or ep <= 0:
                    continue
                shs = int(POSITION_SIZE / ep)
                if shs == 0:
                    continue

                in_trade      = True
                entry_bar     = i + 1
                entry_price   = ep
                initial_risk  = risk
                current_stop  = ep - risk
                target        = ep + target_r * risk
                breakeven_lvl = ep + BREAKEVEN_R * risk
                breakeven_set = False
                trail_high    = ep
                shares        = shs

    return trades


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def calc_metrics(trades_df: pd.DataFrame) -> dict:
    if trades_df.empty:
        return {}

    n         = len(trades_df)
    wins      = trades_df[trades_df["exit_type"] == "target"]
    losses    = trades_df[trades_df["exit_type"] == "stop"]
    time_outs = trades_df[trades_df["exit_type"] == "time"]

    win_rate  = len(wins) / n * 100
    gross_win = wins["pnl_aud"].sum()
    gross_loss= losses["pnl_aud"].sum()
    pf        = gross_win / abs(gross_loss) if gross_loss != 0 else np.nan
    avg_r     = trades_df["pnl_r"].mean()
    expectancy= trades_df["pnl_aud"].mean()
    avg_hold  = trades_df["holding_days"].mean()
    total_pnl = trades_df["pnl_aud"].sum()

    # Max drawdown on cumulative equity
    equity  = trades_df.sort_values("exit_date")["pnl_aud"].cumsum()
    peak    = equity.cummax()
    dd      = equity - peak
    max_dd  = dd.min()

    # Trade-level Sharpe (annualised)
    pnl_r_std = trades_df["pnl_r"].std()
    if pnl_r_std > 0 and avg_hold > 0:
        sharpe = (avg_r / pnl_r_std) * np.sqrt(252 / avg_hold)
    else:
        sharpe = np.nan

    return {
        "total_trades":   n,
        "win_rate_pct":   round(win_rate, 1),
        "wins":           len(wins),
        "losses":         len(losses),
        "time_stops":     len(time_outs),
        "profit_factor":  round(pf, 2) if not np.isnan(pf) else "n/a",
        "avg_r":          round(avg_r, 3),
        "expectancy_aud": round(expectancy, 2),
        "avg_hold_days":  round(avg_hold, 1),
        "total_pnl_aud":  round(total_pnl, 2),
        "max_drawdown_aud": round(max_dd, 2),
        "sharpe":         round(sharpe, 2) if not np.isnan(sharpe) else "n/a",
        "best_trade_aud": round(trades_df["pnl_aud"].max(), 2),
        "worst_trade_aud":round(trades_df["pnl_aud"].min(), 2),
    }


# ---------------------------------------------------------------------------
# Chart
# ---------------------------------------------------------------------------

def save_chart(trades_df: pd.DataFrame, metrics: dict, out_path: Path):
    BG, PANEL = "#0d1117", "#161b22"
    GREEN, RED, GREY = "#26a641", "#da3633", "#8b949e"

    fig = plt.figure(figsize=(16, 10), facecolor=BG)
    gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.38, wspace=0.32,
                             height_ratios=[3, 2])

    ax_eq   = fig.add_subplot(gs[0, :])   # equity curve — full width
    ax_hist = fig.add_subplot(gs[1, 0])   # R-distribution histogram
    ax_stat = fig.add_subplot(gs[1, 1])   # stats table

    for ax in (ax_eq, ax_hist, ax_stat):
        ax.set_facecolor(PANEL)
        ax.tick_params(colors=GREY, labelsize=8)
        for sp in ax.spines.values():
            sp.set_edgecolor("#21262d")

    # ── Equity curve ────────────────────────────────────────────────────────
    sorted_t  = trades_df.sort_values("exit_date")
    equity    = sorted_t["pnl_aud"].cumsum().values
    exit_dates= pd.to_datetime(sorted_t["exit_date"])
    xs        = range(len(equity))

    ax_eq.plot(xs, equity, color=GREEN, lw=1.5, zorder=3)
    ax_eq.fill_between(xs, equity, 0,
                        where=(np.array(equity) >= 0),
                        color=GREEN, alpha=0.12)
    ax_eq.fill_between(xs, equity, 0,
                        where=(np.array(equity) < 0),
                        color=RED, alpha=0.12)
    ax_eq.axhline(0, color=GREY, lw=0.8, ls="--", alpha=0.5)

    # Drawdown shading
    peak = np.maximum.accumulate(equity)
    dd   = equity - peak
    ax_eq.fill_between(xs, equity, peak, where=(dd < 0),
                        color=RED, alpha=0.08)

    ax_eq.set_title("Equity Curve (cumulative P&L AUD)", color="white",
                     fontsize=11, pad=8)
    ax_eq.set_ylabel("AUD", color=GREY, fontsize=9)
    ax_eq.grid(color="#21262d", lw=0.5, alpha=0.7)
    ax_eq.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda v, _: f"${v:,.0f}")
    )

    # x-tick: every ~50 trades, label with date
    step = max(1, len(equity) // 8)
    tick_pos = list(range(0, len(equity), step))
    tick_lbl = [exit_dates.iloc[i].strftime("%b %y")
                for i in tick_pos if i < len(exit_dates)]
    ax_eq.set_xticks(tick_pos[:len(tick_lbl)])
    ax_eq.set_xticklabels(tick_lbl, color=GREY, fontsize=8)

    # ── R distribution ───────────────────────────────────────────────────────
    r_vals  = trades_df["pnl_r"].values
    colours = [GREEN if r > 0 else RED for r in r_vals]
    ax_hist.bar(range(len(r_vals)),
                sorted(r_vals),
                color=sorted([GREEN if r > 0 else RED for r in r_vals],
                             key=lambda c: c != RED),
                alpha=0.8, width=0.8)
    ax_hist.axhline(0, color=GREY, lw=0.8, ls="--")
    ax_hist.set_title("Trade P&L (sorted, R multiples)", color="white",
                       fontsize=10, pad=6)
    ax_hist.set_ylabel("R", color=GREY, fontsize=9)
    ax_hist.grid(color="#21262d", lw=0.5, alpha=0.7, axis="y")
    ax_hist.tick_params(bottom=False, labelbottom=False)

    # ── Stats table ──────────────────────────────────────────────────────────
    ax_stat.axis("off")
    rows = [
        ("Total trades",     str(metrics.get("total_trades", "—"))),
        ("Win rate",         f"{metrics.get('win_rate_pct','—')}%"),
        ("  Targets hit",    str(metrics.get("wins", "—"))),
        ("  Stops hit",      str(metrics.get("losses", "—"))),
        ("  Time stops",     str(metrics.get("time_stops", "—"))),
        ("Profit factor",    str(metrics.get("profit_factor", "—"))),
        ("Avg R per trade",  str(metrics.get("avg_r", "—"))),
        ("Expectancy",       f"${metrics.get('expectancy_aud','—'):,}"),
        ("Avg hold (days)",  str(metrics.get("avg_hold_days", "—"))),
        ("Total P&L",        f"${metrics.get('total_pnl_aud','—'):,}"),
        ("Max drawdown",     f"${metrics.get('max_drawdown_aud','—'):,}"),
        ("Sharpe",           str(metrics.get("sharpe", "—"))),
        ("Best trade",       f"${metrics.get('best_trade_aud','—'):,}"),
        ("Worst trade",      f"${metrics.get('worst_trade_aud','—'):,}"),
    ]
    y = 0.97
    for label, val in rows:
        is_header = not label.startswith("  ")
        col = "white" if is_header else GREY
        ax_stat.text(0.02, y, label, transform=ax_stat.transAxes,
                     color=col, fontsize=9, va="top")
        ax_stat.text(0.98, y, val,   transform=ax_stat.transAxes,
                     color=GREEN if (isinstance(val, str) and val.startswith("$") and
                                     not val.startswith("$-")) else col,
                     fontsize=9, va="top", ha="right", fontweight="bold")
        y -= 0.067

    ax_stat.set_title("Summary Statistics", color="white", fontsize=10, pad=6)
    ax_stat.set_facecolor(PANEL)

    params_text = (f"Stop: {ATR_STOP_MULTIPLE}× ATR  |  "
                   f"Target: {TARGET_R}:1  |  "
                   f"Breakeven at: 1R  |  "
                   f"Time stop: {TIME_STOP_DAYS}d  |  "
                   f"Position: ${POSITION_SIZE:,}/trade")
    fig.text(0.5, 0.01, params_text, color=GREY, fontsize=8,
             ha="center")

    plt.savefig(out_path, dpi=130, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"Chart saved -> {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="ASX Swing Engine — backtest")
    p.add_argument("--period",        default="2y",
                   help="yfinance period string (default: 2y)")
    p.add_argument("--from-screener", action="store_true",
                   help="Use last screener_output.csv as universe")
    p.add_argument("--top-n",         type=int, default=None,
                   help="Use top N tickers from screener (requires --from-screener)")
    return p.parse_args()


SCENARIOS = [
    {
        "label":              "Baseline: 2:1, original filters",
        "target_r":           2.0,
        "trail_atr_multiple": None,
        "use_new_filters":    False,
    },
    {
        "label":              "New filters: ATR>=3%, price<=50, RS>XJO",
        "target_r":           2.0,
        "trail_atr_multiple": None,
        "use_new_filters":    True,
    },
]


def _run_scenario(price_data: dict, target_r: float,
                  trail_atr_multiple: float | None,
                  xjo_close: pd.Series | None = None,
                  use_new_filters: bool = False) -> pd.DataFrame:
    all_trades: list[dict] = []
    for ticker, df in price_data.items():
        all_trades.extend(
            simulate_ticker(ticker, df,
                            target_r=target_r,
                            trail_atr_multiple=trail_atr_multiple,
                            xjo_close=xjo_close,
                            use_new_filters=use_new_filters)
        )
    if not all_trades:
        return pd.DataFrame()
    tdf = pd.DataFrame(all_trades)
    tdf["entry_date"] = pd.to_datetime(tdf["entry_date"])
    tdf["exit_date"]  = pd.to_datetime(tdf["exit_date"])
    return tdf.sort_values("exit_date").reset_index(drop=True)


def _comparison_table(results: list[tuple[str, dict]]) -> None:
    """Print a side-by-side comparison of key metrics across scenarios."""
    KEY_METRICS = [
        ("total_trades",    "Trades"),
        ("win_rate_pct",    "Win rate %"),
        ("wins",            "  Targets hit"),
        ("losses",          "  Stops hit"),
        ("time_stops",      "  Time stops"),
        ("profit_factor",   "Profit factor"),
        ("avg_r",           "Avg R / trade"),
        ("expectancy_aud",  "Expectancy $"),
        ("total_pnl_aud",   "Total P&L $"),
        ("max_drawdown_aud","Max drawdown $"),
        ("sharpe",          "Sharpe"),
        ("avg_hold_days",   "Avg hold days"),
    ]

    labels = [label for label, _ in results]
    col_w  = max(28, max(len(l) for l in labels) + 2)
    metric_w = 18

    # Header
    print("\n" + "=" * (metric_w + col_w * len(labels) + 2))
    print("  SCENARIO COMPARISON")
    print("=" * (metric_w + col_w * len(labels) + 2))
    header = f"  {'Metric':<{metric_w}}"
    for lbl in labels:
        header += f"  {lbl:>{col_w - 2}}"
    print(header)
    print("-" * (metric_w + col_w * len(labels) + 2))

    for key, display in KEY_METRICS:
        row = f"  {display:<{metric_w}}"
        vals = [m.get(key, "—") for _, m in results]
        # Highlight best value (highest numeric) in each row
        numeric = [v for v in vals if isinstance(v, (int, float))]
        best = max(numeric) if numeric and key not in (
            "losses", "time_stops", "max_drawdown_aud"
        ) else (min(numeric) if numeric else None)
        for v in vals:
            cell = f"${v:,.2f}" if key in ("expectancy_aud","total_pnl_aud","max_drawdown_aud") else str(v)
            marker = " *" if (isinstance(v, (int, float)) and v == best) else "  "
            row += f"  {cell:>{col_w - 4}}{marker}"
        print(row)

    print("=" * (metric_w + col_w * len(labels) + 2))
    print("  * = best value in row")


def main():
    args = parse_args()

    # ── Universe ──────────────────────────────────────────────────────────────
    if args.from_screener:
        csv = Path("results/screener_output.csv")
        if not csv.exists():
            print("ERROR: results/screener_output.csv not found. Run screener.py first.")
            sys.exit(1)
        screener_df = pd.read_csv(csv, index_col="rank")
        tickers = screener_df["ticker"].tolist()
        if args.top_n:
            tickers = tickers[:args.top_n]
        print(f"Universe: {len(tickers)} tickers from screener output")
    else:
        tickers = list(dict.fromkeys(ASX_PRESET))
        print(f"Universe: {len(tickers)} preset ASX tickers")

    print(f"Period  : {args.period}")
    print(f"Fixed   : stop={ATR_STOP_MULTIPLE}x ATR  breakeven@1R  time_stop={TIME_STOP_DAYS}d")
    print()

    # ── Download benchmark (for RS vs XJO filter) ────────────────────────────
    print(f"Downloading benchmark ({BENCHMARK}) ...")
    with _quiet():
        xjo_raw = yf.download(BENCHMARK, period=args.period, interval="1d",
                              auto_adjust=True, progress=False)
    _xjo = xjo_raw["Close"]
    xjo_close = (_xjo.iloc[:, 0] if isinstance(_xjo, pd.DataFrame) else _xjo).dropna()
    print(f"  {len(xjo_close)} benchmark bars.\n")

    # ── Download once — reused across all scenarios ───────────────────────────
    print("Downloading price history ...")
    price_data = download_universe(tickers, args.period)
    print(f"  Got data for {len(price_data)} / {len(tickers)} tickers.\n")

    if not price_data:
        print("No data downloaded. Check connection.")
        sys.exit(1)

    # ── Run all scenarios ─────────────────────────────────────────────────────
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    today = date.today().strftime("%Y%m%d")

    scenario_results: list[tuple[str, dict]] = []
    all_trades_per_scenario: list[tuple[str, pd.DataFrame]] = []

    for sc in SCENARIOS:
        print(f"Running: {sc['label']} ...")
        tdf = _run_scenario(price_data, sc["target_r"], sc["trail_atr_multiple"],
                            xjo_close=xjo_close,
                            use_new_filters=sc.get("use_new_filters", False))
        if tdf.empty:
            print("  No trades generated.")
            continue
        m = calc_metrics(tdf)
        scenario_results.append((sc["label"], m))
        all_trades_per_scenario.append((sc["label"], tdf))
        print(f"  {m['total_trades']} trades  |  "
              f"win rate {m['win_rate_pct']}%  |  "
              f"profit factor {m['profit_factor']}  |  "
              f"avg R {m['avg_r']}")

    # ── Side-by-side comparison ───────────────────────────────────────────────
    _comparison_table(scenario_results)

    # ── Save: trade logs + charts for each scenario ───────────────────────────
    print()
    for sc, (label, tdf) in zip(SCENARIOS, all_trades_per_scenario):
        slug = label.split(":")[0].lower().replace(" ", "_")
        csv_path   = RESULTS_DIR / f"trades_{slug}_{today}.csv"
        chart_path = RESULTS_DIR / f"backtest_{slug}_{today}.png"
        tdf.to_csv(csv_path, index=False)
        m = calc_metrics(tdf)
        save_chart(tdf, m, chart_path)
        print(f"  {slug}: log -> {csv_path}")


if __name__ == "__main__":
    main()
