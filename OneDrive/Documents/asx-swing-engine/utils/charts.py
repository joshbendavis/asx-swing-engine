"""
utils/charts.py
---------------
Generate dark-themed candlestick + EMA50/200 + Volume + RSI charts
for the top-N stocks from the screener output.

Output: PNG files saved to results/charts/YYYYMMDD/
Returns: list of (ticker, filepath) tuples
"""

import os
import warnings
from datetime import datetime

import matplotlib
matplotlib.use("Agg")  # non-interactive backend — safe for scheduled runs
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")

# ── colour palette ────────────────────────────────────────────────────────────
BG        = "#0d0d0d"
PANEL_BG  = "#111111"
GREEN     = "#26a69a"
RED       = "#ef5350"
EMA50_C   = "#f5a623"   # amber
EMA200_C  = "#4a90e2"   # blue
RSI_C     = "#ce93d8"   # soft purple
GRID_C    = "#1e1e1e"
TICK_C    = "#888888"


# ── helpers ───────────────────────────────────────────────────────────────────

def _ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta    = close.diff()
    gain     = delta.clip(lower=0)
    loss     = (-delta).clip(lower=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()
    rs       = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _style_ax(ax):
    ax.set_facecolor(PANEL_BG)
    ax.tick_params(colors=TICK_C, labelsize=7)
    ax.yaxis.tick_right()
    ax.yaxis.set_label_position("right")
    for spine in ax.spines.values():
        spine.set_edgecolor(GRID_C)
    ax.grid(axis="y", color=GRID_C, linewidth=0.5, linestyle="--")


# ── main function ─────────────────────────────────────────────────────────────

def generate_charts(
    results_df: pd.DataFrame,
    top_n: int = 10,
    output_dir: str = "results/charts",
    history_period: str = "6mo",
) -> list[tuple[str, str]]:
    """
    Generate one chart per ticker for the top_n rows in results_df.

    Parameters
    ----------
    results_df   : screener output DataFrame (must include 'ticker' column)
    top_n        : number of stocks to chart
    output_dir   : root folder; charts go into a YYYYMMDD sub-folder
    history_period: yfinance period string for OHLCV download

    Returns
    -------
    List of (ticker, absolute_filepath) tuples for successfully created charts.
    """
    today     = datetime.today().strftime("%Y%m%d")
    chart_dir = os.path.join(output_dir, today)
    os.makedirs(chart_dir, exist_ok=True)

    top   = results_df.head(top_n).reset_index()   # 'rank' becomes a column
    paths = []

    for _, row in top.iterrows():
        ticker = row["ticker"]
        print(f"  Charting {ticker} ...")

        try:
            raw = yf.download(
                ticker,
                period=history_period,
                interval="1d",
                auto_adjust=True,
                progress=False,
            )

            # Flatten MultiIndex if present (single ticker → may or may not have it)
            if isinstance(raw.columns, pd.MultiIndex):
                raw.columns = raw.columns.get_level_values(0)

            raw = raw.dropna(how="all")
            if raw.empty or len(raw) < 50:
                print(f"    Skipping {ticker}: insufficient data.")
                continue

            close  = raw["Close"].astype(float)
            high   = raw["High"].astype(float)
            low    = raw["Low"].astype(float)
            open_  = raw["Open"].astype(float)
            volume = raw["Volume"].astype(float)

            ema50  = _ema(close, 50)
            ema200 = _ema(close, 200)
            rsi    = _rsi(close, 14)

            xs = np.arange(len(raw))   # integer x-axis (avoids weekend gaps)

            # ── figure layout: 3 rows (price 3, volume 1, RSI 1) ──────────────
            fig = plt.figure(figsize=(14, 9))
            fig.patch.set_facecolor(BG)

            gs  = gridspec.GridSpec(3, 1, height_ratios=[3, 1, 1], hspace=0.04)
            ax1 = fig.add_subplot(gs[0])   # candlestick + EMAs
            ax2 = fig.add_subplot(gs[1], sharex=ax1)   # volume
            ax3 = fig.add_subplot(gs[2], sharex=ax1)   # RSI

            for ax in (ax1, ax2, ax3):
                _style_ax(ax)

            # ── panel 1: candlesticks ─────────────────────────────────────────
            bar_colors = np.where(close.values >= open_.values, GREEN, RED)

            # wicks
            ax1.vlines(xs, low.values, high.values, colors=bar_colors, linewidth=0.6)
            # bodies
            ax1.bar(xs, (close - open_).values, bottom=open_.values,
                    color=bar_colors, width=0.75, alpha=0.90)

            # EMAs
            ax1.plot(xs, ema50.values,  color=EMA50_C,  linewidth=1.3, label="EMA 50",  zorder=3)
            ax1.plot(xs, ema200.values, color=EMA200_C, linewidth=1.3, label="EMA 200", zorder=3)

            ax1.legend(
                loc="upper left", fontsize=8,
                facecolor="#1a1a1a", labelcolor="white",
                framealpha=0.75, edgecolor="#333333",
            )

            score    = row.get("composite_score", float("nan"))
            rs_val   = row.get("rs_vs_xjo",       float("nan"))
            mom_val  = row.get("momentum_20d",     float("nan"))
            price    = row.get("last_price",       float("nan"))

            ax1.set_title(
                f"{ticker}   |   Last: ${price:.2f}"
                f"   |   Score: {score:.1f}"
                f"   |   RS vs XJO: {rs_val:+.1f}%"
                f"   |   Mom 20d: {mom_val:+.1f}%",
                color="white", fontsize=10, pad=8, loc="left",
            )
            plt.setp(ax1.get_xticklabels(), visible=False)

            # ── panel 2: volume ───────────────────────────────────────────────
            ax2.bar(xs, volume.values, color=bar_colors, width=0.75, alpha=0.65)
            ax2.set_ylabel("Vol", color=TICK_C, fontsize=7)
            ax2.yaxis.set_major_formatter(
                mticker.FuncFormatter(lambda v, _: f"{v/1e6:.1f}M" if v >= 1e6 else f"{v/1e3:.0f}K")
            )
            plt.setp(ax2.get_xticklabels(), visible=False)

            # ── panel 3: RSI ──────────────────────────────────────────────────
            ax3.plot(xs, rsi.values, color=RSI_C, linewidth=1.0)
            ax3.axhline(70, color=RED,      linewidth=0.6, linestyle="--", alpha=0.7)
            ax3.axhline(50, color=TICK_C,   linewidth=0.6, linestyle="--", alpha=0.4)
            ax3.axhline(30, color=GREEN,    linewidth=0.6, linestyle="--", alpha=0.7)
            ax3.fill_between(xs, rsi.values, 50,
                             where=(rsi.values >= 50), alpha=0.12, color=GREEN)
            ax3.fill_between(xs, rsi.values, 50,
                             where=(rsi.values <  50), alpha=0.12, color=RED)
            ax3.set_ylim(0, 100)
            ax3.set_ylabel("RSI", color=TICK_C, fontsize=7)

            # x-axis date labels on bottom panel
            step       = max(1, len(raw) // 8)
            tick_pos   = list(range(0, len(raw), step))
            tick_lbls  = [raw.index[i].strftime("%d %b") for i in tick_pos]
            ax3.set_xticks(tick_pos)
            ax3.set_xticklabels(tick_lbls, rotation=0, fontsize=7, color=TICK_C)

            # ── save ──────────────────────────────────────────────────────────
            out_path = os.path.join(chart_dir, f"{ticker.replace('.AX', '')}.png")
            plt.savefig(out_path, dpi=130, bbox_inches="tight",
                        facecolor=fig.get_facecolor())
            plt.close(fig)

            paths.append((ticker, os.path.abspath(out_path)))
            print(f"    Saved -> {out_path}")

        except Exception as exc:
            print(f"    Chart failed for {ticker}: {exc}")
            plt.close("all")
            continue

    return paths
