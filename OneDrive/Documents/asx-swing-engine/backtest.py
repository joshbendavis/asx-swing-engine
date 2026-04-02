"""
backtest.py — ASX Swing Engine | Portfolio-level walk-forward backtest
----------------------------------------------------------------------
Signal  : all screener filters pass at bar D close
Entry   : open of bar D+1
Stop    : entry - 2xATR(14)   [ATR from signal bar]
Target  : entry + 2x initial_risk  (2:1 R:R)
Breakeven: stop moves to entry once close >= entry + 1R
Time    : exit at close of bar D+1+20 if still open
All exits: on CLOSE (not intraday)

Portfolio rules
  Starting capital : $20,000 AUD
  Risk per trade   : 1.5% of current account balance
  Max concurrent   : 6 open positions (= ~9% max live portfolio risk)
  Position sizing  : shares = floor(risk_aud / (2 x ATR))

Filters applied (tightened / data-driven):
  ATR(14) >= 3%  |  price <= $50  |  RS vs XJO > 0  (plus all base filters)

Usage
  python backtest.py                 # preset ~100-stock ASX universe, 5y
  python backtest.py --period 3y     # shorter lookback
  python backtest.py --from-screener # universe = last screener_output.csv
"""

import argparse
import contextlib
import os
import sys
import time
from collections import defaultdict
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
STARTING_BALANCE  = 20_000.0   # AUD
RISK_PCT          = 0.015      # 1.5% of account at risk per trade
MAX_POSITIONS     = 6          # max concurrent open positions
ATR_STOP_MULTIPLE = 2.0        # stop   = entry - N x ATR
TARGET_R          = 2.0        # target = entry + TARGET_R x initial_risk
BREAKEVEN_R       = 1.0        # move stop to entry once this R hit
TIME_STOP_DAYS    = 20         # bars before forced exit
COMMISSION        = 10.0       # AUD round-trip per trade

# Screener hard filters (mirrors screener.py — new filters always on)
MIN_PRICE          = 0.50
MAX_PRICE          = 50.0
MIN_AVG_VOL_20     = 200_000
EMA_FAST           = 50
EMA_SLOW           = 200
RSI_PERIOD         = 14
RSI_MIN, RSI_MAX   = 40, 65
MAX_EMA50_DIST_PCT = 10.0
ATR_PERIOD         = 14
MIN_ATR_PCT        = 3.0
RS_PERIOD          = 63
BENCHMARK          = "^AXJO"
WARMUP             = EMA_SLOW + RS_PERIOD + 5

BATCH_SIZE  = 20
BATCH_DELAY = 1.0
RESULTS_DIR = Path("results/backtest")

# ---------------------------------------------------------------------------
# Preset universe (~100 liquid ASX names with multi-year history)
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


def _ema(s, n): return s.ewm(span=n, adjust=False).mean()

def _rsi(close, n=14):
    d = close.diff()
    g = d.clip(lower=0).ewm(alpha=1/n, adjust=False).mean()
    l = (-d).clip(lower=0).ewm(alpha=1/n, adjust=False).mean()
    return 100 - 100 / (1 + g / l.replace(0, np.nan))

def _atr(high, low, close, n=14):
    pc = close.shift(1)
    tr = pd.concat([high-low, (high-pc).abs(), (low-pc).abs()], axis=1).max(axis=1)
    return tr.ewm(span=n, adjust=False).mean()


# ---------------------------------------------------------------------------
# Data download
# ---------------------------------------------------------------------------

def download_universe(tickers, period):
    data = {}
    batches = [tickers[i:i+BATCH_SIZE] for i in range(0, len(tickers), BATCH_SIZE)]
    for idx, batch in enumerate(batches, 1):
        print(f"  Batch {idx}/{len(batches)} ...")
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
# Indicators & signals
# ---------------------------------------------------------------------------

def compute_indicators(df, xjo_close=None):
    d = df.copy()
    d["ema50"]      = _ema(d["Close"], EMA_FAST)
    d["ema200"]     = _ema(d["Close"], EMA_SLOW)
    d["rsi"]        = _rsi(d["Close"], RSI_PERIOD)
    d["atr"]        = _atr(d["High"], d["Low"], d["Close"], ATR_PERIOD)
    d["avg_vol"]    = d["Volume"].rolling(20).mean()
    d["ema50_dist"] = (d["Close"] / d["ema50"] - 1) * 100
    d["atr_pct"]    = d["atr"] / d["Close"] * 100
    d["stock_r63"]  = d["Close"].pct_change(RS_PERIOD)
    if xjo_close is not None:
        xjo_r63        = xjo_close.pct_change(RS_PERIOD)
        d["xjo_r63"]   = xjo_r63.reindex(d.index, method="ffill")
        d["rs_vs_xjo"] = d["stock_r63"] - d["xjo_r63"]
    else:
        d["rs_vs_xjo"] = np.nan
    return d


def find_signals(d):
    """Boolean Series — True on bars where all hard filters pass."""
    return (
        (d["Close"]      >  MIN_PRICE)         &
        (d["avg_vol"]    >  MIN_AVG_VOL_20)    &
        (d["Close"]      >  d["ema200"])        &
        (d["Close"]      >  d["ema50"])         &
        (d["ema50_dist"] <= MAX_EMA50_DIST_PCT) &
        (d["rsi"]        >= RSI_MIN)            &
        (d["rsi"]        <= RSI_MAX)            &
        (d["Close"]      <= MAX_PRICE)          &
        (d["atr_pct"]    >= MIN_ATR_PCT)        &
        (d["rs_vs_xjo"]  >  0)
    )


# ---------------------------------------------------------------------------
# Portfolio simulator
# ---------------------------------------------------------------------------

def run_portfolio(price_data, xjo_close):
    """
    Chronological portfolio simulation.

    Returns
    -------
    trades_df : DataFrame — one row per completed trade
    equity_df : DataFrame — daily account snapshot (every trading day)
    """

    # 1. Compute indicators for every ticker
    print("  Computing indicators ...")
    ind = {}
    for ticker, raw_df in price_data.items():
        ind[ticker] = compute_indicators(raw_df, xjo_close)

    # 2. Pre-scan all potential entry signals
    #    Format: entry_date -> list of candidate dicts
    print("  Scanning signals ...")
    by_entry: dict = defaultdict(list)

    for ticker, d in ind.items():
        sigs  = find_signals(d)
        idx   = d.index
        opens = d["Open"].values
        atrs  = d["atr"].values
        n     = len(d)

        for i in range(WARMUP, n - 1):
            if not sigs.iloc[i]:
                continue
            entry_open = opens[i + 1]
            atr_val    = atrs[i]
            if entry_open <= 0 or atr_val <= 0:
                continue
            by_entry[idx[i + 1]].append({
                "ticker":     ticker,
                "entry_open": float(entry_open),
                "atr":        float(atr_val),
                "signal_i":   i,
            })

    total_sigs = sum(len(v) for v in by_entry.values())
    print(f"  {total_sigs} potential signals across {len(ind)} tickers.")

    # 3. Build chronological date list and fast date->index lookup
    all_dates  = sorted(set().union(*[set(d.index) for d in ind.values()]))
    date_to_i  = {dt: i for i, dt in enumerate(all_dates)}

    # 4. Portfolio state
    account    = STARTING_BALANCE
    open_pos   = {}          # ticker -> position dict
    cooldown   = {}          # ticker -> date_index when cooldown expires
    all_trades = []
    eq_log     = []          # daily equity snapshots

    # 5. Main simulation loop
    for dt in all_dates:
        dt_i = date_to_i[dt]

        # ── A. Exit check: evaluate every open position at today's close ──────
        to_close = []
        for ticker, pos in open_pos.items():
            d = ind[ticker]
            if dt not in d.index:
                # Market closed for this ticker today — keep holding
                pos["bars_held"] += 1
                continue

            c = float(d["Close"].loc[dt])

            # Breakeven trigger
            if not pos["be_set"] and c >= pos["be_lvl"]:
                pos["stop"]   = pos["entry"]
                pos["be_set"] = True

            # Exit conditions (evaluated on close)
            exit_type = None
            if c <= pos["stop"]:
                exit_type = "stop"
            elif c >= pos["target"]:
                exit_type = "target"
            elif pos["bars_held"] >= TIME_STOP_DAYS:
                exit_type = "time"

            if exit_type:
                pnl     = (c - pos["entry"]) * pos["shares"] - COMMISSION
                pnl_r   = (c - pos["entry"]) / pos["initial_risk"]
                account += pnl
                all_trades.append({
                    "ticker":        ticker,
                    "entry_date":    pos["entry_date"],
                    "exit_date":     dt,
                    "entry_price":   round(pos["entry"], 4),
                    "exit_price":    round(c, 4),
                    "stop":          round(pos["stop"], 4),
                    "target":        round(pos["target"], 4),
                    "initial_risk":  round(pos["initial_risk"], 4),
                    "shares":        pos["shares"],
                    "risk_aud":      round(pos["risk_aud"], 2),
                    "holding_days":  pos["bars_held"],
                    "exit_type":     exit_type,
                    "breakeven_hit": pos["be_set"],
                    "pnl_r":         round(pnl_r, 3),
                    "pnl_aud":       round(pnl, 2),
                    "account_after": round(account, 2),
                })
                to_close.append(ticker)
                cooldown[ticker] = dt_i + 3   # 3-bar cooldown before re-entry

            else:
                pos["bars_held"] += 1

        for ticker in to_close:
            del open_pos[ticker]

        # ── B. Daily equity snapshot (mark-to-market) ─────────────────────────
        unreal = sum(
            (float(ind[t]["Close"].loc[dt]) - pos["entry"]) * pos["shares"]
            for t, pos in open_pos.items()
            if dt in ind[t].index
        )
        eq_log.append({
            "date":          dt,
            "account":       round(account, 2),
            "n_open":        len(open_pos),
            "mark_to_market":round(account + unreal, 2),
        })

        # ── C. Entry: accept new signals if portfolio has room ─────────────────
        if dt in by_entry and len(open_pos) < MAX_POSITIONS:
            for cand in by_entry[dt]:
                if len(open_pos) >= MAX_POSITIONS:
                    break

                ticker = cand["ticker"]
                ep     = cand["entry_open"]
                atr    = cand["atr"]

                # Skip: already in position, in cooldown, or bad data
                if ticker in open_pos:
                    continue
                if dt_i < cooldown.get(ticker, 0):
                    continue
                if ep <= 0 or atr <= 0:
                    continue

                initial_risk = atr * ATR_STOP_MULTIPLE
                risk_aud     = account * RISK_PCT
                shares       = int(risk_aud / initial_risk)

                if shares == 0:
                    continue

                open_pos[ticker] = {
                    "entry_date":   dt,
                    "entry":        ep,
                    "shares":       shares,
                    "initial_risk": initial_risk,
                    "risk_aud":     risk_aud,
                    "stop":         ep - initial_risk,
                    "target":       ep + TARGET_R * initial_risk,
                    "be_lvl":       ep + BREAKEVEN_R * initial_risk,
                    "be_set":       False,
                    "bars_held":    0,
                }

    # 6. Force-close any positions remaining at end of data
    for ticker, pos in open_pos.items():
        d = ind[ticker]
        last_c  = float(d["Close"].iloc[-1])
        last_dt = d.index[-1]
        pnl     = (last_c - pos["entry"]) * pos["shares"] - COMMISSION
        pnl_r   = (last_c - pos["entry"]) / pos["initial_risk"]
        account += pnl
        all_trades.append({
            "ticker":        ticker,
            "entry_date":    pos["entry_date"],
            "exit_date":     last_dt,
            "entry_price":   round(pos["entry"], 4),
            "exit_price":    round(last_c, 4),
            "stop":          round(pos["stop"], 4),
            "target":        round(pos["target"], 4),
            "initial_risk":  round(pos["initial_risk"], 4),
            "shares":        pos["shares"],
            "risk_aud":      round(pos["risk_aud"], 2),
            "holding_days":  pos["bars_held"],
            "exit_type":     "end_of_data",
            "breakeven_hit": pos["be_set"],
            "pnl_r":         round(pnl_r, 3),
            "pnl_aud":       round(pnl, 2),
            "account_after": round(account, 2),
        })

    # 7. Assemble output DataFrames
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

def calc_metrics(trades_df, equity_df):
    if trades_df.empty:
        return {}

    # Primary stats exclude end-of-data forced closes
    closed = trades_df[trades_df["exit_type"] != "end_of_data"].copy()
    if closed.empty:
        closed = trades_df.copy()

    n         = len(closed)
    wins      = closed[closed["exit_type"] == "target"]
    stops     = closed[closed["exit_type"] == "stop"]
    timeouts  = closed[closed["exit_type"] == "time"]
    win_rate  = len(wins) / n * 100 if n else 0

    gross_win  = wins["pnl_aud"].sum()
    gross_loss = stops["pnl_aud"].sum()
    pf         = gross_win / abs(gross_loss) if gross_loss < 0 else np.nan

    avg_r      = closed["pnl_r"].mean()
    expectancy = closed["pnl_aud"].mean()
    avg_hold   = closed["holding_days"].mean()
    total_pnl  = trades_df["pnl_aud"].sum()

    # Final account value
    final_account = (trades_df["account_after"].iloc[-1]
                     if "account_after" in trades_df.columns
                     else STARTING_BALANCE + total_pnl)
    pct_return = (final_account / STARTING_BALANCE - 1) * 100

    # Max drawdown from equity curve (mark-to-market)
    if not equity_df.empty:
        mtm   = equity_df["mark_to_market"].values
        peak  = np.maximum.accumulate(mtm)
        dd    = mtm - peak
        max_dd_abs = dd.min()
        max_dd_pct = (dd / np.where(peak > 0, peak, 1) * 100).min()
    else:
        cum  = STARTING_BALANCE + closed.sort_values("exit_date")["pnl_aud"].cumsum()
        peak = cum.cummax()
        dd   = cum - peak
        max_dd_abs = dd.min()
        max_dd_pct = (dd / peak * 100).min()

    # Max consecutive losses
    sorted_pnls = closed.sort_values("exit_date")["pnl_aud"].values
    max_consec = cur = 0
    for pnl in sorted_pnls:
        cur = cur + 1 if pnl < 0 else 0
        max_consec = max(max_consec, cur)

    # Sharpe (trade-level, annualised)
    pnl_r_std = closed["pnl_r"].std()
    sharpe = ((avg_r / pnl_r_std) * np.sqrt(252 / avg_hold)
              if pnl_r_std > 0 and avg_hold > 0 else np.nan)

    # Year-by-year breakdown using equity curve for account snapshots
    yearly = {}
    closed2 = trades_df.copy()
    closed2["year"] = pd.to_datetime(closed2["exit_date"]).dt.year

    # Year-end account values from equity curve
    year_end_acct = {}
    if not equity_df.empty:
        eq = equity_df.copy()
        eq["year"] = pd.to_datetime(eq["date"]).dt.year
        for yr, grp in eq.groupby("year"):
            year_end_acct[yr] = grp["mark_to_market"].iloc[-1]

    prev_acct = STARTING_BALANCE
    for yr in sorted(closed2["year"].unique()):
        grp = closed2[closed2["year"] == yr]
        n_y   = len(grp[grp["exit_type"] != "end_of_data"])
        wins_y = (grp["exit_type"] == "target").sum()
        wr_y   = wins_y / n_y * 100 if n_y else 0
        pnl_y  = grp["pnl_aud"].sum()
        gw     = grp[grp["pnl_aud"] > 0]["pnl_aud"].sum()
        gl     = abs(grp[grp["pnl_aud"] < 0]["pnl_aud"].sum())
        pf_y   = round(gw / gl, 2) if gl > 0 else "n/a"
        acct_y = year_end_acct.get(yr, prev_acct + pnl_y)
        ret_y  = (acct_y / prev_acct - 1) * 100 if prev_acct > 0 else 0
        yearly[yr] = {
            "trades":    n_y,
            "win_pct":   round(wr_y, 1),
            "pnl_aud":   round(pnl_y, 2),
            "pf":        pf_y,
            "acct_end":  round(acct_y, 2),
            "return_pct":round(ret_y, 1),
        }
        prev_acct = acct_y

    return {
        "total_trades":      n,
        "win_rate_pct":      round(win_rate, 1),
        "wins":              len(wins),
        "losses":            len(stops),
        "time_stops":        len(timeouts),
        "profit_factor":     round(pf, 2) if not np.isnan(pf) else "n/a",
        "avg_r":             round(avg_r, 3),
        "expectancy_aud":    round(expectancy, 2),
        "avg_hold_days":     round(avg_hold, 1),
        "total_pnl_aud":     round(total_pnl, 2),
        "final_account":     round(final_account, 2),
        "pct_return":        round(pct_return, 1),
        "max_drawdown_aud":  round(max_dd_abs, 2),
        "max_drawdown_pct":  round(max_dd_pct, 1),
        "max_consec_losses": max_consec,
        "sharpe":            round(sharpe, 2) if not np.isnan(sharpe) else "n/a",
        "best_trade_aud":    round(closed["pnl_aud"].max(), 2),
        "worst_trade_aud":   round(closed["pnl_aud"].min(), 2),
        "yearly":            yearly,
    }


# ---------------------------------------------------------------------------
# Console output
# ---------------------------------------------------------------------------

def print_results(metrics):
    m = metrics
    W = 62

    print(f"\n{'='*W}")
    print(f"  5-YEAR PORTFOLIO BACKTEST RESULTS")
    print(f"  ${STARTING_BALANCE:,} | {RISK_PCT*100:.1f}% risk/trade | max {MAX_POSITIONS} positions")
    print(f"{'='*W}")

    # ── Core metrics ─────────────────────────────────────────────────────────
    rows = [
        ("Starting balance",   f"${STARTING_BALANCE:,.2f}"),
        ("Final account",      f"${m.get('final_account', 0):,.2f}"),
        ("Total return",       f"{m.get('pct_return', 0):+.1f}%"),
        ("Total P&L",          f"${m.get('total_pnl_aud', 0):,.2f}"),
        ("",                   ""),
        ("Total trades",       str(m.get("total_trades", "—"))),
        ("Win rate",           f"{m.get('win_rate_pct','—')}%"),
        ("  Targets hit",      str(m.get("wins","—"))),
        ("  Stops hit",        str(m.get("losses","—"))),
        ("  Time stops",       str(m.get("time_stops","—"))),
        ("",                   ""),
        ("Profit factor",      str(m.get("profit_factor","—"))),
        ("Avg R / trade",      str(m.get("avg_r","—"))),
        ("Expectancy / trade", f"${m.get('expectancy_aud', 0):,.2f}"),
        ("Avg hold (days)",    str(m.get("avg_hold_days","—"))),
        ("",                   ""),
        ("Max drawdown $",     f"${m.get('max_drawdown_aud', 0):,.2f}"),
        ("Max drawdown %",     f"{m.get('max_drawdown_pct','—')}%"),
        ("Max consec. losses", str(m.get("max_consec_losses","—"))),
        ("Sharpe (annualised)",str(m.get("sharpe","—"))),
        ("",                   ""),
        ("Best trade",         f"${m.get('best_trade_aud', 0):,.2f}"),
        ("Worst trade",        f"${m.get('worst_trade_aud', 0):,.2f}"),
    ]
    for label, val in rows:
        if not label:
            print()
        else:
            print(f"  {label:<24} {val}")

    # ── Year-by-year breakdown ────────────────────────────────────────────────
    yearly = m.get("yearly", {})
    if yearly:
        print(f"\n  {'-'*W}")
        print(f"  {'YEAR-BY-YEAR BREAKDOWN':}")
        print(f"  {'-'*W}")
        hdr = f"  {'Year':<6} {'Trades':>7} {'Win%':>6} {'P&L':>11} {'PF':>6} {'Return':>8}  Account end"
        print(hdr)
        print(f"  {'-'*W}")
        for yr in sorted(yearly.keys()):
            y = yearly[yr]
            sign = "+" if y["pnl_aud"] >= 0 else ""
            ret_sign = "+" if y["return_pct"] >= 0 else ""
            print(
                f"  {yr:<6} {y['trades']:>7} {y['win_pct']:>5.1f}% "
                f"{sign}{y['pnl_aud']:>10,.0f} {str(y['pf']):>6} "
                f"{ret_sign}{y['return_pct']:>6.1f}%  "
                f"${y['acct_end']:>10,.0f}"
            )
        print(f"  {'-'*W}")

    print(f"\n{'='*W}\n")


# ---------------------------------------------------------------------------
# Chart
# ---------------------------------------------------------------------------

def save_chart(trades_df, equity_df, metrics, out_path):
    BG, PANEL        = "#0d1117", "#161b22"
    GREEN, RED, GREY = "#26a641", "#da3633", "#8b949e"
    AMBER            = "#e3b341"

    fig = plt.figure(figsize=(18, 14), facecolor=BG)
    gs  = gridspec.GridSpec(3, 2, figure=fig,
                            hspace=0.44, wspace=0.30,
                            height_ratios=[3, 2, 2])

    ax_eq   = fig.add_subplot(gs[0, :])   # equity curve — full width
    ax_yr   = fig.add_subplot(gs[1, 0])   # year-by-year P&L
    ax_hist = fig.add_subplot(gs[1, 1])   # R distribution
    ax_stat = fig.add_subplot(gs[2, :])   # stats table — full width

    for ax in (ax_eq, ax_yr, ax_hist, ax_stat):
        ax.set_facecolor(PANEL)
        ax.tick_params(colors=GREY, labelsize=8)
        for sp in ax.spines.values():
            sp.set_edgecolor("#21262d")

    # ── Equity curve ──────────────────────────────────────────────────────────
    if not equity_df.empty:
        eq_dates = pd.to_datetime(equity_df["date"])
        eq_vals  = equity_df["mark_to_market"].values
    else:
        st = trades_df.sort_values("exit_date")
        eq_vals  = STARTING_BALANCE + st["pnl_aud"].cumsum().values
        eq_dates = pd.to_datetime(st["exit_date"])

    xs   = np.arange(len(eq_vals))
    base = STARTING_BALANCE

    ax_eq.plot(xs, eq_vals, color=GREEN, lw=1.4, zorder=3)
    ax_eq.axhline(base, color=GREY, lw=0.8, ls="--", alpha=0.5)
    ax_eq.fill_between(xs, eq_vals, base,
                       where=(eq_vals >= base), color=GREEN, alpha=0.12)
    ax_eq.fill_between(xs, eq_vals, base,
                       where=(eq_vals < base),  color=RED,   alpha=0.15)

    peak = np.maximum.accumulate(eq_vals)
    ax_eq.fill_between(xs, eq_vals, peak, where=(eq_vals < peak),
                       color=RED, alpha=0.07, label="Drawdown")

    final = metrics.get("final_account", 0)
    ret   = metrics.get("pct_return", 0)
    ax_eq.set_title(
        f"Account Equity Curve  |  "
        f"${STARTING_BALANCE:,} -> ${final:,.0f}  ({ret:+.1f}%  over 5 years)",
        color="white", fontsize=11, pad=8)
    ax_eq.set_ylabel("AUD", color=GREY, fontsize=9)
    ax_eq.grid(color="#21262d", lw=0.5, alpha=0.6)
    ax_eq.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda v, _: f"${v:,.0f}"))

    # x-axis: sample ~8 date labels
    if len(eq_dates) > 1:
        step = max(1, len(eq_dates) // 8)
        tpos = list(range(0, len(eq_dates), step))
        tlbl = [pd.Timestamp(eq_dates.iloc[i]).strftime("%b %y")
                for i in tpos if i < len(eq_dates)]
        ax_eq.set_xticks(tpos[:len(tlbl)])
        ax_eq.set_xticklabels(tlbl, color=GREY, fontsize=8)

    # ── Year-by-year P&L bar chart ────────────────────────────────────────────
    yearly = metrics.get("yearly", {})
    if yearly:
        years  = sorted(yearly.keys())
        y_pnls = [yearly[y]["pnl_aud"] for y in years]
        y_cols = [GREEN if p >= 0 else RED for p in y_pnls]

        bars = ax_yr.bar(years, y_pnls, color=y_cols, alpha=0.85, width=0.6)
        ax_yr.axhline(0, color=GREY, lw=0.8, ls="--")
        ax_yr.set_title("Year-by-Year P&L", color="white", fontsize=10, pad=6)
        ax_yr.set_ylabel("AUD", color=GREY, fontsize=9)
        ax_yr.yaxis.set_major_formatter(
            mticker.FuncFormatter(lambda v, _: f"${v:,.0f}"))
        ax_yr.grid(color="#21262d", lw=0.5, alpha=0.6, axis="y")
        ax_yr.set_xticks(years)
        ax_yr.set_xticklabels([str(y) for y in years], color=GREY, fontsize=8)

        y_min, y_max = min(y_pnls), max(y_pnls)
        y_range = max(abs(y_min), abs(y_max)) * 0.12 or 1
        for yr, pnl in zip(years, y_pnls):
            offset = y_range if pnl >= 0 else -y_range
            va     = "bottom" if pnl >= 0 else "top"
            ax_yr.text(yr, pnl + offset, f"${pnl:,.0f}",
                       ha="center", va=va, color=GREY, fontsize=6.5)

    # ── R distribution (sorted bar) ───────────────────────────────────────────
    closed  = trades_df[trades_df["exit_type"] != "end_of_data"]
    r_vals  = sorted(closed["pnl_r"].values)
    r_cols  = [GREEN if r > 0 else RED for r in r_vals]
    ax_hist.bar(range(len(r_vals)), r_vals, color=r_cols, alpha=0.8, width=0.8)
    ax_hist.axhline(0, color=GREY, lw=0.8, ls="--")
    ax_hist.set_title("Trade Outcomes (sorted, R multiples)", color="white",
                      fontsize=10, pad=6)
    ax_hist.set_ylabel("R", color=GREY, fontsize=9)
    ax_hist.grid(color="#21262d", lw=0.5, alpha=0.6, axis="y")
    ax_hist.tick_params(bottom=False, labelbottom=False)

    # ── Stats table ───────────────────────────────────────────────────────────
    ax_stat.axis("off")
    m = metrics

    col_a = [
        ("Starting balance",   f"${STARTING_BALANCE:,}"),
        ("Final account",      f"${m.get('final_account', 0):,.0f}"),
        ("Total return",       f"{m.get('pct_return', 0):+.1f}%"),
        ("Total P&L",          f"${m.get('total_pnl_aud', 0):,.0f}"),
        ("",                   ""),
        ("Total trades",       str(m.get("total_trades","—"))),
        ("Win rate",           f"{m.get('win_rate_pct','—')}%"),
        ("Targets / Stops / Time", f"{m.get('wins','—')} / {m.get('losses','—')} / {m.get('time_stops','—')}"),
    ]
    col_b = [
        ("Profit factor",      str(m.get("profit_factor","—"))),
        ("Avg R / trade",      str(m.get("avg_r","—"))),
        ("Expectancy",         f"${m.get('expectancy_aud', 0):,.0f}"),
        ("Avg hold days",      str(m.get("avg_hold_days","—"))),
        ("",                   ""),
        ("Max drawdown $",     f"${m.get('max_drawdown_aud', 0):,.0f}"),
        ("Max drawdown %",     f"{m.get('max_drawdown_pct','—')}%"),
        ("Max consec. losses", str(m.get("max_consec_losses","—"))),
    ]
    col_c = [
        ("Sharpe",             str(m.get("sharpe","—"))),
        ("Best trade",         f"${m.get('best_trade_aud', 0):,.0f}"),
        ("Worst trade",        f"${m.get('worst_trade_aud', 0):,.0f}"),
        ("",                   ""),
        ("",                   ""),
        ("Risk / trade",       f"{RISK_PCT*100:.1f}%"),
        ("Max positions",      str(MAX_POSITIONS)),
        ("Commission",         f"${COMMISSION:.0f} / trade"),
    ]

    y = 0.92
    dy = 0.115
    for (la, va), (lb, vb), (lc, vc) in zip(col_a, col_b, col_c):
        if la:
            ax_stat.text(0.01, y, la, transform=ax_stat.transAxes,
                         color=GREY, fontsize=8.5, va="top")
            ax_stat.text(0.23, y, va, transform=ax_stat.transAxes,
                         color="white", fontsize=8.5, va="top", fontweight="bold")
        if lb:
            ax_stat.text(0.37, y, lb, transform=ax_stat.transAxes,
                         color=GREY, fontsize=8.5, va="top")
            ax_stat.text(0.59, y, vb, transform=ax_stat.transAxes,
                         color="white", fontsize=8.5, va="top", fontweight="bold")
        if lc:
            ax_stat.text(0.69, y, lc, transform=ax_stat.transAxes,
                         color=GREY, fontsize=8.5, va="top")
            ax_stat.text(0.91, y, vc, transform=ax_stat.transAxes,
                         color="white", fontsize=8.5, va="top", fontweight="bold")
        y -= dy

    ax_stat.set_title("Summary Statistics", color="white", fontsize=10, pad=6)

    fig.text(0.5, 0.002,
             f"Stop: {ATR_STOP_MULTIPLE}x ATR  |  Target: {TARGET_R}:1  |  "
             f"Breakeven @ 1R  |  Time stop: {TIME_STOP_DAYS}d  |  "
             f"Filters: ATR>={MIN_ATR_PCT}%  price<=${MAX_PRICE:.0f}  RS>XJO>0",
             color=GREY, fontsize=7.5, ha="center")

    plt.savefig(out_path, dpi=130, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"  Chart -> {out_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="ASX Swing Engine — portfolio backtest")
    p.add_argument("--period",        default="5y",
                   help="yfinance period string (default: 5y)")
    p.add_argument("--from-screener", action="store_true",
                   help="Use last screener_output.csv as universe")
    p.add_argument("--top-n",         type=int, default=None,
                   help="Top N from screener (requires --from-screener)")
    return p.parse_args()


def main():
    args = parse_args()

    if args.from_screener:
        csv = Path("results/screener_output.csv")
        if not csv.exists():
            print("ERROR: screener_output.csv not found. Run screener.py first.")
            sys.exit(1)
        sdf     = pd.read_csv(csv, index_col="rank")
        tickers = sdf["ticker"].tolist()
        if args.top_n:
            tickers = tickers[:args.top_n]
        print(f"Universe : {len(tickers)} tickers from screener output")
    else:
        tickers = list(dict.fromkeys(ASX_PRESET))
        print(f"Universe : {len(tickers)} preset ASX tickers")

    print(f"Period   : {args.period}")
    print(f"Account  : ${STARTING_BALANCE:,}  |  "
          f"Risk: {RISK_PCT*100:.1f}%/trade  |  Max positions: {MAX_POSITIONS}")
    print(f"Filters  : ATR>={MIN_ATR_PCT}%  price<=${MAX_PRICE:.0f}  RS>XJO>0  (+ all base filters)")
    print(f"Exit     : stop={ATR_STOP_MULTIPLE}xATR  "
          f"breakeven@1R  target={TARGET_R}:1  time_stop={TIME_STOP_DAYS}d\n")

    # Benchmark
    print(f"Downloading benchmark ({BENCHMARK}) ...")
    with _quiet():
        xjo_raw = yf.download(BENCHMARK, period=args.period, interval="1d",
                              auto_adjust=True, progress=False)
    _xjo      = xjo_raw["Close"]
    xjo_close = (_xjo.iloc[:, 0] if isinstance(_xjo, pd.DataFrame) else _xjo).dropna()
    print(f"  {len(xjo_close)} benchmark bars.\n")

    # Price history
    print("Downloading price history ...")
    price_data = download_universe(tickers, args.period)
    print(f"  Data for {len(price_data)} / {len(tickers)} tickers.\n")

    if not price_data:
        print("No data downloaded. Check connection.")
        sys.exit(1)

    # Portfolio simulation
    print("Running portfolio simulation ...")
    trades_df, equity_df = run_portfolio(price_data, xjo_close)

    if trades_df.empty:
        print("No trades generated.")
        sys.exit(0)

    metrics = calc_metrics(trades_df, equity_df)
    print_results(metrics)

    # Save outputs
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    today      = date.today().strftime("%Y%m%d")
    csv_path   = RESULTS_DIR / f"portfolio_trades_{today}.csv"
    eq_path    = RESULTS_DIR / f"equity_curve_{today}.csv"
    chart_path = RESULTS_DIR / f"portfolio_backtest_{today}.png"

    trades_df.to_csv(csv_path, index=False)
    equity_df.to_csv(eq_path,  index=False)
    save_chart(trades_df, equity_df, metrics, chart_path)

    print(f"\n  Trade log  -> {csv_path}")
    print(f"  Equity CSV -> {eq_path}")


if __name__ == "__main__":
    main()
