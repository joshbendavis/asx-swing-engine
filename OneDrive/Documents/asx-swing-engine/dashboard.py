"""
dashboard.py  --  ASX Swing Engine — Professional Monitoring Dashboard
----------------------------------------------------------------------
Run with:  streamlit run dashboard.py

Layout
  Row 1  — Portfolio Health          (equity curve, drawdown, exposure/heat)
  Row 2  — Edge Validation           (rolling 30-trade metrics, slippage, fill rate)
  Panel  — Missed Trades Tracker     (signals → eligible → executed → missed + hypothetical P&L)
  Row 3  — System Diagnostics        (regime matrix, gap stops, trade lifecycle, losing streak)
  Panel  — Heat Over Time            (daily portfolio heat % line chart)
  Panel  — Trade Quality Buckets     (C-filter blocked vs F-filter allowed P&L comparison)
  Panel  — Rolling Expectancy        (rolling 10/30/50 avg R chart)
  Panel  — Backtest vs Live          (side-by-side table, amber >15% drift, red >30%)
  Bottom — Full trade log

Auto-refreshes every 5 minutes.
"""

import json
import math
import re
import time
from datetime import datetime, date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
try:
    import yfinance as yf
    _YF_AVAILABLE = True
except ImportError:
    _YF_AVAILABLE = False

# ---------------------------------------------------------------------------
# Page config  (must be first Streamlit call)
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="ASX Swing Engine",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ---------------------------------------------------------------------------
# Mobile detection — inject JS that sets ?view=mobile on narrow screens.
# On re-render Streamlit reads the query param and routes to mobile view.
# ---------------------------------------------------------------------------
st.markdown("""
<script>
(function() {
    try {
        var w = window.innerWidth || document.documentElement.clientWidth || 768;
        var url = new URL(window.location.href);
        var cur = url.searchParams.get('view');
        if (w < 768 && cur !== 'mobile') {
            url.searchParams.set('view', 'mobile');
            window.location.replace(url.toString());
        } else if (w >= 768 && cur === 'mobile') {
            url.searchParams.delete('view');
            window.location.replace(url.toString());
        }
    } catch(e) {}
})();
</script>
""", unsafe_allow_html=True)

_is_mobile = st.query_params.get("view", "") == "mobile"

# ---------------------------------------------------------------------------
# Dark theme CSS
# ---------------------------------------------------------------------------
st.markdown("""
<style>
  /* global */
  .stApp { background-color: #0d0d0d; color: #e0e0e0; }
  section[data-testid="stSidebar"] { background-color: #111111; }

  /* metric tiles */
  [data-testid="stMetric"] {
      background: #161616;
      border: 1px solid #2a2a2a;
      border-radius: 8px;
      padding: 14px 18px;
  }
  [data-testid="stMetricLabel"] { color: #888 !important; font-size: 12px !important; }
  [data-testid="stMetricValue"] { color: #e0e0e0 !important; font-size: 22px !important; }
  [data-testid="stMetricDelta"] { font-size: 13px !important; }

  /* dataframe */
  [data-testid="stDataFrame"] { border: 1px solid #2a2a2a; border-radius: 6px; }

  /* section headers */
  h2 { color: #f5a623 !important; font-size: 16px !important;
       border-bottom: 1px solid #2a2a2a; padding-bottom: 6px; margin-top: 28px !important; }
  h3 { color: #aaaaaa !important; font-size: 13px !important; }

  /* pills */
  .pill { display:inline-block; padding:2px 8px; border-radius:10px;
          font-size:11px; font-weight:600; }
  .pill-green { background:#0d2b1e; color:#26a69a; border:1px solid #26a69a44; }
  .pill-red   { background:#2b0d0d; color:#ef5350; border:#ef535044 1px solid; }
  .pill-amber { background:#2b1e0d; color:#f5a623; border:#f5a62344 1px solid; }
  .pill-blue  { background:#0d1a2b; color:#58a6ff; border:#58a6ff44 1px solid; }

  /* data collection phase note */
  .phase-note {
      color: #888; font-size: 11px; font-style: italic;
      border-left: 2px solid #333; padding-left: 8px; margin-top: 6px;
  }

  /* kill conditions banner */
  .kill-banner {
      background: #1a0505;
      border: 2px solid #ef5350;
      border-radius: 10px;
      padding: 16px 22px;
      margin-bottom: 16px;
  }
  .kill-banner-title {
      color: #ef5350;
      font-size: 16px;
      font-weight: 700;
      letter-spacing: 0.08em;
      margin-bottom: 10px;
  }
  .kill-item {
      color: #ffcdd2;
      font-size: 13px;
      padding: 3px 0;
      border-left: 3px solid #ef5350;
      padding-left: 10px;
      margin-bottom: 5px;
  }
  .kill-clear {
      background: #051a0a;
      border: 2px solid #26a69a;
      border-radius: 10px;
      padding: 12px 22px;
      margin-bottom: 16px;
      color: #26a69a;
      font-size: 14px;
      font-weight: 600;
  }

  /* low-confidence overlay label */
  .low-conf-note {
      color: #f5a623; font-size: 11px; font-style: italic;
      opacity: 0.8; margin-top: 2px;
  }

  /* divider */
  hr { border-color: #2a2a2a; }

  /* hide streamlit chrome */
  #MainMenu, footer, header { visibility: hidden; }

  /* ── Mobile-first responsive ───────────────────────────────── */
  @media (max-width: 768px) {
    /* tighter page padding on mobile */
    .block-container { padding: 0.5rem !important; }

    /* columns: allow wrapping, minimum 45% so 2-per-row */
    div[data-testid="column"] { min-width: 45% !important; flex: 1 1 45% !important; }

    /* larger base text for readability */
    .stApp, p, div, span, li { font-size: 16px !important; }

    /* section headers bigger on mobile */
    h2 { font-size: 18px !important; }

    /* metric tiles: larger numbers on small screen */
    [data-testid="stMetricValue"] { font-size: 26px !important; }
    [data-testid="stMetricLabel"] { font-size: 13px !important; }

    /* snap bar: stack cells vertically on mobile */
    .snap-bar { flex-direction: column !important; }
    .snap-cell {
      border-right: none !important;
      border-bottom: 1px solid #1c1c1c !important;
      padding: 16px 18px !important;
    }
    .snap-cell:last-child { border-bottom: none !important; }
    .snap-value { font-size: 28px !important; }
    .snap-label { font-size: 11px !important; }
    .regime-badge { font-size: 18px !important; padding: 8px 18px !important; }

    /* kill conditions: full width, large text, impossible to miss */
    .kill-banner { padding: 20px 16px !important; }
    .kill-banner-title { font-size: 20px !important; }
    .kill-item { font-size: 16px !important; }
    .snap-kill-pill { font-size: 18px !important; padding: 10px 16px !important; }

    /* charts: ensure full width, no overflow */
    [data-testid="stPlotlyChart"] { width: 100% !important; }

    /* dataframes: horizontal scroll on mobile */
    [data-testid="stDataFrame"] {
      overflow-x: auto !important;
      -webkit-overflow-scrolling: touch;
    }

    /* cooldown / soft breach alerts: larger on mobile */
    .mobile-alert { font-size: 16px !important; padding: 16px !important; }
  }

  @media (min-width: 769px) {
    /* desktop: columns don't wrap */
    div[data-testid="column"] { min-width: 0 !important; }
  }

  /* ── Mobile-only simplified view ──────────────────────────── */
  .mobile-pnl {
    font-size: 52px;
    font-weight: bold;
    text-align: center;
    padding: 24px 16px 8px 16px;
    line-height: 1.1;
    letter-spacing: -0.02em;
  }
  .mobile-pnl-pct {
    font-size: 22px;
    font-weight: 600;
    text-align: center;
    padding-bottom: 24px;
    opacity: 0.85;
  }
  .mobile-ticker {
    font-size: 28px;
    font-weight: bold;
    line-height: 1.1;
  }
  .mobile-trade-pnl {
    font-size: 22px;
    font-weight: 600;
  }
  .mobile-trade-row {
    padding: 16px 12px;
    border-bottom: 1px solid #222;
    display: flex;
    justify-content: space-between;
    align-items: center;
  }
  .mob-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 12px 14px;
    background: #111;
    border-bottom: 1px solid #222;
    margin-bottom: 4px;
  }
  .mob-run-dot {
    font-size: 13px;
    font-weight: 600;
    letter-spacing: 0.04em;
  }
  .mob-section-title {
    color: #555;
    font-size: 11px;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    padding: 16px 12px 6px 12px;
    font-weight: 600;
  }

  /* ── Snapshot bar ─────────────────────────────────────── */
  .snap-bar {
    display: flex;
    background: #111111;
    border: 1px solid #252525;
    border-radius: 10px;
    margin-bottom: 18px;
    overflow: hidden;
  }
  .snap-cell {
    flex: 1;
    padding: 14px 22px;
    border-right: 1px solid #1c1c1c;
    min-width: 0;
  }
  .snap-cell:last-child { border-right: none; flex: 1.6; }
  .snap-label {
    color: #555;
    font-size: 9px;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    margin-bottom: 6px;
    font-weight: 600;
  }
  .snap-value {
    font-size: 24px;
    font-weight: 700;
    line-height: 1.05;
    letter-spacing: -0.02em;
    white-space: nowrap;
  }
  .snap-sub {
    font-size: 11px;
    color: #555;
    margin-top: 4px;
  }
  .regime-badge {
    display: inline-block;
    padding: 5px 14px;
    border-radius: 6px;
    font-size: 14px;
    font-weight: 800;
    letter-spacing: 0.06em;
    margin-top: 1px;
  }
  /* Kill conditions details/summary */
  .snap-cell details { width: 100%; }
  .snap-cell details summary {
    cursor: pointer;
    list-style: none;
    outline: none;
    -webkit-user-select: none;
    user-select: none;
  }
  .snap-cell details summary::-webkit-details-marker { display: none; }
  .snap-kill-pill {
    display: block;
    padding: 7px 12px;
    border-radius: 6px;
    font-size: 15px;
    font-weight: 700;
    text-align: center;
    letter-spacing: 0.02em;
  }
  .snap-kill-detail {
    margin-top: 8px;
    background: #0a0a0a;
    border: 1px solid #252525;
    border-radius: 6px;
    padding: 10px 14px;
    font-size: 11px;
    line-height: 1.6;
    position: relative;
    z-index: 10;
  }
  .snap-kill-item {
    color: #ffcdd2;
    padding: 3px 0 3px 8px;
    border-left: 2px solid #ef5350;
    margin-bottom: 5px;
  }
  .snap-kill-clear { color: #26a69a; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
TRADES_CSV       = Path("logs/trades.csv")
PENDING_CSV      = Path("logs/pending_orders.csv")
EQUITY_CSV       = Path("results/equity_curve.csv")
REGIME_JSON      = Path("results/regime.json")
SIGNALS_CSV      = Path("results/signals_output.csv")
COOLDOWN_JSON    = Path("results/kill_cooldown.json")
SOFT_BREACH_JSON = Path("results/soft_breach.json")

STALE_ORDER_DAYS = 3   # pending orders older than this are flagged

REFRESH_SECS     = 300
STARTING_BALANCE = 20_000.0
MAX_POSITIONS    = 6
HEAT_BUDGET      = 9.0          # %
AVG_RISK_PER_TRADE = 300.0      # ~1.5% of $20k, used for hypothetical miss P&L

GREEN = "#26a69a"
RED   = "#ef5350"
AMBER = "#f5a623"
GREY  = "#888888"
BLUE  = "#58a6ff"
PURPLE = "#b39ddb"

PLOT_LAYOUT = dict(
    paper_bgcolor="#161616",
    plot_bgcolor="#161616",
    font=dict(color="#e0e0e0", size=11),
    margin=dict(l=45, r=15, t=30, b=35),
    xaxis=dict(gridcolor="#2a2a2a", linecolor="#2a2a2a", zeroline=False),
    yaxis=dict(gridcolor="#2a2a2a", linecolor="#2a2a2a", zeroline=False),
)

# Backtest benchmark constants — Variant F 8-year validation (2017–2026, 102 trades)
BT = {
    "win_rate":        15.7,    # %
    "avg_r":           0.293,
    "profit_factor":   0.91,
    "gap_stop_pct":    16.7,    # gap stops as % of all exits  (17/102)
    "avg_hold_days":   14.8,
    "max_consec_loss": 4,
    "sharpe":          1.06,
    "cagr":            4.5,
    "max_dd":          9.0,     # % (positive number)
    "expectancy":      89.0,    # AUD / trade
    # Regime breakdown from 8-year backtest
    "regime_trades":   {"STRONG_BULL": 55, "WEAK_BULL": 12, "CHOPPY_BEAR": 35},
    "regime_win_rate": {"STRONG_BULL": 18.2, "WEAK_BULL": 16.7, "CHOPPY_BEAR": 11.4},
    "regime_avg_r":    {"STRONG_BULL": 0.42, "WEAK_BULL": 0.31, "CHOPPY_BEAR": 0.09},
}
DRIFT_WARN      = 0.15   # amber at 15% drift
DRIFT_THRESHOLD = 0.30   # red at 30% drift
DATA_COLLECTION_TRADES = 30   # trades before stats are meaningful

# Kill-condition thresholds
KILL_EXPECTANCY_MIN         = 0.0    # rolling-30 avg R must be > 0
KILL_PF_MIN                 = 0.7    # rolling-30 profit factor floor
KILL_GAP_STOP_MAX           = BT["gap_stop_pct"] * 2   # ~33.4% — double backtest rate
KILL_SLIPPAGE_MAX           = 0.40   # % — live entry slippage ceiling
KILL_REGIME_INVERSION       = True   # flag: CHOPPY_BEAR avg R > STRONG_BULL avg R
KILL_REGIME_MIN_TRADES      = 10     # minimum trades per regime before inversion can fire
KILL_COOLDOWN_DAYS          = 5      # trading-day pause after any kill condition fires
SOFT_BREACH_WARN_DAYS       = 3      # consecutive amber days before manual review warning


# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------

@st.cache_data(ttl=REFRESH_SECS)
def _load_regime() -> dict:
    default = {
        "regime": "BULL",
        "f_regime": "STRONG_BULL",
        "f_regime_size_multiplier": 1.0,
        "position_size_multiplier": 1.0,
    }
    if not REGIME_JSON.exists():
        return default
    try:
        with open(REGIME_JSON) as f:
            data = json.load(f)
        if "f_regime" not in data:
            data["f_regime"] = "STRONG_BULL"
        if "f_regime_size_multiplier" not in data:
            data["f_regime_size_multiplier"] = 1.0
        return data
    except Exception:
        return default


@st.cache_data(ttl=REFRESH_SECS)
def _load_trades() -> pd.DataFrame:
    """Entry records from logs/trades.csv — deduped on order_ref."""
    if not TRADES_CSV.exists():
        return pd.DataFrame()
    try:
        df = pd.read_csv(TRADES_CSV, parse_dates=["timestamp"])
        if "order_ref" in df.columns:
            df = df.drop_duplicates(subset="order_ref", keep="last")
        return df
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=REFRESH_SECS)
def _load_equity() -> pd.DataFrame:
    """Closed-trade P&L from results/equity_curve.csv."""
    if not EQUITY_CSV.exists():
        return pd.DataFrame()
    try:
        df = pd.read_csv(EQUITY_CSV, parse_dates=["date"])
        df.sort_values("date", inplace=True)
        df.reset_index(drop=True, inplace=True)
        return df
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=REFRESH_SECS)
def _load_pending() -> pd.DataFrame:
    """Load pending_orders.csv — submitted brackets not yet filled."""
    if not PENDING_CSV.exists():
        return pd.DataFrame()
    try:
        df = pd.read_csv(PENDING_CSV, parse_dates=["timestamp"])
        if df.empty:
            return df
        # Exclude any order_refs already promoted to trades.csv
        if TRADES_CSV.exists():
            try:
                trades = pd.read_csv(TRADES_CSV)
                if not trades.empty:
                    filled = set(trades["order_ref"].dropna())
                    df = df[~df["order_ref"].isin(filled)]
            except Exception:
                pass
        return df.drop_duplicates(subset="order_ref", keep="last").reset_index(drop=True)
    except Exception:
        return pd.DataFrame()


def _stale_pending(pending_df: pd.DataFrame) -> pd.DataFrame:
    """
    Return rows from pending_df where the order has been waiting
    >= STALE_ORDER_DAYS trading days without filling.
    """
    if pending_df.empty or "timestamp" not in pending_df.columns:
        return pd.DataFrame()
    today = date.today()
    rows  = []
    for _, row in pending_df.iterrows():
        try:
            sub_date = pd.to_datetime(row["timestamp"]).date()
            bdays    = int(np.busday_count(sub_date.isoformat(), today.isoformat()))
            if bdays >= STALE_ORDER_DAYS:
                rows.append({**row.to_dict(), "_waiting_days": bdays})
        except Exception:
            pass
    return pd.DataFrame(rows) if rows else pd.DataFrame()


@st.cache_data(ttl=REFRESH_SECS)
def _load_signals() -> pd.DataFrame:
    """Latest signal scan output from results/signals_output.csv."""
    if not SIGNALS_CSV.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(SIGNALS_CSV)
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=60)
def _fetch_live_prices(tickers: tuple) -> dict:
    """Latest prices for open positions — yfinance, 60-second cache."""
    if not tickers or not _YF_AVAILABLE:
        return {}
    try:
        data = yf.download(list(tickers), period="1d", progress=False, auto_adjust=True)
        if data.empty:
            return {}
        close = data["Close"] if "Close" in data.columns else data
        if hasattr(close, "columns"):
            out = {}
            for t in tickers:
                if t in close.columns:
                    s = close[t].dropna()
                    if not s.empty:
                        out[t] = float(s.iloc[-1])
            return out
        elif len(tickers) == 1:
            s = close.dropna()
            if len(s) > 0:
                return {tickers[0]: float(s.iloc[-1])}
        return {}
    except Exception:
        return {}


def _calc_live_pnl(
    open_df: pd.DataFrame,
    equity_df: pd.DataFrame,
    live_prices: dict,
) -> tuple:
    """
    Total P&L = realised (closed trades) + unrealised (open positions at live prices).
    Returns (total_pnl_aud, total_pnl_pct).
    """
    realised = float(equity_df["pnl_aud"].sum()) if (
        not equity_df.empty and "pnl_aud" in equity_df.columns) else 0.0

    unrealised = 0.0
    if not open_df.empty:
        for _, row in open_df.iterrows():
            ticker = str(row.get("ticker", ""))
            price  = live_prices.get(ticker, 0.0)
            entry  = float(row.get("entry", 0.0))
            shares = int(row.get("shares", 0))
            if price > 0 and shares > 0:
                unrealised += (price - entry) * shares

    total_pnl = round(realised + unrealised, 2)
    total_pct = round(total_pnl / STARTING_BALANCE * 100, 2) if STARTING_BALANCE > 0 else 0.0
    return total_pnl, total_pct


@st.cache_data(ttl=REFRESH_SECS)
def _load_backtest_equity() -> pd.Series:
    """Variant F backtest equity curve (mark_to_market column)."""
    bt_dir = Path("results/backtest")
    if not bt_dir.exists():
        return pd.Series(dtype=float)
    files = sorted(bt_dir.glob("equity_8y_f_*.csv"))
    if not files:
        return pd.Series(dtype=float)
    try:
        eq = pd.read_csv(files[-1], index_col=0, parse_dates=True)
        col = "mark_to_market" if "mark_to_market" in eq.columns else eq.columns[0]
        return eq[col].dropna()
    except Exception:
        return pd.Series(dtype=float)


@st.cache_data(ttl=REFRESH_SECS)
def _load_backtest_trades() -> pd.DataFrame:
    """Variant F backtest trades CSV."""
    bt_dir = Path("results/backtest")
    if not bt_dir.exists():
        return pd.DataFrame()
    files = sorted(bt_dir.glob("trades_8y_f_regime_conditional_*.csv"))
    if not files:
        return pd.DataFrame()
    try:
        return pd.read_csv(files[-1], parse_dates=["entry_date", "exit_date"])
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=REFRESH_SECS)
def _parse_last_run() -> dict:
    log_dir = Path("logs")
    if not log_dir.exists():
        return {}
    run_logs = sorted(log_dir.glob("run_*.log"), reverse=True)
    if not run_logs:
        return {}
    log_path = run_logs[0]
    fname_m = re.match(r"run_(\d{4})(\d{2})(\d{2})_(\d{2})(\d{2})\d{2}\.log", log_path.name)
    if not fname_m:
        return {}
    y, mo, d, h, mi = fname_m.groups()
    run_date = date(int(y), int(mo), int(d))
    run_time = f"{h}:{mi}"
    try:
        content = log_path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return {}
    steps = {k: None for k in ["regime", "screener", "signals", "risk", "ibkr", "exit_logger"]}
    screener_count = signals_count = signals_eligible = orders_placed = None
    orders_blocked = complete = False
    for line in content.splitlines():
        if re.search(r"Step \d+/\d+ - Detecting market regime", line):
            steps["regime"] = "ok"
        if "Regime written" in line:
            steps["regime"] = "ok"
        if re.search(r"Step \d+/\d+ - Running screener", line):
            steps["screener"] = "ok"
        m = re.search(r"Screener complete: (\d+) stocks passed", line)
        if m:
            screener_count = int(m.group(1))
        if re.search(r"Step \d+/\d+ - Running signal scanner", line):
            steps["signals"] = "ok"
        if "Signal scanner failed" in line:
            steps["signals"] = "error"
        m = re.search(r"(\d+) signal\(s\) found", line)
        if m:
            signals_count = int(m.group(1))
        # Eligible after regime/momentum filter
        m = re.search(r"(\d+) signal\(s\) after (?:regime|momentum) filter", line)
        if m:
            signals_eligible = int(m.group(1))
        if "No entry signals today" in line and signals_count is None:
            signals_count = 0
        if re.search(r"Step \d+/\d+ - Running risk engine", line):
            steps["risk"] = "ok"
        if "Risk engine BLOCKED all orders" in line:
            steps["risk"] = "warn"
        m = re.search(r"Orders placed[: ]+(\d+)", line)
        if m:
            orders_placed = int(m.group(1))
        if "Pipeline complete" in line or "Daily run complete" in line:
            complete = True
    return dict(
        run_date=run_date, run_time=run_time, steps=steps,
        screener_count=screener_count,
        signals_count=signals_count,
        signals_eligible=signals_eligible,
        orders_placed=orders_placed,
        orders_blocked=orders_blocked,
        complete=complete, log_file=log_path.name,
    )


# ---------------------------------------------------------------------------
# Derived data helpers
# ---------------------------------------------------------------------------

def _build_live_equity(equity_df: pd.DataFrame) -> pd.Series:
    """Convert equity_curve exits into a running account balance series."""
    if equity_df.empty or "pnl_aud" not in equity_df.columns:
        return pd.Series(dtype=float)
    return equity_df.set_index("date")["pnl_aud"].cumsum() + STARTING_BALANCE


def _compute_drawdown(equity: pd.Series) -> pd.Series:
    if equity.empty:
        return pd.Series(dtype=float)
    peak = equity.cummax()
    return (equity - peak) / peak * 100


def _split_trades(trades_df: pd.DataFrame, equity_df: pd.DataFrame):
    """Return (open_df, closed_df) by matching order_ref."""
    if trades_df.empty:
        return pd.DataFrame(), pd.DataFrame()
    if equity_df.empty or "order_ref" not in equity_df.columns:
        return trades_df.copy(), pd.DataFrame()
    closed_refs = set(equity_df["order_ref"].dropna())
    is_closed   = trades_df["order_ref"].isin(closed_refs)
    return trades_df[~is_closed].copy(), trades_df[is_closed].copy()


def _enrich_closed(closed_df: pd.DataFrame, equity_df: pd.DataFrame) -> pd.DataFrame:
    """Join entry records with exit records to build full trade log."""
    if closed_df.empty or equity_df.empty:
        return pd.DataFrame()
    exits = equity_df[["order_ref", "date", "exit_type", "pnl_aud"]].copy()
    exits.rename(columns={"date": "exit_date"}, inplace=True)
    merged = closed_df.merge(exits, on="order_ref", how="inner")
    if "risk_aud" in merged.columns:
        merged["pnl_r"] = (merged["pnl_aud"] / merged["risk_aud"]).round(2)
    else:
        merged["pnl_r"] = np.nan
    return merged


def _rolling_metrics(merged: pd.DataFrame, n: int = 30) -> dict:
    result = dict(win_rate=np.nan, profit_factor=np.nan, expectancy=np.nan, avg_r=np.nan, n=0)
    if merged.empty or "pnl_r" not in merged.columns:
        return result
    recent = merged.tail(n).copy()
    result["n"] = len(recent)
    if len(recent) == 0:
        return result
    wins   = recent[recent["pnl_r"] > 0]["pnl_r"]
    losses = recent[recent["pnl_r"] <= 0]["pnl_r"]
    result["win_rate"]  = round(len(wins) / len(recent) * 100, 1)
    result["avg_r"]     = round(float(recent["pnl_r"].mean()), 3)
    if "pnl_aud" in recent.columns:
        result["expectancy"] = round(float(recent["pnl_aud"].mean()), 1)
    gross_profit = wins.sum()
    gross_loss   = abs(losses.sum())
    result["profit_factor"] = round(gross_profit / gross_loss, 2) if gross_loss > 0 else np.nan
    return result


def _losing_streak(merged: pd.DataFrame) -> tuple[int, int]:
    if merged.empty or "pnl_r" not in merged.columns:
        return 0, 0
    sort_col = "exit_date" if "exit_date" in merged.columns else merged.columns[0]
    results  = (merged.sort_values(sort_col)["pnl_r"] > 0).tolist()
    max_streak = cur_streak = 0
    for win in results:
        if not win:
            cur_streak += 1
            max_streak = max(max_streak, cur_streak)
        else:
            cur_streak = 0
    cur = 0
    for win in reversed(results):
        if not win:
            cur += 1
        else:
            break
    return cur, max_streak


def _exit_type_breakdown(merged: pd.DataFrame) -> dict:
    if merged.empty or "exit_type" not in merged.columns:
        return {}
    return merged["exit_type"].value_counts().to_dict()


def _gap_stop_stats(merged: pd.DataFrame) -> dict:
    result = dict(count=0, pct=0.0, avg_gap_loss_r=np.nan,
                  entry_slip_pct=np.nan, exit_slip_r=np.nan, gap_excess_r=np.nan)
    if merged.empty or "exit_type" not in merged.columns:
        return result
    gap     = merged[merged["exit_type"].str.upper().str.contains("GAP", na=False)]
    non_gap = merged[~merged["exit_type"].str.upper().str.contains("GAP", na=False)]
    result["count"] = len(gap)
    result["pct"]   = round(len(gap) / len(merged) * 100, 1) if len(merged) else 0.0
    if not gap.empty and "pnl_r" in merged.columns:
        result["avg_gap_loss_r"] = round(float(gap["pnl_r"].mean()), 2)
        # Gap excess loss beyond normal -1R stop
        result["gap_excess_r"] = round(float((gap["pnl_r"] - (-1.0)).mean()), 2)
    # Exit slippage for normal stops: deviation of actual R from expected -1R
    if "exit_type" in merged.columns and "pnl_r" in merged.columns:
        normal_stops = merged[
            merged["exit_type"].str.upper().str.contains("STOP", na=False) &
            ~merged["exit_type"].str.upper().str.contains("GAP", na=False)
        ]
        if not normal_stops.empty:
            result["exit_slip_r"] = round(float((normal_stops["pnl_r"] - (-1.0)).mean()), 3)
    return result


def _regime_performance_matrix(merged: pd.DataFrame) -> pd.DataFrame:
    """Full regime breakdown: trades, win rate, avg R, total P&L — validates Variant F thesis."""
    col = None
    for c in ["f_regime", "regime", "regime_at_entry"]:
        if c in merged.columns:
            col = c
            break
    if col is None or merged.empty:
        return pd.DataFrame()
    rows = []
    for regime in ["STRONG_BULL", "WEAK_BULL", "CHOPPY_BEAR"]:
        grp = merged[merged[col] == regime]
        if grp.empty:
            rows.append({
                "Regime": regime, "Trades": 0,
                "Win %": "—", "Avg R": "—", "Total P&L": "—",
                "BT Win %": f"{BT['regime_win_rate'].get(regime, '—'):.1f}%",
                "BT Avg R": f"{BT['regime_avg_r'].get(regime, '—'):.2f}",
            })
            continue
        wins     = (grp["pnl_r"] > 0).sum() if "pnl_r" in grp.columns else 0
        total    = len(grp)
        win_rate = round(wins / total * 100, 1) if total else 0.0
        avg_r    = round(float(grp["pnl_r"].mean()), 3) if "pnl_r" in grp.columns else np.nan
        pnl_aud  = round(float(grp["pnl_aud"].sum()), 0) if "pnl_aud" in grp.columns else np.nan
        rows.append({
            "Regime":    regime,
            "Trades":    total,
            "Win %":     f"{win_rate:.1f}%",
            "Avg R":     f"{avg_r:.3f}" if not math.isnan(avg_r) else "—",
            "Total P&L": f"${pnl_aud:+,.0f}" if not math.isnan(pnl_aud) else "—",
            "BT Win %":  f"{BT['regime_win_rate'].get(regime, 0):.1f}%",
            "BT Avg R":  f"{BT['regime_avg_r'].get(regime, 0):.2f}",
        })
    return pd.DataFrame(rows)


def _missed_trades_tracker(last_run: dict, signals_df: pd.DataFrame,
                            trades_df: pd.DataFrame) -> dict:
    """
    Funnel: signals generated → eligible after filters → executed → missed.
    Missed hypothetical P&L uses backtest avg R × avg risk per trade.
    """
    out = dict(
        generated=0, eligible=0, executed=0, missed=0,
        hypo_pnl=0.0, hypo_r=0.0,
    )
    # signals generated (raw RS + trigger scan before regime filter)
    out["generated"] = last_run.get("signals_count") or 0

    # eligible = signals that passed regime/momentum filter
    # signals_eligible is parsed from log; fallback to len(signals_df)
    eligible = last_run.get("signals_eligible")
    if eligible is None:
        eligible = len(signals_df) if not signals_df.empty else out["generated"]
    out["eligible"] = eligible

    # executed = orders placed today
    out["executed"] = last_run.get("orders_placed") or 0

    # missed = eligible but not placed (heat/sector cap/position limit)
    out["missed"] = max(0, out["eligible"] - out["executed"])

    # hypothetical missed P&L
    avg_risk = float(trades_df["risk_aud"].mean()) if (
        not trades_df.empty and "risk_aud" in trades_df.columns) else AVG_RISK_PER_TRADE
    out["hypo_r"]   = round(out["missed"] * BT["avg_r"], 2)
    out["hypo_pnl"] = round(out["missed"] * BT["avg_r"] * avg_risk, 0)
    return out


def _heat_over_time_series(trades_df: pd.DataFrame, equity_df: pd.DataFrame) -> pd.Series:
    """
    Reconstruct daily portfolio heat % from open risk for each trading day.
    For each calendar day, sum risk_aud of all trades open on that day,
    divide by STARTING_BALANCE × 100.
    """
    if trades_df.empty or "timestamp" not in trades_df.columns:
        return pd.Series(dtype=float)
    if "risk_aud" not in trades_df.columns:
        return pd.Series(dtype=float)

    entries = trades_df[["timestamp", "order_ref", "risk_aud"]].copy()
    # Keep as Timestamps throughout — avoids dtype mismatch when exit_date is all-NaT
    entries["entry_date"] = pd.to_datetime(entries["timestamp"])

    # Get exit dates from equity_df
    if not equity_df.empty and "order_ref" in equity_df.columns and "date" in equity_df.columns:
        exit_map = equity_df.set_index("order_ref")["date"].to_dict()
    else:
        exit_map = {}

    entries["exit_date"] = pd.to_datetime(entries["order_ref"].map(exit_map))

    if entries["entry_date"].empty:
        return pd.Series(dtype=float)

    first_day = entries["entry_date"].min()
    last_day  = pd.Timestamp(date.today())
    all_days  = pd.date_range(first_day, last_day, freq="B")  # business days

    heat_vals = []
    for day in all_days:
        # Compare Timestamps directly — no .date() conversion needed
        open_risk = entries[
            (entries["entry_date"] <= day) &
            (entries["exit_date"].isna() | (entries["exit_date"] >= day))
        ]["risk_aud"].sum()
        heat_vals.append(open_risk / STARTING_BALANCE * 100)

    return pd.Series(heat_vals, index=all_days)


def _trade_quality_buckets(merged: pd.DataFrame, signals_df: pd.DataFrame) -> dict:
    """
    Classify closed trades into quality buckets:
      - F-only pass: STRONG_BULL / WEAK_BULL (always allowed by both C and F)
      - F-pass, C-block: CHOPPY_BEAR where roc_20 > 0 AND ema50_slope > 0
                         (F allowed via momentum gate; C would have blocked all CHOPPY_BEAR)
      - Both block (shouldn't exist in live data)

    If roc_20 / ema50_slope columns not available, infer from f_regime only.
    """
    if merged.empty:
        return {}

    regime_col = None
    for c in ["f_regime", "regime"]:
        if c in merged.columns:
            regime_col = c
            break

    has_momentum = "roc_20" in merged.columns and "ema50_slope" in merged.columns

    def bucket(row):
        regime = row.get(regime_col, "") if regime_col else ""
        if regime in ("STRONG_BULL", "WEAK_BULL"):
            return "F-pass / C-pass"
        if regime == "CHOPPY_BEAR":
            if has_momentum:
                roc  = row.get("roc_20", 0)
                slp  = row.get("ema50_slope", 0)
                if roc > 0 and slp > 0:
                    return "F-pass / C-block"
                else:
                    return "Both block"
            return "F-pass / C-block"   # CHOPPY_BEAR that passed momentum gate
        return "Unknown"

    if regime_col:
        merged = merged.copy()
        merged["bucket"] = merged.apply(bucket, axis=1)
    else:
        return {}

    result = {}
    for bkt, grp in merged.groupby("bucket"):
        wins    = (grp["pnl_r"] > 0).sum() if "pnl_r" in grp.columns else 0
        total   = len(grp)
        avg_r   = round(float(grp["pnl_r"].mean()), 3) if ("pnl_r" in grp.columns and total) else np.nan
        pnl_aud = round(float(grp["pnl_aud"].sum()), 0) if ("pnl_aud" in grp.columns and total) else np.nan
        result[bkt] = {
            "trades":   total,
            "win_rate": round(wins / total * 100, 1) if total else 0.0,
            "avg_r":    avg_r,
            "pnl_aud":  pnl_aud,
        }
    return result


def _rolling_expectancy_series(merged: pd.DataFrame) -> dict[int, pd.Series]:
    """Return dict of {window: rolling mean pnl_r} for windows 10, 30, 50."""
    out: dict[int, pd.Series] = {}
    if merged.empty or "pnl_r" not in merged.columns:
        return out
    sort_col = "exit_date" if "exit_date" in merged.columns else merged.columns[0]
    s = merged.sort_values(sort_col)["pnl_r"].reset_index(drop=True)
    for w in [10, 30, 50]:
        if len(s) >= w:
            out[w] = s.rolling(w).mean()
    return out


def _slippage_decomposition(merged: pd.DataFrame, has_paper: bool) -> dict:
    """
    Decompose slippage/cost sources:
      entry_slip_pct  — 0% for paper, N/A for live (no signal_price stored)
      exit_slip_r     — deviation of normal-stop exits from expected -1R
      gap_excess_r    — gap stop average loss beyond normal -1R
      target_slip_r   — deviation of target exits from expected +2R
    """
    out = dict(
        entry_slip_pct = 0.0 if has_paper else None,
        exit_slip_r    = None,
        gap_excess_r   = None,
        target_slip_r  = None,
    )
    if merged.empty or "pnl_r" not in merged.columns or "exit_type" not in merged.columns:
        return out

    def _r_dev(grp, expected):
        if grp.empty:
            return None
        return round(float((grp["pnl_r"] - expected).mean()), 3)

    normal_stops = merged[
        merged["exit_type"].str.upper().str.contains("STOP", na=False) &
        ~merged["exit_type"].str.upper().str.contains("GAP", na=False)
    ]
    gap_stops = merged[merged["exit_type"].str.upper().str.contains("GAP", na=False)]
    targets   = merged[merged["exit_type"].str.upper().str.contains("TARGET|TP", na=False)]

    out["exit_slip_r"]  = _r_dev(normal_stops, -1.0)
    out["gap_excess_r"] = _r_dev(gap_stops,    -1.0)
    out["target_slip_r"]= _r_dev(targets,      +2.0)
    return out


# ---------------------------------------------------------------------------
# Kill conditions
# ---------------------------------------------------------------------------

def _check_kill_conditions(
    roll30: dict,
    gs: dict,
    slip_decomp: dict,
    regime_matrix: pd.DataFrame,
    n_closed: int,
) -> list[str]:
    """
    Return a list of human-readable kill condition strings that are triggered.
    Empty list = all clear.  Conditions are only evaluated after 30 trades.
    """
    if n_closed < DATA_COLLECTION_TRADES:
        return []   # still in data-collection phase

    triggered = []

    # 1. Rolling-30 expectancy turned negative
    ar = roll30.get("avg_r")
    if ar is not None and not math.isnan(ar) and ar < KILL_EXPECTANCY_MIN:
        triggered.append(
            f"Rolling-30 expectancy negative: {ar:.3f}R  "
            f"(threshold ≥ {KILL_EXPECTANCY_MIN}R)"
        )

    # 2. Profit factor below floor
    pf = roll30.get("profit_factor")
    if pf is not None and not math.isnan(pf) and pf < KILL_PF_MIN:
        triggered.append(
            f"Rolling-30 profit factor: {pf:.2f}  "
            f"(threshold ≥ {KILL_PF_MIN})"
        )

    # 3. Gap stop rate doubled vs backtest
    gap_pct = gs.get("pct", 0.0)
    if gap_pct > KILL_GAP_STOP_MAX:
        triggered.append(
            f"Gap stop rate {gap_pct:.1f}% exceeds 2× backtest baseline "
            f"({KILL_GAP_STOP_MAX:.1f}%) — review gap-risk model"
        )

    # 4. Live entry slippage > 0.40% (only meaningful for non-paper)
    entry_slip = slip_decomp.get("entry_slip_pct")
    if entry_slip is not None and not math.isnan(entry_slip) and entry_slip > KILL_SLIPPAGE_MAX:
        triggered.append(
            f"Entry slippage {entry_slip:.2f}% consistently exceeds "
            f"{KILL_SLIPPAGE_MAX:.2f}% ceiling — execution quality degraded"
        )

    # 5. CHOPPY_BEAR regime outperforming STRONG_BULL (regime inversion)
    #    Requires minimum KILL_REGIME_MIN_TRADES trades in each regime before firing.
    if not regime_matrix.empty and "Avg R" in regime_matrix.columns and KILL_REGIME_INVERSION:
        r_map = {}
        t_map = {}   # trade counts per regime
        for _, row in regime_matrix.iterrows():
            regime_name = row["Regime"]
            val = row.get("Avg R", "—")
            try:
                r_map[regime_name] = float(str(val).replace("—", "nan"))
            except (ValueError, TypeError):
                r_map[regime_name] = float("nan")
            try:
                t_map[regime_name] = int(row.get("Trades", 0))
            except (ValueError, TypeError):
                t_map[regime_name] = 0

        cb_r = r_map.get("CHOPPY_BEAR", float("nan"))
        sb_r = r_map.get("STRONG_BULL", float("nan"))
        cb_n = t_map.get("CHOPPY_BEAR", 0)
        sb_n = t_map.get("STRONG_BULL", 0)

        if (not math.isnan(cb_r) and not math.isnan(sb_r) and cb_r > sb_r):
            if cb_n < KILL_REGIME_MIN_TRADES or sb_n < KILL_REGIME_MIN_TRADES:
                # Not enough data — note it as a watch item, not a kill trigger
                triggered.append(
                    f"REGIME INVERSION WATCH: CHOPPY_BEAR avg R ({cb_r:.3f}) > "
                    f"STRONG_BULL avg R ({sb_r:.3f}) — "
                    f"insufficient data to confirm (CB:{cb_n} SB:{sb_n} trades, "
                    f"need {KILL_REGIME_MIN_TRADES} each)"
                )
            else:
                triggered.append(
                    f"REGIME INVERSION: CHOPPY_BEAR avg R ({cb_r:.3f}) > "
                    f"STRONG_BULL avg R ({sb_r:.3f}) — Variant F thesis breaking down "
                    f"(CB:{cb_n} SB:{sb_n} trades — statistically significant)"
                )

    return triggered


# ---------------------------------------------------------------------------
# Capital at risk / latency / missed breakdown
# ---------------------------------------------------------------------------

def _capital_at_risk(open_df: pd.DataFrame, current_bal: float) -> dict:
    """
    Worst-case overnight loss if every open stop is hit simultaneously.
    Returns: max_loss_aud, max_loss_pct, per_trade details.
    """
    if open_df.empty or "risk_aud" not in open_df.columns:
        return {"max_loss_aud": 0.0, "max_loss_pct": 0.0, "positions": []}
    total_risk = float(open_df["risk_aud"].sum())
    positions  = []
    for _, row in open_df.iterrows():
        positions.append({
            "ticker":    row.get("ticker", "—"),
            "risk_aud":  float(row.get("risk_aud", 0)),
            "stop_loss": float(row.get("stop_loss", 0)) if "stop_loss" in row else None,
            "entry":     float(row.get("entry", 0))     if "entry"     in row else None,
        })
    return {
        "max_loss_aud": round(total_risk, 0),
        "max_loss_pct": round(total_risk / current_bal * 100, 2) if current_bal > 0 else 0.0,
        "positions":    positions,
    }


def _parse_execution_latency(last_run: dict) -> float | None:
    """
    Read the most recent run log and return seconds between signal scan completion
    and first order placement.  Returns None if unparseable.
    """
    log_file = last_run.get("log_file")
    if not log_file:
        return None
    log_path = Path("logs") / log_file
    if not log_path.exists():
        return None
    try:
        content = log_path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return None

    signal_ts  = None
    order_ts   = None

    for line in content.splitlines():
        # Match HH:MM:SS at start of line (common logging format)
        ts_m = re.match(r"(\d{2}):(\d{2}):(\d{2})", line.strip())
        if not ts_m:
            continue
        h, mi, s = int(ts_m.group(1)), int(ts_m.group(2)), int(ts_m.group(3))
        secs = h * 3600 + mi * 60 + s

        if signal_ts is None and re.search(
            r"signal\(s\) found|signal scan complete|Signal scanner complete", line, re.I
        ):
            signal_ts = secs

        if order_ts is None and re.search(
            r"order(?:s)? placed|placing order|submitted order", line, re.I
        ):
            order_ts = secs

    if signal_ts is not None and order_ts is not None and order_ts >= signal_ts:
        return float(order_ts - signal_ts)
    return None


def _parse_missed_breakdown(last_run: dict) -> dict:
    """
    Parse run log to split missed signals into four buckets:
      no_fill      — order submitted but fill not confirmed
      risk_blocked — blocked by risk engine (heat / sector cap / position cap)
      regime_blocked — blocked by regime/momentum filter before risk engine
      exec_failure — IBKR submission error
    """
    out = dict(no_fill=0, risk_blocked=0, regime_blocked=0, exec_failure=0)
    log_file = last_run.get("log_file")
    if not log_file:
        return out
    log_path = Path("logs") / log_file
    if not log_path.exists():
        return out
    try:
        content = log_path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return out

    for line in content.splitlines():
        # Regime / momentum block — happens in signals.py before risk engine
        if re.search(r"blocked.{0,30}(regime|momentum|choppy)", line, re.I):
            out["regime_blocked"] += 1
        # Risk engine blocks — heat, sector, position cap
        elif re.search(r"blocked.{0,40}(heat|sector|position|cap|budget)", line, re.I):
            out["risk_blocked"] += 1
        # Execution / fill failure
        elif re.search(r"(fill failed|order failed|submission failed|ibkr error)", line, re.I):
            out["exec_failure"] += 1
        # No fill confirmation (order sent but no ack)
        elif re.search(r"(no fill|unfilled|not filled)", line, re.I):
            out["no_fill"] += 1

    return out


# ---------------------------------------------------------------------------
# Cool-down state management
# ---------------------------------------------------------------------------

def _business_days_since(from_date: date) -> int:
    """Count business days elapsed from from_date to today (inclusive of today)."""
    today = date.today()
    if from_date > today:
        return 0
    return int(np.busday_count(from_date.isoformat(), today.isoformat()))


def _load_cooldown_state() -> dict:
    """
    Load kill-condition cooldown state from results/kill_cooldown.json.
    Schema: {triggered_date: "YYYY-MM-DD", conditions: [...], active: bool}
    """
    default = {"active": False, "triggered_date": None, "conditions": [], "days_elapsed": 0}
    if not COOLDOWN_JSON.exists():
        return default
    try:
        with open(COOLDOWN_JSON) as f:
            data = json.load(f)
        if data.get("active") and data.get("triggered_date"):
            triggered = date.fromisoformat(data["triggered_date"])
            elapsed   = _business_days_since(triggered)
            data["days_elapsed"]   = elapsed
            data["days_remaining"] = max(0, KILL_COOLDOWN_DAYS - elapsed)
            if elapsed >= KILL_COOLDOWN_DAYS:
                data["active"] = False   # cooldown expired
        return data
    except Exception:
        return default


def _save_cooldown_state(conditions: list[str]) -> None:
    """
    Write cooldown state.  Called when kill conditions first fire (today's date
    becomes the trigger date) or when they clear after expiry.
    Only sets triggered_date if not already active — preserves original trigger date.
    """
    try:
        existing = {}
        if COOLDOWN_JSON.exists():
            with open(COOLDOWN_JSON) as f:
                existing = json.load(f)

        if conditions:
            # Only stamp the date if we're not already in an active cooldown
            if not existing.get("active"):
                existing["triggered_date"] = date.today().isoformat()
            existing["active"]     = True
            existing["conditions"] = conditions
        else:
            # Conditions cleared — let cooldown expire naturally (keep triggered_date)
            pass

        COOLDOWN_JSON.parent.mkdir(parents=True, exist_ok=True)
        with open(COOLDOWN_JSON, "w") as f:
            json.dump(existing, f, indent=2)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Soft breach counter
# ---------------------------------------------------------------------------

def _current_amber_metrics(roll30: dict, gs: dict) -> list[str]:
    """
    Return list of metric names currently in the amber zone (15–30% drift from BT).
    Used to feed the soft breach counter.
    """
    amber = []

    def _is_amber(val, bt_val, lower_is_better=False):
        if val is None or (isinstance(val, float) and math.isnan(val)):
            return False
        if bt_val == 0:
            return False
        drift = (val - bt_val) / abs(bt_val)
        if lower_is_better:
            drift = -drift
        return -DRIFT_THRESHOLD < drift < -DRIFT_WARN   # amber band only

    if _is_amber(roll30.get("avg_r"),         BT["avg_r"]):          amber.append("Avg R")
    if _is_amber(roll30.get("win_rate"),       BT["win_rate"]):       amber.append("Win Rate")
    if _is_amber(roll30.get("profit_factor"),  BT["profit_factor"]):  amber.append("Profit Factor")
    if _is_amber(roll30.get("expectancy"),     BT["expectancy"]):     amber.append("Expectancy")
    if _is_amber(gs.get("pct", 0.0), BT["gap_stop_pct"], lower_is_better=True):
        amber.append("Gap Stop %")
    return amber


def _load_soft_breach_state() -> dict:
    """
    Load soft breach counter from results/soft_breach.json.
    Schema: {consecutive_days: N, last_date: "YYYY-MM-DD", metrics: [...]}
    """
    default = {"consecutive_days": 0, "last_date": None, "metrics": []}
    if not SOFT_BREACH_JSON.exists():
        return default
    try:
        with open(SOFT_BREACH_JSON) as f:
            return json.load(f)
    except Exception:
        return default


def _update_soft_breach_state(roll30: dict, gs: dict, n_closed: int) -> dict:
    """
    Compare today's metrics against yesterday's state and update the counter.
    Only runs once per calendar day (guards against dashboard re-renders).
    Only active after data-collection phase.
    Returns the current state dict.
    """
    state    = _load_soft_breach_state()
    today_s  = date.today().isoformat()

    # Don't update during data collection, and only update once per day
    if n_closed < DATA_COLLECTION_TRADES or state.get("last_date") == today_s:
        return state

    amber_metrics = _current_amber_metrics(roll30, gs)

    if amber_metrics:
        # Only increment if we had amber yesterday too (consecutive)
        yesterday = (date.today() - timedelta(days=1)).isoformat()
        was_amber_yesterday = (state.get("last_date") in (yesterday, today_s)
                               and state.get("consecutive_days", 0) > 0)
        state["consecutive_days"] = (state.get("consecutive_days", 0) + 1
                                     if was_amber_yesterday else 1)
    else:
        state["consecutive_days"] = 0

    state["last_date"] = today_s
    state["metrics"]   = amber_metrics

    try:
        SOFT_BREACH_JSON.parent.mkdir(parents=True, exist_ok=True)
        with open(SOFT_BREACH_JSON, "w") as f:
            json.dump(state, f, indent=2)
    except Exception:
        pass

    return state


# ---------------------------------------------------------------------------
# Chart builders
# ---------------------------------------------------------------------------

def _chart_equity(live_eq: pd.Series, bt_eq: pd.Series) -> go.Figure:
    fig = go.Figure()
    if not bt_eq.empty:
        bt_norm = bt_eq / bt_eq.iloc[0] * STARTING_BALANCE
        fig.add_trace(go.Scatter(
            x=bt_norm.index, y=bt_norm.values,
            mode="lines", name="Backtest",
            line=dict(color="#404040", width=1, dash="dot"),
        ))
    if not live_eq.empty:
        fig.add_trace(go.Scatter(
            x=live_eq.index, y=live_eq.values,
            mode="lines", name="Live",
            line=dict(color=GREEN, width=2),
            fill="tozeroy", fillcolor="rgba(38,166,154,0.07)",
        ))
    else:
        fig.add_trace(go.Scatter(
            x=[datetime.today()], y=[STARTING_BALANCE],
            mode="markers", name="Start",
            marker=dict(color=AMBER, size=8),
        ))
    fig.add_hline(y=STARTING_BALANCE, line_dash="dot", line_color="#333333", line_width=1)
    layout = dict(**PLOT_LAYOUT, height=220)
    layout["yaxis"] = dict(**PLOT_LAYOUT["yaxis"], tickprefix="$", tickformat=",.0f")
    fig.update_layout(**layout, showlegend=True,
                      legend=dict(x=0.01, y=0.99, bgcolor="rgba(0,0,0,0)", font=dict(size=10)))
    return fig


def _chart_drawdown(live_eq: pd.Series) -> go.Figure:
    fig = go.Figure()
    if not live_eq.empty:
        dd = _compute_drawdown(live_eq)
        fig.add_trace(go.Scatter(
            x=dd.index, y=dd.values,
            mode="lines", name="Drawdown",
            line=dict(color=RED, width=1.5),
            fill="tozeroy", fillcolor="rgba(239,83,80,0.12)",
        ))
        fig.add_hline(y=-BT["max_dd"], line_dash="dot", line_color="#555555", line_width=1,
                      annotation_text=f"BT max −{BT['max_dd']}%",
                      annotation_font_color="#888",
                      annotation_position="bottom right")
    layout = dict(**PLOT_LAYOUT, height=130)
    layout["yaxis"] = dict(**PLOT_LAYOUT["yaxis"], ticksuffix="%")
    fig.update_layout(**layout, showlegend=False)
    return fig


def _chart_heat_over_time(heat_series: pd.Series) -> go.Figure:
    fig = go.Figure()
    if not heat_series.empty:
        colours = [RED if v > HEAT_BUDGET else AMBER if v > HEAT_BUDGET * 0.5 else GREEN
                   for v in heat_series.values]
        fig.add_trace(go.Scatter(
            x=heat_series.index, y=heat_series.values,
            mode="lines", name="Heat %",
            line=dict(color=AMBER, width=1.5),
            fill="tozeroy", fillcolor="rgba(245,166,35,0.08)",
        ))
        fig.add_hline(y=HEAT_BUDGET, line_dash="dot", line_color=RED, line_width=1,
                      annotation_text=f"Budget {HEAT_BUDGET}%",
                      annotation_font_color=RED,
                      annotation_position="top right")
    layout = dict(**PLOT_LAYOUT, height=180)
    layout["yaxis"] = dict(**PLOT_LAYOUT["yaxis"], ticksuffix="%")
    fig.update_layout(**layout, showlegend=False)
    return fig


def _chart_rolling_expectancy(roll_dict: dict[int, pd.Series], n_closed: int) -> go.Figure:
    fig = go.Figure()
    # Rolling 10: reduced opacity + dashed until 30 trades (low-confidence zone)
    low_conf = n_closed < DATA_COLLECTION_TRADES
    palette  = {10: BLUE, 30: GREEN, 50: AMBER}
    for w, series in sorted(roll_dict.items()):
        series_clean = series.dropna()
        if series_clean.empty:
            continue
        is_10  = w == 10
        opacity  = 0.35 if (is_10 and low_conf) else 1.0
        dash     = "dot" if (is_10 and low_conf) else "solid"
        width    = 1.0 if is_10 else 1.8
        name     = f"Rolling {w}" + (" (low confidence)" if is_10 and low_conf else "")
        fig.add_trace(go.Scatter(
            x=list(range(len(series_clean))),
            y=series_clean.values,
            mode="lines", name=name,
            opacity=opacity,
            line=dict(color=palette.get(w, GREY), width=width, dash=dash),
        ))
    fig.add_hline(y=0, line_dash="dot", line_color="#333333", line_width=1)
    fig.add_hline(y=BT["avg_r"], line_dash="dot", line_color="#404040", line_width=1,
                  annotation_text=f"BT {BT['avg_r']}R",
                  annotation_font_color="#888",
                  annotation_position="top right")
    # Shade the low-confidence region (first 30 trades)
    if low_conf and roll_dict:
        max_x = max(len(s.dropna()) for s in roll_dict.values()) if roll_dict else 0
        if max_x > 0:
            fig.add_vrect(
                x0=0, x1=min(max_x, DATA_COLLECTION_TRADES),
                fillcolor="rgba(245,166,35,0.05)",
                line_width=0,
                annotation_text="data collection",
                annotation_position="top left",
                annotation_font_color="#f5a623",
                annotation_font_size=10,
            )
    layout = dict(**PLOT_LAYOUT, height=220)
    layout["xaxis"] = dict(**PLOT_LAYOUT["xaxis"], title="Trade #")
    layout["yaxis"] = dict(**PLOT_LAYOUT["yaxis"], ticksuffix="R")
    fig.update_layout(**layout, showlegend=True,
                      legend=dict(x=0.01, y=0.99, bgcolor="rgba(0,0,0,0)", font=dict(size=10)))
    return fig


def _chart_regime_dist(bt_trades: pd.DataFrame, live_merged: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    regime_col = None
    for c in ["regime", "f_regime"]:
        if c in bt_trades.columns:
            regime_col = c
            break
    regimes = ["STRONG_BULL", "WEAK_BULL", "CHOPPY_BEAR"]
    if regime_col and not bt_trades.empty:
        bt_counts = bt_trades[regime_col].value_counts()
        fig.add_trace(go.Bar(
            name="Backtest", x=regimes,
            y=[bt_counts.get(r, 0) for r in regimes],
            marker_color=["rgba(38,166,154,0.35)", "rgba(245,166,35,0.35)", "rgba(239,83,80,0.35)"],
        ))
    if not live_merged.empty:
        for c in ["f_regime", "regime", "regime_at_entry"]:
            if c in live_merged.columns:
                lv_counts = live_merged[c].value_counts()
                fig.add_trace(go.Bar(
                    name="Live", x=regimes,
                    y=[lv_counts.get(r, 0) for r in regimes],
                    marker_color=[GREEN, AMBER, RED],
                ))
                break
    layout = dict(**PLOT_LAYOUT, height=180, barmode="group")
    fig.update_layout(**layout, showlegend=True,
                      legend=dict(x=0.01, y=0.99, bgcolor="rgba(0,0,0,0)", font=dict(size=10)))
    return fig


# ---------------------------------------------------------------------------
# UI helpers
# ---------------------------------------------------------------------------

def _metric(label: str, value: str, delta: str | None = None, delta_colour: str = GREY):
    delta_html = (f'<div style="color:{delta_colour};font-size:12px;">{delta}</div>'
                  if delta else "")
    st.markdown(
        f"""<div style="background:#161616;border:1px solid #2a2a2a;border-radius:8px;
            padding:12px 16px;margin-bottom:4px;">
          <div style="color:#888;font-size:11px;margin-bottom:4px;">{label}</div>
          <div style="color:#e0e0e0;font-size:20px;font-weight:600;">{value}</div>
          {delta_html}
        </div>""",
        unsafe_allow_html=True,
    )


def _regime_pill(f_regime: str) -> str:
    cls = {"STRONG_BULL": "pill-green", "WEAK_BULL": "pill-amber", "CHOPPY_BEAR": "pill-red"}
    return f'<span class="pill {cls.get(f_regime, "pill-amber")}">{f_regime}</span>'


def _phase_note():
    st.markdown(
        '<div class="phase-note">ℹ️ First 30 trades = data collection phase, not evaluation phase</div>',
        unsafe_allow_html=True,
    )


def _safe_nan(v) -> bool:
    """True if v is None or a float NaN."""
    return v is None or (isinstance(v, float) and math.isnan(v))


def _drift_colour(drift: float) -> str:
    """Return colour string based on drift magnitude."""
    if drift < -DRIFT_THRESHOLD:
        return RED
    if drift < -DRIFT_WARN:
        return AMBER
    return GREEN


# ---------------------------------------------------------------------------
# ── MAIN ────────────────────────────────────────────────────────────────────
# ---------------------------------------------------------------------------

# --- Load all data ---
regime      = _load_regime()
trades_df   = _load_trades()
pending_df  = _load_pending()
equity_df   = _load_equity()
signals_df  = _load_signals()
bt_eq       = _load_backtest_equity()
bt_trades   = _load_backtest_trades()
last_run    = _parse_last_run()
stale_df    = _stale_pending(pending_df)

# --- Derived ---
live_eq              = _build_live_equity(equity_df)
open_df, closed_df   = _split_trades(trades_df, equity_df)
merged               = _enrich_closed(closed_df, equity_df)

# --- Portfolio state ---
n_open        = len(open_df)
heat_pct      = float(open_df["risk_aud"].sum() / STARTING_BALANCE * 100) if (
    not open_df.empty and "risk_aud" in open_df.columns) else 0.0
exposure_pct  = n_open / MAX_POSITIONS * 100
current_bal   = float(live_eq.iloc[-1]) if not live_eq.empty else STARTING_BALANCE
total_pnl     = current_bal - STARTING_BALANCE
total_ret     = total_pnl / STARTING_BALANCE * 100
current_dd    = float(_compute_drawdown(live_eq).iloc[-1]) if not live_eq.empty else 0.0
max_dd_live   = float(_compute_drawdown(live_eq).min())    if not live_eq.empty else 0.0

# --- Paper trade detection ---
has_paper = (not trades_df.empty and "paper" in trades_df.columns
             and trades_df["paper"].astype(str).str.lower().eq("true").all())

# --- Rolling stats ---
roll30    = _rolling_metrics(merged, 30)
cur_streak, max_streak = _losing_streak(merged)
gs        = _gap_stop_stats(merged)
missed    = _missed_trades_tracker(last_run, signals_df, trades_df)
heat_ts   = _heat_over_time_series(trades_df, equity_df)
roll_exp  = _rolling_expectancy_series(merged)
quality   = _trade_quality_buckets(merged, signals_df)
slip_decomp = _slippage_decomposition(merged, has_paper)
regime_matrix = _regime_performance_matrix(merged)

n_closed  = len(merged)

# --- New derived: kill conditions, capital at risk, latency, missed breakdown ---
kill_conditions  = _check_kill_conditions(roll30, gs, slip_decomp, regime_matrix, n_closed)
cap_at_risk      = _capital_at_risk(open_df, current_bal)
exec_latency_s   = _parse_execution_latency(last_run)
missed_breakdown = _parse_missed_breakdown(last_run)

# --- Cool-down: persist trigger date on first fire; load current state ---
if kill_conditions:
    _save_cooldown_state(kill_conditions)
cooldown_state  = _load_cooldown_state()

# --- Soft breach: update counter once per day, load current state ---
soft_breach     = _update_soft_breach_state(roll30, gs, n_closed)

# ---------------------------------------------------------------------------
# ── MOBILE VIEW  (rendered when ?view=mobile, then st.stop())
# ---------------------------------------------------------------------------
if _is_mobile:
    # Live prices for per-trade P&L
    _mob_tickers  = tuple(open_df["ticker"].tolist()) if not open_df.empty else ()
    _mob_prices   = _fetch_live_prices(_mob_tickers)
    _mob_pnl, _mob_pct = _calc_live_pnl(open_df, equity_df, _mob_prices)

    # Regime
    _mob_regime   = regime.get("f_regime", "STRONG_BULL")
    _mob_r_bg, _mob_r_col = {
        "STRONG_BULL": ("#0d2b1e", "#26a69a"),
        "WEAK_BULL":   ("#2b1e0d", "#f5a623"),
        "CHOPPY_BEAR": ("#2b1500", "#ff7043"),
        "BEAR":        ("#2b0d0d", "#ef5350"),
        "HIGH_VOL":    ("#1a0d2b", "#b39ddb"),
    }.get(_mob_regime, ("#1a1a1a", "#888"))

    # Last run dot
    _mob_run_date  = last_run.get("run_date")
    _mob_ran_today = (_mob_run_date == date.today()) if _mob_run_date else False
    _mob_dot_col   = "#26a69a" if _mob_ran_today else "#ef5350"
    _mob_dot       = "●"
    _mob_date_str  = (
        _mob_run_date.strftime("%d %b") if hasattr(_mob_run_date, "strftime")
        else str(_mob_run_date or "—")
    )

    # P&L colours
    _mob_pnl_col  = "#26a69a" if _mob_pnl >= 0 else "#ef5350"
    _mob_pnl_sign = "+" if _mob_pnl >= 0 else ""
    _mob_pct_sign = "+" if _mob_pct >= 0 else ""

    # ── Header strip ───────────────────────────────────────────────────────
    st.markdown(
        f'<div class="mob-header">'
        f'  <span style="background:{_mob_r_bg};color:{_mob_r_col};padding:6px 14px;'
        f'    border-radius:6px;font-size:15px;font-weight:800;letter-spacing:0.05em;">'
        f'    {_mob_regime.replace("_", " ")}'
        f'  </span>'
        f'  <span class="mob-run-dot" style="color:{_mob_dot_col};">'
        f'    {_mob_date_str} {_mob_dot}'
        f'  </span>'
        f'</div>',
        unsafe_allow_html=True,
    )

    # ── Total P&L ───────────────────────────────────────────────────────────
    st.markdown(
        f'<div class="mobile-pnl" style="color:{_mob_pnl_col};">'
        f'  {_mob_pnl_sign}${abs(_mob_pnl):,.0f}'
        f'</div>'
        f'<div class="mobile-pnl-pct" style="color:{_mob_pnl_col};">'
        f'  {_mob_pct_sign}{_mob_pct:.2f}%'
        f'</div>',
        unsafe_allow_html=True,
    )

    # ── Open trades list ────────────────────────────────────────────────────
    st.markdown('<div class="mob-section-title">OPEN POSITIONS</div>', unsafe_allow_html=True)

    if open_df.empty:
        st.markdown(
            '<div style="color:#555;text-align:center;padding:24px;font-size:16px;">'
            'No open positions</div>',
            unsafe_allow_html=True,
        )
    else:
        trade_rows_html = ""
        for _, row in open_df.iterrows():
            ticker = str(row.get("ticker", "?")).replace(".AX", "")
            try:
                entry  = float(row.get("entry", 0))
                shares = int(row.get("shares", 0))
            except (ValueError, TypeError):
                entry = shares = 0

            live_px = _mob_prices.get(str(row.get("ticker", "")), 0.0)
            if live_px > 0 and shares > 0:
                trade_unrl = (live_px - entry) * shares
                arrow      = "↑" if trade_unrl >= 0 else "↓"
                t_col      = "#26a69a" if trade_unrl >= 0 else "#ef5350"
                t_sign     = "+" if trade_unrl >= 0 else ""
                pnl_str    = f"{t_sign}${abs(trade_unrl):,.0f}"
            else:
                t_col   = "#555"
                arrow   = "—"
                pnl_str = "—"

            trade_rows_html += (
                f'<div class="mobile-trade-row">'
                f'  <span class="mobile-ticker" style="color:#e0e0e0;">{ticker}</span>'
                f'  <span style="display:flex;align-items:center;gap:10px;">'
                f'    <span class="mobile-trade-pnl" style="color:{t_col};">{pnl_str}</span>'
                f'    <span style="font-size:26px;color:{t_col};">{arrow}</span>'
                f'  </span>'
                f'</div>'
            )

        st.markdown(
            f'<div style="background:#0d0d0d;border:1px solid #222;border-radius:10px;'
            f'overflow:hidden;margin-top:4px;">{trade_rows_html}</div>',
            unsafe_allow_html=True,
        )

    # Kill conditions — just a single line at the bottom if triggered
    if kill_conditions:
        st.markdown(
            f'<div style="margin-top:20px;background:#1a0505;border:2px solid #ef5350;'
            f'border-radius:8px;padding:16px;text-align:center;">'
            f'<span style="color:#ef5350;font-size:20px;font-weight:700;">'
            f'🛑 {len(kill_conditions)} KILL CONDITION{"S" if len(kill_conditions) != 1 else ""} TRIGGERED'
            f'</span></div>',
            unsafe_allow_html=True,
        )

    # Auto-refresh then stop — don't render desktop view
    time.sleep(REFRESH_SECS)
    st.rerun()

# ---------------------------------------------------------------------------
# ── SNAPSHOT BAR ─────────────────────────────────────────────────────────
# ---------------------------------------------------------------------------

# ── Live price fetch (60-second cache) ──────────────────────────────────
_open_tickers  = tuple(open_df["ticker"].tolist()) if not open_df.empty else ()
_live_prices   = _fetch_live_prices(_open_tickers)
_live_pnl, _live_pct = _calc_live_pnl(open_df, equity_df, _live_prices)

# ── Regime cell ─────────────────────────────────────────────────────────
f_regime   = regime.get("f_regime", "STRONG_BULL")
regime_mlt = regime.get("f_regime_size_multiplier", 1.0)

_REGIME_STYLE = {
    "STRONG_BULL": ("#0d2b1e", "#26a69a"),
    "WEAK_BULL":   ("#2b1e0d", "#f5a623"),
    "CHOPPY_BEAR": ("#2b1500", "#ff7043"),
    "BEAR":        ("#2b0d0d", "#ef5350"),
    "HIGH_VOL":    ("#1a0d2b", "#b39ddb"),
}
_r_bg, _r_col = _REGIME_STYLE.get(f_regime, ("#1a1a1a", "#888888"))
_regime_html = (
    f'<div class="snap-cell">'
    f'  <div class="snap-label">REGIME</div>'
    f'  <span class="regime-badge" style="background:{_r_bg};color:{_r_col};">'
    f'    {f_regime.replace("_", " ")}'
    f'  </span>'
    f'  <div class="snap-sub">size ×{regime_mlt:.1f}</div>'
    f'</div>'
)

# ── Last run cell ────────────────────────────────────────────────────────
_run_date = last_run.get("run_date")
_run_today = (_run_date == date.today()) if _run_date else False
_run_col  = GREEN if _run_today else (RED if _run_date else GREY)
_run_time = last_run.get("run_time", "—")
_run_ok   = "✓" if last_run.get("complete") else ("⚠" if last_run else "—")
_run_date_str = _run_date.strftime("%d %b") if hasattr(_run_date, "strftime") else str(_run_date or "—")
_lastrun_html = (
    f'<div class="snap-cell">'
    f'  <div class="snap-label">LAST RUN</div>'
    f'  <div class="snap-value" style="color:{_run_col};font-size:20px;">{_run_time} {_run_ok}</div>'
    f'  <div class="snap-sub">{_run_date_str}</div>'
    f'</div>'
)

# ── P&L cell ─────────────────────────────────────────────────────────────
_pnl_col  = GREEN if _live_pnl >= 0 else RED
_pnl_sign = "+" if _live_pnl >= 0 else ""
_pct_sign = "+" if _live_pct >= 0 else ""
_has_live = bool(_live_prices) and not open_df.empty
_pnl_sub  = f'{_pct_sign}{_live_pct:.2f}%' + (' · live' if _has_live else ' · entry')
_pnl_html = (
    f'<div class="snap-cell">'
    f'  <div class="snap-label">PORTFOLIO P&L</div>'
    f'  <div class="snap-value" style="color:{_pnl_col};">'
    f'    {_pnl_sign}${abs(_live_pnl):,.0f}'
    f'  </div>'
    f'  <div class="snap-sub" style="color:{_pnl_col};">{_pnl_sub}</div>'
    f'</div>'
)

# ── Heat cell ─────────────────────────────────────────────────────────────
_heat_ratio = heat_pct / HEAT_BUDGET if HEAT_BUDGET > 0 else 0
_heat_col   = RED if _heat_ratio >= 0.8 else AMBER if _heat_ratio >= 0.6 else GREEN
_heat_html  = (
    f'<div class="snap-cell">'
    f'  <div class="snap-label">HEAT</div>'
    f'  <div class="snap-value" style="color:{_heat_col};">{heat_pct:.1f}%</div>'
    f'  <div class="snap-sub">of {HEAT_BUDGET:.0f}% budget</div>'
    f'</div>'
)

# ── Positions cell ────────────────────────────────────────────────────────
_pos_col  = RED if n_open >= MAX_POSITIONS else AMBER if n_open >= int(MAX_POSITIONS * 0.67) else GREEN
_pos_html = (
    f'<div class="snap-cell">'
    f'  <div class="snap-label">POSITIONS</div>'
    f'  <div class="snap-value" style="color:{_pos_col};">{n_open}<span style="color:#333;font-size:16px;">/{MAX_POSITIONS}</span></div>'
    f'  <div class="snap-sub">slots used</div>'
    f'</div>'
)

# ── Kill conditions cell ──────────────────────────────────────────────────
_has_kill   = bool(kill_conditions)
_kpill_bg   = "#2b0d0d" if _has_kill else "#0d2b1e"
_kpill_col  = RED       if _has_kill else GREEN
_kpill_bdr  = "#ef535044" if _has_kill else "#26a69a44"
_kpill_txt  = f"🛑 {len(kill_conditions)} TRIGGERED" if _has_kill else "✅ ALL CLEAR"

# Build kill detail content for the <details> expand
if _has_kill:
    _kill_items_html = "".join(
        f'<div class="snap-kill-item">⛔ {c}</div>' for c in kill_conditions
    )
    _kill_detail_html = f'<div class="snap-kill-detail">{_kill_items_html}</div>'
elif n_closed < DATA_COLLECTION_TRADES:
    _kill_detail_html = (
        f'<div class="snap-kill-detail snap-kill-clear">'
        f'Data collection phase — {n_closed}/{DATA_COLLECTION_TRADES} trades. '
        f'Kill conditions activate after trade {DATA_COLLECTION_TRADES}.</div>'
    )
else:
    _kill_detail_html = (
        f'<div class="snap-kill-detail snap-kill-clear">'
        f'All 5 conditions within parameters:<br>'
        f'Expectancy · Profit factor · Gap stop rate · Entry slippage · Regime inversion'
        f'</div>'
    )

_kill_html = (
    f'<div class="snap-cell">'
    f'  <div class="snap-label">KILL CONDITIONS</div>'
    f'  <details>'
    f'    <summary>'
    f'      <span class="snap-kill-pill" style="background:{_kpill_bg};color:{_kpill_col};border:1px solid {_kpill_bdr};">'
    f'        {_kpill_txt}'
    f'      </span>'
    f'    </summary>'
    f'    {_kill_detail_html}'
    f'  </details>'
    f'</div>'
)

st.markdown(
    f'<div class="snap-bar">{_regime_html}{_lastrun_html}{_pnl_html}{_heat_html}{_pos_html}{_kill_html}</div>',
    unsafe_allow_html=True,
)

# ── Cool-down alert (operational — stays below bar) ─────────────────────
refresh_placeholder = st.empty()

_cd_active    = cooldown_state.get("active", False)
_cd_remaining = cooldown_state.get("days_remaining", 0)
_cd_elapsed   = cooldown_state.get("days_elapsed", 0)
_cd_triggered = cooldown_state.get("triggered_date", "—")

if _cd_active and _cd_remaining > 0:
    st.markdown(
        f'<div style="background:#1a0505;border:2px solid #ef5350;border-radius:8px;'
        f'padding:12px 20px;margin-bottom:12px;">'
        f'<span style="color:#ef5350;font-weight:700;font-size:14px;">'
        f'⏳ COOL-DOWN: {_cd_remaining} trading day{"s" if _cd_remaining != 1 else ""} remaining</span>'
        f'<span style="color:#888;font-size:12px;"> · triggered {_cd_triggered} · '
        f'{_cd_elapsed}/{KILL_COOLDOWN_DAYS} days elapsed · '
        f'delete <code>results/kill_cooldown.json</code> to override</span>'
        f'</div>',
        unsafe_allow_html=True,
    )
elif _cd_active and _cd_remaining == 0:
    st.markdown(
        f'<div style="background:#0d1a0d;border:1px solid #26a69a55;border-radius:8px;'
        f'padding:10px 18px;margin-bottom:12px;">'
        f'<span style="color:#26a69a;font-weight:600;">✅ Cooldown expired</span>'
        f'<span style="color:#888;font-size:12px;"> · triggered {_cd_triggered} · '
        f'resolve kill conditions before resuming</span>'
        f'</div>',
        unsafe_allow_html=True,
    )

# ── Soft breach warning ──────────────────────────────────────────────────
_sb_days    = soft_breach.get("consecutive_days", 0)
_sb_metrics = soft_breach.get("metrics", [])
if _sb_days >= SOFT_BREACH_WARN_DAYS:
    metrics_str = ", ".join(_sb_metrics) if _sb_metrics else "multiple metrics"
    st.markdown(
        f'<div style="background:#1a1500;border:1px solid {AMBER};border-radius:8px;'
        f'padding:12px 18px;margin-bottom:12px;">'
        f'<span style="color:{AMBER};font-weight:700;">⚠️ SOFT BREACH: {_sb_days} consecutive days in amber</span>'
        f'<span style="color:#888;font-size:12px;"> · {metrics_str} drifting 15–30% from backtest</span>'
        f'<div style="color:#aaa;font-size:11px;margin-top:4px;">'
        f'Manual review recommended. Not a kill condition, but trend is deteriorating.</div>'
        f'</div>',
        unsafe_allow_html=True,
    )
elif _sb_days > 0 and n_closed >= DATA_COLLECTION_TRADES:
    metrics_str = ", ".join(_sb_metrics) if _sb_metrics else "metrics"
    st.markdown(
        f'<div class="phase-note" style="margin-bottom:10px;">'
        f'🟡 {_sb_days} day{"s" if _sb_days != 1 else ""} in amber ({metrics_str}) — '
        f'watch for {SOFT_BREACH_WARN_DAYS - _sb_days} more to trigger review</div>',
        unsafe_allow_html=True,
    )

# ===========================================================================
# ROW 1 — PORTFOLIO HEALTH
# ===========================================================================
st.markdown("## 📊 Portfolio Health")

# Equity curve + drawdown — full width so chart is readable on mobile
st.markdown("**Equity Curve — Live vs Backtest**")
st.plotly_chart(_chart_equity(live_eq, bt_eq),
                use_container_width=True, config={"displayModeBar": False})
st.plotly_chart(_chart_drawdown(live_eq),
                use_container_width=True, config={"displayModeBar": False})

# Account + Exposure tiles — 4 across on desktop, 2 across on mobile (CSS handles wrap)
st.markdown("**Account & Exposure**")
ph_c1, ph_c2, ph_c3, ph_c4 = st.columns(4)

with ph_c1:
    _metric("Balance", f"${current_bal:,.0f}",
            delta=f"{total_pnl:+,.0f} ({total_ret:+.1f}%)",
            delta_colour=GREEN if total_pnl >= 0 else RED)
with ph_c2:
    dd_clr = RED if current_dd < -3 else AMBER if current_dd < 0 else GREEN
    _metric("Drawdown", f"{current_dd:.1f}%",
            delta=f"Max {max_dd_live:.1f}% · BT −{BT['max_dd']}%",
            delta_colour=dd_clr)
with ph_c3:
    heat_clr = RED if heat_pct > HEAT_BUDGET * 0.8 else AMBER if heat_pct > HEAT_BUDGET * 0.5 else GREEN
    _metric("Heat %", f"{heat_pct:.1f}%",
            delta=f"{n_open}/{MAX_POSITIONS} slots · budget {HEAT_BUDGET}%",
            delta_colour=heat_clr)
with ph_c4:
    car_aud = cap_at_risk["max_loss_aud"]
    car_pct = cap_at_risk["max_loss_pct"]
    car_clr = RED if car_pct > 5 else AMBER if car_pct > 3 else GREEN
    _metric("Capital at Risk", f"−${car_aud:,.0f}",
            delta=f"−{car_pct:.1f}% if ALL stops hit",
            delta_colour=car_clr)

# ── Open Positions (confirmed entry fills) ───────────────────────────────
st.markdown(
    "**Open Positions** "
    "<span style='color:#888;font-size:11px;'>entry filled · consuming heat budget</span>",
    unsafe_allow_html=True,
)
if not open_df.empty:
    disp = open_df.copy()
    if "timestamp" in disp.columns:
        disp["Date"] = pd.to_datetime(disp["timestamp"]).dt.strftime("%d %b")
    if "risk_aud" in disp.columns:
        disp["Risk"] = disp["risk_aud"].map(lambda x: f"${x:.0f}")
    if "entry" in disp.columns:
        disp["Entry"] = disp["entry"].map(lambda x: f"${x:.3f}")
    if "stop_loss" in disp.columns:
        disp["Stop"] = disp["stop_loss"].map(lambda x: f"${x:.3f}")
    if "target" in disp.columns:
        disp["Target"] = disp["target"].map(lambda x: f"${x:.3f}")
    # Mobile-priority column order — most important first
    mobile_cols = [c for c in ["ticker","Risk","Entry","Stop","Target","Date","shares"]
                   if c in disp.columns]
    st.dataframe(disp[mobile_cols].rename(columns={"ticker":"Ticker","shares":"Shares"}),
                 use_container_width=True, hide_index=True)
else:
    st.caption("No filled positions.")

# ── Pending Orders (submitted, awaiting entry fill) ───────────────────────
st.markdown(
    "**Pending Orders** "
    "<span style='color:#888;font-size:11px;'>submitted · awaiting limit fill · "
    "not consuming heat budget</span>",
    unsafe_allow_html=True,
)
if not stale_df.empty:
    stale_refs = set(stale_df["order_ref"].dropna())
    st.markdown(
        f'<div style="background:#1a1200;border:1px solid {AMBER};border-radius:6px;'
        f'padding:8px 14px;margin-bottom:8px;">'
        f'<span style="color:{AMBER};font-weight:600;">⚠️ {len(stale_df)} STALE order'
        f'{"s" if len(stale_df) != 1 else ""} — waiting ≥{STALE_ORDER_DAYS} trading days without filling</span>'
        f'<div style="color:#aaa;font-size:11px;margin-top:3px;">'
        f'Tickers: {", ".join(str(t).replace(".AX","") for t in stale_df["ticker"].values)} — '
        f'consider cancelling in TWS if the setup has moved on</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

if not pending_df.empty:
    today = date.today()
    pend_disp = []
    for _, row in pending_df.iterrows():
        try:
            sub_date  = pd.to_datetime(row["timestamp"]).date()
            wait_days = int(np.busday_count(sub_date.isoformat(), today.isoformat()))
        except Exception:
            wait_days = 0
        is_stale  = wait_days >= STALE_ORDER_DAYS
        pend_disp.append({
            "Submitted":  pd.to_datetime(row["timestamp"]).strftime("%Y-%m-%d"),
            "Ticker":     str(row.get("ticker","—")),
            "Shares":     int(row.get("shares", 0)),
            "Limit":      f"{float(row['entry']):.3f}"    if "entry"     in row else "—",
            "Stop":       f"{float(row['stop_loss']):.3f}" if "stop_loss" in row else "—",
            "Target":     f"{float(row['target']):.3f}"   if "target"    in row else "—",
            "Risk":       f"${float(row['risk_aud']):.0f}" if "risk_aud"  in row else "—",
            "Waiting":    f"{wait_days}d",
            "Status":     "⚠️ STALE" if is_stale else "⏳ working",
        })
    pend_show = pd.DataFrame(pend_disp)

    def _style_pending(df):
        styles = pd.DataFrame("", index=df.index, columns=df.columns)
        for i, row in df.iterrows():
            if "STALE" in str(row["Status"]):
                styles.loc[i, "Status"]  = f"color:{AMBER};font-weight:600"
                styles.loc[i, "Waiting"] = f"color:{AMBER}"
            else:
                styles.loc[i, "Status"]  = f"color:{BLUE}"
        return styles

    st.dataframe(
        pend_show.style.apply(_style_pending, axis=None),
        use_container_width=True, hide_index=True,
    )
else:
    st.caption("No pending orders.")

# ===========================================================================
# ROW 2 — EDGE VALIDATION
# ===========================================================================
st.markdown(
    "## 🎯 Edge Validation  "
    "<span style='color:#888;font-size:12px;'>(last 30 closed trades)</span>",
    unsafe_allow_html=True,
)

def _safe_pct(v, bt_v, lower_is_better=False):
    """Return delta colour based on live vs BT with amber/red thresholds."""
    if _safe_nan(v) or bt_v == 0:
        return GREY
    drift = (v - bt_v) / abs(bt_v)
    if lower_is_better:
        drift = -drift
    return _drift_colour(drift)

# Row 1: core edge metrics — 4 across desktop, 2 across mobile
ev1, ev2, ev3, ev4 = st.columns(4)
with ev1:
    wr  = roll30["win_rate"]
    _metric("Win Rate",       f"{wr:.1f}%" if not _safe_nan(wr) else "—",
            delta=f"BT {BT['win_rate']}%", delta_colour=_safe_pct(wr, BT["win_rate"]))
with ev2:
    pf  = roll30["profit_factor"]
    _metric("Profit Factor",  f"{pf:.2f}" if not _safe_nan(pf) else "—",
            delta=f"BT {BT['profit_factor']}", delta_colour=_safe_pct(pf, BT["profit_factor"]))
with ev3:
    exp = roll30["expectancy"]
    _metric("Expectancy",     f"${exp:.0f}" if not _safe_nan(exp) else "—",
            delta=f"BT ${BT['expectancy']:.0f}", delta_colour=_safe_pct(exp, BT["expectancy"]))
with ev4:
    ar  = roll30["avg_r"]
    _metric("Avg R / Trade",  f"{ar:.3f}R" if not _safe_nan(ar) else "—",
            delta=f"BT {BT['avg_r']}R", delta_colour=_safe_pct(ar, BT["avg_r"]))

# Row 2: execution quality — 3 across desktop, wraps to mobile
ev5, ev6, ev7 = st.columns(3)
with ev5:
    slip_str = "0.00%" if has_paper else "N/A"
    _metric("Avg Entry Slip", slip_str,
            delta="BT assumption 0.20%",
            delta_colour=GREEN if has_paper else GREY)
with ev6:
    sigs   = last_run.get("signals_count") or 0
    placed = last_run.get("orders_placed") or 0
    _metric("Fill Rate (today)", f"{placed}/{sigs}" if sigs else "—",
            delta="signals → placed")
with ev7:
    if exec_latency_s is not None:
        lat_clr = RED if exec_latency_s > 120 else AMBER if exec_latency_s > 60 else GREEN
        lat_str = (f"{exec_latency_s:.0f}s"
                   if exec_latency_s < 60 else f"{exec_latency_s/60:.1f}m")
        _metric("Exec Latency", lat_str,
                delta="signal scan → first order",
                delta_colour=lat_clr)
    else:
        _metric("Exec Latency", "—",
                delta="signal scan → first order")

_phase_note()

# ===========================================================================
# MISSED TRADES TRACKER
# ===========================================================================
st.markdown("## 🔍 Missed Trades Tracker")
st.markdown(
    "<span style='color:#888;font-size:12px;'>Today's run · "
    "Hypothetical P&L uses backtest avg R × avg trade risk</span>",
    unsafe_allow_html=True,
)

# Row 1: funnel counts — 4 across desktop, 2 across mobile
mt1, mt2, mt3, mt4 = st.columns(4)
with mt1:
    _metric("Signals Generated", str(missed["generated"]),
            delta="RS + trigger scan")
with mt2:
    _metric("Eligible After Filters", str(missed["eligible"]),
            delta="Passed regime / momentum gate")
with mt3:
    blocked_by_regime = max(0, missed["generated"] - missed["eligible"])
    _metric("Blocked by Regime", str(blocked_by_regime),
            delta="CHOPPY_BEAR momentum filter",
            delta_colour=AMBER if blocked_by_regime > 0 else GREEN)
with mt4:
    _metric("Executed", str(missed["executed"]),
            delta="Orders placed",
            delta_colour=GREEN if missed["executed"] > 0 else GREY)

# Row 2: missed summary — single tile
mt5, mt5b = st.columns(2)
with mt5:
    hypo_str = (f"+{missed['hypo_r']:.2f}R / ${missed['hypo_pnl']:+,.0f}"
                if missed["missed"] > 0 else "—")
    _metric("Missed (cap/heat)", str(missed["missed"]),
            delta=f"Hypothetical: {hypo_str}",
            delta_colour=AMBER if missed["missed"] > 0 else GREEN)

# Missed trades breakdown by reason (parsed from log)
total_breakdown = sum(missed_breakdown.values())
if total_breakdown > 0:
    st.markdown(
        "<span style='color:#888;font-size:12px;font-style:italic;'>"
        "Missed breakdown by reason (from run log):</span>",
        unsafe_allow_html=True,
    )
    mb1, mb2, mb3, mb4 = st.columns(4)
    bd_colour = lambda n: AMBER if n > 0 else GREY
    with mb1:
        _metric("Blocked by Regime",
                str(missed_breakdown["regime_blocked"]),
                delta="momentum gate / f_regime filter",
                delta_colour=bd_colour(missed_breakdown["regime_blocked"]))
    with mb2:
        _metric("Blocked by Risk Engine",
                str(missed_breakdown["risk_blocked"]),
                delta="heat budget / sector cap / position limit",
                delta_colour=bd_colour(missed_breakdown["risk_blocked"]))
    with mb3:
        _metric("Execution Failure",
                str(missed_breakdown["exec_failure"]),
                delta="IBKR submission error",
                delta_colour=RED if missed_breakdown["exec_failure"] > 0 else GREY)
    with mb4:
        _metric("No Fill",
                str(missed_breakdown["no_fill"]),
                delta="order sent, fill not confirmed",
                delta_colour=RED if missed_breakdown["no_fill"] > 0 else GREY)

# ===========================================================================
# ROW 3 — SYSTEM DIAGNOSTICS
# ===========================================================================
st.markdown("## 🔬 System Diagnostics")

# ── Regime Matrix + Losing Streak side by side on desktop, stacked on mobile
sd1, sd2 = st.columns(2)

with sd1:
    st.markdown("**Regime Performance Matrix**")
    st.caption("Validates Variant F: CHOPPY_BEAR should underperform STRONG_BULL")
    if not regime_matrix.empty:
        def _style_regime(df):
            styles = pd.DataFrame("", index=df.index, columns=df.columns)
            for i, row in df.iterrows():
                regime_val = row["Regime"]
                colour = {"STRONG_BULL": GREEN, "WEAK_BULL": AMBER, "CHOPPY_BEAR": RED}.get(regime_val, GREY)
                styles.loc[i, "Regime"] = f"color: {colour}; font-weight: 600"
            return styles
        st.dataframe(
            regime_matrix.style.apply(_style_regime, axis=None),
            use_container_width=True, hide_index=True,
        )
    else:
        st.plotly_chart(_chart_regime_dist(bt_trades, merged),
                        use_container_width=True, config={"displayModeBar": False})
        st.caption("No live trades yet — showing backtest regime distribution.")

with sd2:
    st.markdown("**Losing Streak Tracker**")
    streak_clr = RED if cur_streak >= BT["max_consec_loss"] else AMBER if cur_streak >= 2 else GREEN
    _metric("Current Streak", f"{cur_streak} losses",
            delta=f"Max ever {max_streak} · BT max {BT['max_consec_loss']}",
            delta_colour=streak_clr)
    if not merged.empty and "pnl_r" in merged.columns:
        recent20 = merged.tail(20)["pnl_r"].values
        fig_str  = go.Figure()
        fig_str.add_trace(go.Bar(
            x=list(range(len(recent20))),
            y=recent20,
            marker_color=[GREEN if r > 0 else RED for r in recent20],
            showlegend=False,
        ))
        layout_s = dict(**PLOT_LAYOUT, height=160)
        layout_s["xaxis"] = dict(**PLOT_LAYOUT["xaxis"], showticklabels=False)
        layout_s["yaxis"] = dict(**PLOT_LAYOUT["yaxis"], ticksuffix="R")
        fig_str.update_layout(**layout_s)
        st.plotly_chart(fig_str, use_container_width=True, config={"displayModeBar": False})

# ── Slippage + Trade Lifecycle — 4 tiles across, 2-per-row on mobile
sd3a, sd3b, sd3c, sd3d = st.columns(4)

with sd3a:
    st.markdown("**Slippage**")
    entry_slip = "0.00%" if has_paper else "N/A"
    _metric("Entry Slip", entry_slip, delta="Paper: signal px = fill px")

with sd3b:
    es = slip_decomp["exit_slip_r"]
    _metric("Stop Slip",
            f"{es:+.3f}R" if not _safe_nan(es) else "—",
            delta="deviation from −1R",
            delta_colour=RED if (not _safe_nan(es) and es < -0.05) else GREEN)

with sd3c:
    ge = slip_decomp["gap_excess_r"]
    _metric("Gap Excess",
            f"{ge:+.3f}R" if not _safe_nan(ge) else "—",
            delta="gap stop avg vs −1R",
            delta_colour=RED if (not _safe_nan(ge) and ge < -0.1) else AMBER)

with sd3d:
    if not merged.empty:
        if "holding_days" in merged.columns:
            avg_hold = round(float(merged["holding_days"].mean()), 1)
        elif "exit_date" in merged.columns and "timestamp" in merged.columns:
            avg_hold = round(float(
                (pd.to_datetime(merged["exit_date"]) - pd.to_datetime(merged["timestamp"]))
                .dt.days.mean()
            ), 1)
        else:
            avg_hold = np.nan
        _metric("Avg Hold", f"{avg_hold:.1f}d" if not _safe_nan(avg_hold) else "—",
                delta=f"BT {BT['avg_hold_days']}d")
    else:
        _metric("Avg Hold", "—")

# ===========================================================================
# HEAT OVER TIME
# ===========================================================================
st.markdown("## 🌡️ Portfolio Heat Over Time")
st.markdown(
    "<span style='color:#888;font-size:12px;'>Daily open risk as % of account · "
    f"budget = {HEAT_BUDGET}%</span>",
    unsafe_allow_html=True,
)
if not heat_ts.empty:
    st.plotly_chart(_chart_heat_over_time(heat_ts),
                    use_container_width=True, config={"displayModeBar": False})
else:
    st.info("Heat history will build as trades are entered and exited.", icon="📊")

# ===========================================================================
# TRADE QUALITY BUCKETS
# ===========================================================================
st.markdown("## 🪣 Trade Quality Buckets")
st.markdown(
    "<span style='color:#888;font-size:12px;'>"
    "F-pass/C-pass = normal trades · F-pass/C-block = CHOPPY_BEAR momentum-gated trades · "
    "validates that Variant F's selective filter outperforms Variant C's hard block</span>",
    unsafe_allow_html=True,
)

if quality:
    bkt_rows = []
    for bkt, stats in quality.items():
        ar    = stats["avg_r"]
        pnl   = stats["pnl_aud"]
        bkt_rows.append({
            "Bucket":    bkt,
            "Trades":    stats["trades"],
            "Win %":     f"{stats['win_rate']:.1f}%",
            "Avg R":     f"{ar:.3f}" if not _safe_nan(ar) else "—",
            "Total P&L": f"${pnl:+,.0f}" if not _safe_nan(pnl) else "—",
        })
    bkt_df = pd.DataFrame(bkt_rows)

    def _style_bucket(df):
        styles = pd.DataFrame("", index=df.index, columns=df.columns)
        colours = {
            "F-pass / C-pass":  GREEN,
            "F-pass / C-block": AMBER,
            "Both block":       RED,
        }
        for i, row in df.iterrows():
            c = colours.get(row["Bucket"], GREY)
            styles.loc[i, "Bucket"] = f"color: {c}; font-weight: 600"
        return styles

    st.dataframe(
        bkt_df.style.apply(_style_bucket, axis=None),
        use_container_width=True, hide_index=True,
    )
    if not merged.empty and "f_regime" not in merged.columns and "regime" not in merged.columns:
        st.caption("⚠️ Regime column not found in trade log — bucket classification unavailable until regime data is stored at entry.")
else:
    st.info(
        "Bucket data builds as live trades close. Requires regime column in trade records.",
        icon="🪣",
    )

# ===========================================================================
# ROLLING EXPECTANCY CHART
# ===========================================================================
st.markdown("## 📈 Rolling Expectancy")
st.markdown(
    "<span style='color:#888;font-size:12px;'>"
    "Rolling avg R over 10 / 30 / 50 trades · "
    "converging toward BT baseline indicates system is behaving as expected</span>",
    unsafe_allow_html=True,
)

if roll_exp:
    st.plotly_chart(_chart_rolling_expectancy(roll_exp, n_closed),
                    use_container_width=True, config={"displayModeBar": False})
    if n_closed < DATA_COLLECTION_TRADES:
        st.markdown(
            '<div class="low-conf-note">⚠️ Rolling-10 line shown with reduced opacity — '
            f'low confidence until trade {DATA_COLLECTION_TRADES} '
            f'(currently {n_closed})</div>',
            unsafe_allow_html=True,
        )
    _phase_note()
else:
    st.info(
        f"Rolling expectancy requires at least 10 closed trades "
        f"(currently {n_closed}).",
        icon="📈",
    )

# ===========================================================================
# BACKTEST vs LIVE COMPARISON PANEL
# ===========================================================================
st.markdown("## 📐 Backtest vs Live Comparison")
st.markdown(
    "<span style='color:#888;font-size:12px;'>"
    f"🟡 Amber = 15–30% drift from Variant F baseline · "
    f"🔴 Red = >30% drift (investigate)</span>",
    unsafe_allow_html=True,
)


def _drift_row(metric_name, bt_val, live_val, fmt_fn, lower_is_better=False):
    if _safe_nan(live_val):
        return {
            "Metric": metric_name,
            "Backtest (Variant F)": fmt_fn(bt_val),
            "Live (actual)": "—",
            "Delta": "—",
            "Status": "—",
        }
    live_str = fmt_fn(live_val)
    if bt_val != 0:
        drift = (live_val - bt_val) / abs(bt_val)
        if lower_is_better:
            drift = -drift
        if drift < -DRIFT_THRESHOLD:
            status = "🔴 ALERT"
        elif drift < -DRIFT_WARN:
            status = "🟡 WATCH"
        else:
            status = "🟢 OK"
        delta_str = f"{drift:+.0%}"
    else:
        delta_str = "—"
        status    = "—"
    return {
        "Metric": metric_name,
        "Backtest (Variant F)": fmt_fn(bt_val),
        "Live (actual)": live_str,
        "Delta": delta_str,
        "Status": status,
    }


rows = [
    _drift_row("Win Rate %",       BT["win_rate"],       roll30["win_rate"],       lambda v: f"{v:.1f}%"),
    _drift_row("Avg R / Trade",    BT["avg_r"],          roll30["avg_r"],          lambda v: f"{v:.3f}R"),
    _drift_row("Profit Factor",    BT["profit_factor"],  roll30["profit_factor"],  lambda v: f"{v:.2f}"),
    _drift_row("Expectancy (AUD)", BT["expectancy"],     roll30["expectancy"],     lambda v: f"${v:.0f}"),
    _drift_row("Gap Stop %",       BT["gap_stop_pct"],   gs["pct"],               lambda v: f"{v:.1f}%",
               lower_is_better=True),
    _drift_row("Avg Entry Slip %", 0.20, 0.0 if has_paper else None, lambda v: f"{v:.2f}%",
               lower_is_better=True),
    _drift_row("Max Consec. Loss", float(BT["max_consec_loss"]),
               float(max_streak) if max_streak else np.nan,
               lambda v: f"{int(v)}", lower_is_better=True),
]

comp_df = pd.DataFrame(rows)

def _style_comp(df):
    styles = pd.DataFrame("", index=df.index, columns=df.columns)
    for i, row in df.iterrows():
        s = row["Status"]
        if "🔴" in str(s):
            c = RED
        elif "🟡" in str(s):
            c = AMBER
        elif "🟢" in str(s):
            c = GREEN
        else:
            c = GREY
        styles.loc[i, "Status"] = f"color: {c}; font-weight: 600"
        styles.loc[i, "Delta"]  = f"color: {c}"
    return styles

st.dataframe(
    comp_df.style.apply(_style_comp, axis=None),
    use_container_width=True, hide_index=True,
)
_phase_note()

# ===========================================================================
# FULL TRADE LOG
# ===========================================================================
st.markdown("## 📋 Full Trade Log")

if merged.empty and open_df.empty:
    st.info("No trades recorded yet. Trades will appear here after the first live run.", icon="📭")
else:
    log_rows = []
    sort_col = "exit_date" if "exit_date" in merged.columns else (merged.columns[0] if not merged.empty else None)

    for _, row in (merged.sort_values(sort_col, ascending=False)
                   if sort_col else merged).iterrows():
        entry_dt  = pd.to_datetime(row.get("timestamp","")).strftime("%Y-%m-%d") if pd.notna(row.get("timestamp")) else "—"
        exit_dt   = pd.to_datetime(row.get("exit_date","")).strftime("%Y-%m-%d")  if pd.notna(row.get("exit_date"))  else "—"
        pnl_r     = row.get("pnl_r",   np.nan)
        pnl_aud   = row.get("pnl_aud", np.nan)
        regime_e  = row.get("f_regime", row.get("regime", "—"))
        fill_px   = f"{row['entry']:.3f}" if "entry" in row and pd.notna(row.get("entry")) else "—"
        log_rows.append({
            # Mobile-priority columns first (shown even when table is narrow)
            "Ticker":    row.get("ticker", "—"),
            "R Result":  f"{pnl_r:+.2f}R" if pd.notna(pnl_r) else "—",
            "Exit Type": row.get("exit_type", "—"),
            "Date":      entry_dt,
            # Secondary columns (scroll right on mobile)
            "Exit":      exit_dt,
            "P&L":       f"${pnl_aud:+,.0f}" if pd.notna(pnl_aud) else "—",
            "Fill Px":   fill_px,
            "Regime":    regime_e,
            "_status":   "CLOSED",
        })

    for _, row in open_df.iterrows():
        entry_dt = pd.to_datetime(row.get("timestamp","")).strftime("%Y-%m-%d") if pd.notna(row.get("timestamp")) else "—"
        fill_px  = f"{row['entry']:.3f}" if "entry" in row and pd.notna(row.get("entry")) else "—"
        log_rows.append({
            "Ticker":    row.get("ticker", "—"),
            "R Result":  "OPEN",
            "Exit Type": "—",
            "Date":      entry_dt,
            "Exit":      "—",
            "P&L":       "—",
            "Fill Px":   fill_px,
            "Regime":    "—",
            "_status":   "OPEN",
        })

    log_df = pd.DataFrame(log_rows)

    def _style_log(row):
        if row["_status"] == "OPEN":
            return [f"color: {BLUE}"] * len(row)
        r = row["R Result"]
        if str(r).startswith("+"):
            return [f"color: {GREEN}"] * len(row)
        if str(r).startswith("-"):
            return [f"color: {RED}"] * len(row)
        return [""] * len(row)

    # Apply style on log_df (which contains _status), then hide _status column
    st.dataframe(
        log_df.style.apply(_style_log, axis=1).hide(subset=["_status"], axis="columns"),
        use_container_width=True, hide_index=True,
    )

# ===========================================================================
# PIPELINE STATUS FOOTER
# ===========================================================================
if last_run:
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("**Pipeline Status** — last run", unsafe_allow_html=True)
    step_cols  = st.columns(6)
    step_names = ["regime", "screener", "signals", "risk", "ibkr", "exit_logger"]
    step_labels= ["Regime", "Screener", "Signals", "Risk", "IBKR", "Exit Logger"]
    icons      = {"ok": "✅", "warn": "⚠️", "error": "❌", "skip": "⏭️", None: "⬜"}
    for col, name, label in zip(step_cols, step_names, step_labels):
        status = last_run.get("steps", {}).get(name)
        col.markdown(
            f"{icons.get(status,'⬜')} **{label}**<br/>"
            f"<span style='color:#888;font-size:11px;'>{status or 'not run'}</span>",
            unsafe_allow_html=True,
        )

# ===========================================================================
# AUTO-REFRESH
# ===========================================================================
refresh_placeholder.markdown(
    f"<div style='text-align:right;color:#555;font-size:11px;margin-top:20px;'>"
    f"↻ auto-refresh {REFRESH_SECS//60}min</div>",
    unsafe_allow_html=True,
)
time.sleep(REFRESH_SECS)
st.rerun()
