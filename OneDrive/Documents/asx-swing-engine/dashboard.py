"""
dashboard.py  --  ASX Swing Engine daily monitoring dashboard
-------------------------------------------------------------
Run with:  streamlit run dashboard.py

Sections
  1. Today's signals        -- results/signals_output.csv
  2. Open trades            -- logs/trades.csv  +  live prices via yfinance
  3. Performance summary    -- derived from logs/trades.csv closed trades
  4. Equity curve           -- account balance over time
  5. Top screener stocks    -- results/screener_output.csv  top 10

Auto-refreshes every 5 minutes.
"""

import math
from datetime import datetime, date
from pathlib import Path

import pandas as pd
import numpy as np
import streamlit as st
import yfinance as yf

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="ASX Swing Engine",
    page_icon="K",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ---------------------------------------------------------------------------
# Dark theme CSS
# ---------------------------------------------------------------------------
st.markdown("""
<style>
  /* ── global background ── */
  .stApp { background-color: #0d0d0d; color: #e0e0e0; }
  section[data-testid="stSidebar"] { background-color: #111111; }

  /* ── metric tiles ── */
  [data-testid="stMetric"] {
      background: #161616;
      border: 1px solid #2a2a2a;
      border-radius: 8px;
      padding: 14px 18px;
  }
  [data-testid="stMetricLabel"] { color: #888 !important; font-size: 12px !important; }
  [data-testid="stMetricValue"] { color: #e0e0e0 !important; font-size: 22px !important; }
  [data-testid="stMetricDelta"] { font-size: 13px !important; }

  /* ── dataframe tables ── */
  [data-testid="stDataFrame"] { border: 1px solid #2a2a2a; border-radius: 6px; }
  .stDataFrame thead th {
      background-color: #1e1e1e !important;
      color: #888 !important;
      font-size: 11px !important;
      font-weight: 600 !important;
      text-transform: uppercase;
      letter-spacing: 0.05em;
  }

  /* ── section headers ── */
  h2 { color: #f5a623 !important; font-size: 16px !important;
       border-bottom: 1px solid #2a2a2a; padding-bottom: 6px; margin-top: 28px !important; }
  h3 { color: #e0e0e0 !important; font-size: 13px !important; }

  /* ── pill badges ── */
  .pill {
      display: inline-block;
      padding: 2px 8px;
      border-radius: 10px;
      font-size: 11px;
      font-weight: 600;
  }
  .pill-green { background: #0d2b1e; color: #26a69a; border: 1px solid #26a69a44; }
  .pill-red   { background: #2b0d0d; color: #ef5350; border: 1px solid #ef535044; }
  .pill-amber { background: #2b1e0d; color: #f5a623; border: 1px solid #f5a62344; }

  /* ── divider ── */
  hr { border-color: #2a2a2a; }

  /* ── hide streamlit chrome ── */
  #MainMenu, footer, header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SIGNALS_CSV   = Path("results/signals_output.csv")
TRADES_CSV    = Path("logs/trades.csv")
SCREENER_CSV  = Path("results/screener_output.csv")
REFRESH_SECS  = 300   # 5 minutes

GREEN = "#26a69a"
RED   = "#ef5350"
AMBER = "#f5a623"
GREY  = "#888888"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _colour(val: float, good_positive: bool = True) -> str:
    if val > 0:
        return GREEN if good_positive else RED
    if val < 0:
        return RED if good_positive else GREEN
    return GREY


def _fmt_pct(v: float) -> str:
    return f"{v:+.1f}%" if not math.isnan(v) else "n/a"


def _fmt_aud(v: float) -> str:
    return f"${v:,.0f}" if not math.isnan(v) else "n/a"


@st.cache_data(ttl=REFRESH_SECS)
def _fetch_prices(tickers: list[str]) -> dict[str, float]:
    """Fetch latest close prices for a list of tickers."""
    if not tickers:
        return {}
    prices = {}
    try:
        raw = yf.download(tickers, period="2d", interval="1d",
                          auto_adjust=True, progress=False, threads=True)
        close = raw["Close"]
        if isinstance(close, pd.Series):
            close = close.to_frame(name=tickers[0])
        for t in tickers:
            if t in close.columns:
                last = close[t].dropna()
                if not last.empty:
                    prices[t] = float(last.iloc[-1])
    except Exception:
        pass
    return prices


@st.cache_data(ttl=REFRESH_SECS)
def _load_signals() -> pd.DataFrame:
    if not SIGNALS_CSV.exists():
        return pd.DataFrame()
    return pd.read_csv(SIGNALS_CSV)


@st.cache_data(ttl=REFRESH_SECS)
def _load_trades() -> pd.DataFrame:
    if not TRADES_CSV.exists():
        return pd.DataFrame()
    df = pd.read_csv(TRADES_CSV, parse_dates=["timestamp"])
    return df


@st.cache_data(ttl=REFRESH_SECS)
def _load_screener() -> pd.DataFrame:
    if not SCREENER_CSV.exists():
        return pd.DataFrame()
    return pd.read_csv(SCREENER_CSV)


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
col_title, col_refresh = st.columns([5, 1])
with col_title:
    st.markdown(
        f"<h1 style='color:#f5a623;font-size:24px;margin:0'>K ASX Swing Engine</h1>"
        f"<p style='color:#555;font-size:12px;margin:2px 0 0'>"
        f"Last loaded: {datetime.now().strftime('%a %d %b %Y  %H:%M:%S')} "
        f"&nbsp;·&nbsp; Auto-refresh every 5 min</p>",
        unsafe_allow_html=True,
    )
with col_refresh:
    if st.button("Refresh now", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

st.markdown("---")

# ---------------------------------------------------------------------------
# Auto-refresh via meta tag injection
# ---------------------------------------------------------------------------
st.markdown(
    f"<meta http-equiv='refresh' content='{REFRESH_SECS}'>",
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# 1. Today's signals
# ---------------------------------------------------------------------------
st.markdown("## Today's Signals")

signals = _load_signals()

if signals.empty:
    st.info("No signals file found at results/signals_output.csv - run signals.py first.")
else:
    # Trigger badge column
    def _trigger_badges(row):
        badges = []
        if row.get("t_rsi_bounce"):  badges.append('<span class="pill pill-green">RSI</span>')
        if row.get("t_macd_cross"):  badges.append('<span class="pill pill-amber">MACD</span>')
        if row.get("t_vol_break"):   badges.append('<span class="pill pill-green">VOL</span>')
        return " ".join(badges)

    display_sig = signals[[
        "rank", "ticker", "entry", "stop_loss", "target",
        "trigger_count", "composite_score", "rs_vs_xjo", "momentum_20d",
        "t_rsi_bounce", "t_macd_cross", "t_vol_break",
    ]].copy()

    display_sig["risk $"] = ((display_sig["entry"] - display_sig["stop_loss"])
                              .apply(lambda x: f"${x:.3f}"))
    display_sig["r:r"]    = (
        (display_sig["target"] - display_sig["entry"])
        / (display_sig["entry"] - display_sig["stop_loss"])
    ).apply(lambda x: f"{x:.1f}:1")

    # Build HTML table
    rows_html = ""
    for _, r in display_sig.iterrows():
        badges = _trigger_badges(r)
        score_c = "#26a69a" if r["composite_score"] >= 70 else (
                  "#f5a623" if r["composite_score"] >= 50 else "#ef5350")
        rs_c  = _colour(r["rs_vs_xjo"])
        mom_c = _colour(r["momentum_20d"])
        rows_html += f"""
        <tr style="border-bottom:1px solid #1e1e1e">
          <td style="padding:8px 10px;color:#555">{int(r['rank'])}</td>
          <td style="padding:8px 10px;font-weight:700;color:#e0e0e0">{r['ticker'].replace('.AX','')}</td>
          <td style="padding:8px 10px;text-align:right">${r['entry']:.3f}</td>
          <td style="padding:8px 10px;text-align:right;color:{RED}">${r['stop_loss']:.3f}</td>
          <td style="padding:8px 10px;text-align:right;color:{GREEN}">${r['target']:.3f}</td>
          <td style="padding:8px 10px;text-align:center;color:{score_c};font-weight:700">{r['composite_score']:.1f}</td>
          <td style="padding:8px 10px;text-align:right;color:{rs_c}">{r['rs_vs_xjo']:+.1f}%</td>
          <td style="padding:8px 10px;text-align:right;color:{mom_c}">{r['momentum_20d']:+.1f}%</td>
          <td style="padding:8px 10px">{badges}</td>
        </tr>"""

    st.markdown(f"""
    <table style="width:100%;border-collapse:collapse;font-size:13px;background:#111111;border-radius:8px;overflow:hidden">
      <thead>
        <tr style="background:#1e1e1e;color:#666;font-size:11px;text-transform:uppercase;letter-spacing:0.05em">
          <th style="padding:10px;text-align:left">#</th>
          <th style="padding:10px;text-align:left">Ticker</th>
          <th style="padding:10px;text-align:right">Entry</th>
          <th style="padding:10px;text-align:right">Stop</th>
          <th style="padding:10px;text-align:right">Target</th>
          <th style="padding:10px;text-align:center">Score</th>
          <th style="padding:10px;text-align:right">RS vs XJO</th>
          <th style="padding:10px;text-align:right">Mom 20d</th>
          <th style="padding:10px;text-align:left">Triggers</th>
        </tr>
      </thead>
      <tbody>{rows_html}</tbody>
    </table>
    """, unsafe_allow_html=True)

st.markdown("---")

# ---------------------------------------------------------------------------
# 2. Open trades  (deduplicate: keep latest row per ticker per day)
# ---------------------------------------------------------------------------
st.markdown("## Open Trades")

trades_raw = _load_trades()

if trades_raw.empty:
    st.info("No trades logged yet - logs/trades.csv is empty.")
else:
    # Keep latest submission per ticker (most recent order_ref wins)
    trades = (trades_raw
              .sort_values("timestamp")
              .drop_duplicates(subset="ticker", keep="last")
              .copy())

    tickers = trades["ticker"].tolist()
    live_prices = _fetch_prices(tickers)

    rows_open = ""
    total_unrealised = 0.0

    for _, r in trades.iterrows():
        tk      = r["ticker"]
        symbol  = tk.replace(".AX", "")
        ep      = r["entry"]
        sl      = r["stop_loss"]
        tgt     = r["target"]
        shares  = r["shares"]
        risk    = r["risk_aud"]

        last    = live_prices.get(tk, np.nan)
        if not math.isnan(last):
            pnl_aud = (last - ep) * shares
            pnl_r   = (last - ep) / (ep - sl) if (ep - sl) != 0 else 0.0
            pnl_c   = _colour(pnl_aud)
            price_s = f"${last:.3f}"
            pnl_s   = f'<span style="color:{pnl_c}">{_fmt_aud(pnl_aud)} ({pnl_r:+.2f}R)</span>'
            total_unrealised += pnl_aud
        else:
            price_s = "<span style='color:#555'>n/a</span>"
            pnl_s   = "<span style='color:#555'>--</span>"

        dist_sl  = (last - sl)  / sl  * 100 if not math.isnan(last) else np.nan
        dist_tgt = (tgt - last) / last * 100 if not math.isnan(last) else np.nan

        ts_date = pd.to_datetime(r["timestamp"]).strftime("%d %b")
        mode    = '<span class="pill pill-amber">PAPER</span>' if r.get("paper") else \
                  '<span class="pill pill-red">LIVE</span>'

        rows_open += f"""
        <tr style="border-bottom:1px solid #1e1e1e">
          <td style="padding:8px 10px;font-weight:700;color:#e0e0e0">{symbol}</td>
          <td style="padding:8px 10px;color:#888">{ts_date}</td>
          <td style="padding:8px 10px;text-align:right">${ep:.3f}</td>
          <td style="padding:8px 10px;text-align:right">{price_s}</td>
          <td style="padding:8px 10px;text-align:right">{pnl_s}</td>
          <td style="padding:8px 10px;text-align:right;color:{RED}">${sl:.3f}</td>
          <td style="padding:8px 10px;text-align:right;color:{GREEN}">${tgt:.3f}</td>
          <td style="padding:8px 10px;text-align:right">{shares:,}</td>
          <td style="padding:8px 10px;text-align:center">{mode}</td>
        </tr>"""

    unrealised_c = _colour(total_unrealised)
    summary_line = (f'<p style="text-align:right;color:{unrealised_c};font-size:13px;margin:6px 0 0">'
                    f'Unrealised P&amp;L: <strong>{_fmt_aud(total_unrealised)}</strong></p>')

    st.markdown(f"""
    <table style="width:100%;border-collapse:collapse;font-size:13px;background:#111111;border-radius:8px;overflow:hidden">
      <thead>
        <tr style="background:#1e1e1e;color:#666;font-size:11px;text-transform:uppercase;letter-spacing:0.05em">
          <th style="padding:10px;text-align:left">Ticker</th>
          <th style="padding:10px;text-align:left">Date</th>
          <th style="padding:10px;text-align:right">Entry</th>
          <th style="padding:10px;text-align:right">Last</th>
          <th style="padding:10px;text-align:right">Unreal. P&amp;L</th>
          <th style="padding:10px;text-align:right">Stop</th>
          <th style="padding:10px;text-align:right">Target</th>
          <th style="padding:10px;text-align:right">Shares</th>
          <th style="padding:10px;text-align:center">Mode</th>
        </tr>
      </thead>
      <tbody>{rows_open}</tbody>
    </table>
    {summary_line}
    """, unsafe_allow_html=True)

st.markdown("---")

# ---------------------------------------------------------------------------
# 3. Performance summary  +  4. Equity curve   (side by side)
# ---------------------------------------------------------------------------
col_perf, col_eq = st.columns([1, 2], gap="large")

with col_perf:
    st.markdown("## Performance")

    if trades_raw.empty:
        st.info("No trade data yet.")
    else:
        df = trades_raw.copy()

        # Deduplicate: one entry per order_ref (last row wins)
        df = df.sort_values("timestamp").drop_duplicates("order_ref", keep="last")

        # We don't have exit prices in trades.csv (executor logs entries only).
        # Use risk_aud as a proxy: each closed win = +2R, loss = -1R for now.
        # When a proper exit log is added this section will auto-update.
        n_trades  = len(df)
        n_paper   = int(df["paper"].sum()) if "paper" in df.columns else n_trades
        total_risk = df["risk_aud"].sum()
        avg_score  = df["composite_score"].mean()

        # Summary metrics
        m1, m2 = st.columns(2)
        m1.metric("Positions logged", n_trades)
        m2.metric("Paper trades", n_paper)

        m3, m4 = st.columns(2)
        m3.metric("Total risk deployed", f"${total_risk:,.0f}")
        m4.metric("Avg signal score", f"{avg_score:.1f}")

        st.markdown(
            "<p style='color:#555;font-size:11px;margin-top:12px'>"
            "P&amp;L metrics will populate once exit logging is wired in."
            "</p>",
            unsafe_allow_html=True,
        )

with col_eq:
    st.markdown("## Equity Curve")

    # Look for the most recent equity CSV in results/backtest/
    eq_files = sorted(Path("results/backtest").glob("equity_rs_top_20pct_*.csv")) if \
               Path("results/backtest").exists() else []

    if not eq_files:
        st.info("No equity curve data found. Run python backtest.py to generate one.")
    else:
        eq_path = eq_files[-1]
        eq = pd.read_csv(eq_path, index_col=0, parse_dates=True)
        eq_col = "mark_to_market" if "mark_to_market" in eq.columns else eq.columns[0]
        eq_series = eq[eq_col].dropna()

        # Build chart dataframe
        chart_df = eq_series.rename("Account ($AUD)").reset_index()
        chart_df.columns = ["Date", "Account ($AUD)"]

        start = float(eq_series.iloc[0])
        end   = float(eq_series.iloc[-1])
        ret   = (end / start - 1) * 100

        ret_c = _colour(ret)
        st.markdown(
            f"<p style='color:#888;font-size:12px;margin:0 0 8px'>"
            f"5-year backtest  &nbsp;·&nbsp;  "
            f"${start:,.0f} &rarr; "
            f"<strong style='color:{ret_c}'>${end:,.0f}</strong>  "
            f"(<span style='color:{ret_c}'>{ret:+.1f}%</span>)"
            f"</p>",
            unsafe_allow_html=True,
        )

        st.line_chart(
            chart_df.set_index("Date"),
            color=GREEN,
            use_container_width=True,
            height=260,
        )

st.markdown("---")

# ---------------------------------------------------------------------------
# 5. Top screener stocks
# ---------------------------------------------------------------------------
st.markdown("## Top Screener Stocks")

screener = _load_screener()

if screener.empty:
    st.info("No screener data found. Run python screener.py first.")
else:
    top10 = screener.head(10).copy()

    rows_scr = ""
    for _, r in top10.iterrows():
        score_c = "#26a69a" if r["composite_score"] >= 70 else (
                  "#f5a623" if r["composite_score"] >= 50 else "#ef5350")
        rs_c  = _colour(r["rs_vs_xjo"])
        mom_c = _colour(r["momentum_20d"])
        rows_scr += f"""
        <tr style="border-bottom:1px solid #1e1e1e">
          <td style="padding:8px 10px;color:#555">{int(r['rank'])}</td>
          <td style="padding:8px 10px;font-weight:700;color:#e0e0e0">{r['ticker'].replace('.AX','')}</td>
          <td style="padding:8px 10px;text-align:right">${r['last_price']:.2f}</td>
          <td style="padding:8px 10px;text-align:right">{r['market_cap_m']:,.0f}M</td>
          <td style="padding:8px 10px;text-align:center;font-weight:700;color:{score_c}">{r['composite_score']:.1f}</td>
          <td style="padding:8px 10px;text-align:right;color:{rs_c}">{r['rs_vs_xjo']:+.1f}%</td>
          <td style="padding:8px 10px;text-align:right;color:{mom_c}">{r['momentum_20d']:+.1f}%</td>
          <td style="padding:8px 10px;text-align:right;color:#888">{r['atr_pct']:.1f}%</td>
          <td style="padding:8px 10px;text-align:right;color:#888">{r['vol_ratio']:.2f}x</td>
          <td style="padding:8px 10px;text-align:right;color:#888">{r['rsi_14']:.0f}</td>
        </tr>"""

    scr_date = ""
    if SCREENER_CSV.exists():
        mtime = datetime.fromtimestamp(SCREENER_CSV.stat().st_mtime)
        scr_date = f" &nbsp;·&nbsp; last run {mtime.strftime('%a %d %b %H:%M')}"

    st.markdown(f"""
    <p style="color:#555;font-size:12px;margin:0 0 8px">{len(screener)} stocks passed all filters{scr_date}</p>
    <table style="width:100%;border-collapse:collapse;font-size:13px;background:#111111;border-radius:8px;overflow:hidden">
      <thead>
        <tr style="background:#1e1e1e;color:#666;font-size:11px;text-transform:uppercase;letter-spacing:0.05em">
          <th style="padding:10px;text-align:left">#</th>
          <th style="padding:10px;text-align:left">Ticker</th>
          <th style="padding:10px;text-align:right">Price</th>
          <th style="padding:10px;text-align:right">Mkt Cap</th>
          <th style="padding:10px;text-align:center">Score</th>
          <th style="padding:10px;text-align:right">RS vs XJO</th>
          <th style="padding:10px;text-align:right">Mom 20d</th>
          <th style="padding:10px;text-align:right">ATR%</th>
          <th style="padding:10px;text-align:right">Vol Ratio</th>
          <th style="padding:10px;text-align:right">RSI</th>
        </tr>
      </thead>
      <tbody>{rows_scr}</tbody>
    </table>
    """, unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.markdown(
    "<p style='color:#333;font-size:11px;text-align:center;margin-top:36px'>"
    "ASX Swing Engine &nbsp;·&nbsp; paper mode &nbsp;·&nbsp; "
    "not financial advice &nbsp;·&nbsp; data: Yahoo Finance"
    "</p>",
    unsafe_allow_html=True,
)
