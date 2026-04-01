"""
utils/emailer.py
----------------
Build a dark-themed HTML email from screener results and send it
via SMTP (defaults to Outlook/Hotmail — works for any STARTTLS server).

Credentials are read from environment variables (loaded via python-dotenv
in run_daily.py):
    EMAIL_FROM      sender address  (e.g. joshbendavis@hotmail.com)
    EMAIL_PASSWORD  app password
    EMAIL_TO        recipient       (defaults to EMAIL_FROM if not set)
    SMTP_HOST       default: smtp-mail.outlook.com
    SMTP_PORT       default: 587
"""

import os
import smtplib
from datetime import datetime
from email.mime.image import MIMEImage
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

import pandas as pd


# ── colour helpers ─────────────────────────────────────────────────────────────

def _score_colour(v: float) -> str:
    if v >= 75: return "#1a9450"
    if v >= 60: return "#91cf60"
    if v >= 45: return "#fee08b"
    return "#fc8d59"


def _delta_colour(v: float) -> str:
    return "#26a69a" if v >= 0 else "#ef5350"


# ── HTML builder ───────────────────────────────────────────────────────────────

def build_html(
    results_df: pd.DataFrame,
    run_date: str,
    chart_cids: list[tuple[str, str]],   # [(ticker, content-id), ...]
) -> str:

    top20 = results_df.head(20)

    # ── table rows ────────────────────────────────────────────────────────────
    rows_html = ""
    for rank, row in top20.iterrows():
        sc  = _score_colour(row["composite_score"])
        mc  = _delta_colour(row["momentum_20d"])
        rc  = _delta_colour(row["rs_vs_xjo"])
        bgr = "#161616" if rank % 2 == 0 else "#111111"

        rows_html += f"""
        <tr style="background:{bgr}">
          <td style="text-align:center;color:#666;padding:7px 5px">{rank}</td>
          <td style="padding:7px 8px"><strong style="color:#e0e0e0">{row['ticker'].replace('.AX','')}</strong></td>
          <td style="text-align:right;padding:7px 8px">${row['last_price']:.2f}</td>
          <td style="text-align:right;padding:7px 8px">${row['market_cap_m']:,.0f}M</td>
          <td style="text-align:center;padding:7px 8px;font-weight:bold;color:{sc}">{row['composite_score']:.1f}</td>
          <td style="text-align:right;padding:7px 8px;color:{rc}">{row['rs_vs_xjo']:+.1f}%</td>
          <td style="text-align:right;padding:7px 8px;color:{mc}">{row['momentum_20d']:+.1f}%</td>
          <td style="text-align:right;padding:7px 8px">{row['atr_pct']:.1f}%</td>
          <td style="text-align:right;padding:7px 8px">{row['vol_ratio']:.2f}x</td>
          <td style="text-align:center;padding:7px 8px">{row['rsi_14']:.0f}</td>
          <td style="text-align:right;padding:7px 8px">{row['ema50_dist_pct']:+.1f}%</td>
        </tr>"""

    # ── inline charts ─────────────────────────────────────────────────────────
    charts_html = ""
    if chart_cids:
        charts_html = """
        <h2 style="color:#ddd;margin-top:36px;border-bottom:1px solid #333;padding-bottom:8px">
            📈 Top 10 Charts
        </h2>"""
        for ticker, cid in chart_cids:
            charts_html += f"""
        <p style="color:#aaa;font-size:12px;margin:12px 0 4px">{ticker}</p>
        <img src="cid:{cid}"
             style="width:100%;max-width:920px;border-radius:6px;
                    margin-bottom:20px;display:block">"""

    total   = len(results_df)
    top1    = results_df.iloc[0]
    top1_nm = top1["ticker"].replace(".AX", "")

    return f"""<!DOCTYPE html>
<html lang="en">
<head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"></head>
<body style="background:#0d0d0d;color:#e0e0e0;font-family:Arial,sans-serif;
             padding:28px 24px;max-width:980px;margin:auto">

  <!-- header -->
  <table style="width:100%;margin-bottom:24px">
    <tr>
      <td>
        <h1 style="margin:0;color:#f5a623;font-size:22px">🦘 ASX Swing Engine</h1>
        <p  style="margin:4px 0 0;color:#666;font-size:13px">{run_date}</p>
      </td>
      <td style="text-align:right;vertical-align:top">
        <span style="background:#1e1e1e;border:1px solid #333;border-radius:6px;
                     padding:6px 14px;color:#aaa;font-size:12px">
          {total} stocks passed all filters
        </span>
      </td>
    </tr>
  </table>

  <!-- top pick callout -->
  <div style="background:#1a1a1a;border-left:3px solid #f5a623;
              border-radius:6px;padding:12px 16px;margin-bottom:24px">
    <span style="color:#888;font-size:12px">⭐ Top pick today</span><br>
    <span style="font-size:18px;font-weight:bold;color:#f5a623">{top1_nm}</span>
    &nbsp;
    <span style="color:#aaa;font-size:13px">
      Score {top1['composite_score']:.1f} &nbsp;·&nbsp;
      RS vs XJO {top1['rs_vs_xjo']:+.1f}% &nbsp;·&nbsp;
      Mom 20d {top1['momentum_20d']:+.1f}% &nbsp;·&nbsp;
      Last ${top1['last_price']:.2f}
    </span>
  </div>

  <!-- results table -->
  <h2 style="color:#ddd;font-size:15px;margin-bottom:10px">
      Top 20 — Ranked by Composite Score
  </h2>
  <table style="border-collapse:collapse;width:100%;font-size:12px">
    <thead>
      <tr style="background:#1e1e1e;color:#888;border-bottom:1px solid #333">
        <th style="padding:8px 5px">#</th>
        <th style="padding:8px 8px;text-align:left">Ticker</th>
        <th style="padding:8px 8px;text-align:right">Price</th>
        <th style="padding:8px 8px;text-align:right">Mkt Cap</th>
        <th style="padding:8px 8px;text-align:center">Score</th>
        <th style="padding:8px 8px;text-align:right">RS vs XJO</th>
        <th style="padding:8px 8px;text-align:right">Mom 20d</th>
        <th style="padding:8px 8px;text-align:right">ATR%</th>
        <th style="padding:8px 8px;text-align:right">Vol Ratio</th>
        <th style="padding:8px 8px;text-align:center">RSI</th>
        <th style="padding:8px 8px;text-align:right">EMA50 Dist</th>
      </tr>
    </thead>
    <tbody>{rows_html}
    </tbody>
  </table>

  {charts_html}

  <!-- footer -->
  <p style="color:#333;font-size:11px;margin-top:36px;border-top:1px solid #1e1e1e;padding-top:12px">
    Filters: Price &gt; $0.50 &nbsp;|&nbsp; Avg Vol &gt; 200K &nbsp;|&nbsp;
    Mkt Cap &gt; $100M &nbsp;|&nbsp; Above 50 &amp; 200 EMA &nbsp;|&nbsp;
    RSI 40–65 &nbsp;|&nbsp; Within 10% of EMA50<br>
    Score weights: RS vs XJO 35% &nbsp;|&nbsp; Momentum 25% &nbsp;|&nbsp;
    ATR Sweet Spot 20% &nbsp;|&nbsp; Volume Ratio 20%<br>
    Data: Yahoo Finance via yfinance &nbsp;·&nbsp; Universe: ASX Listed Companies CSV
  </p>

</body>
</html>"""


# ── sender ─────────────────────────────────────────────────────────────────────

def send_email(
    results_df: pd.DataFrame,
    chart_paths: list[tuple[str, str]],   # [(ticker, filepath), ...]
    smtp_host:  str,
    smtp_port:  int,
    from_addr:  str,
    password:   str,
    to_addr:    str,
) -> None:
    """
    Build and dispatch the HTML email with inline chart images.

    Parameters
    ----------
    results_df  : full screener results DataFrame
    chart_paths : list of (ticker, png_filepath) from generate_charts()
    smtp_host   : SMTP server hostname
    smtp_port   : SMTP port (usually 587 for STARTTLS)
    from_addr   : sender email address
    password    : SMTP / app password
    to_addr     : recipient email address
    """
    run_date = datetime.today().strftime("%A %d %B %Y")
    subject  = (
        f"ASX Swing Engine — {datetime.today().strftime('%d %b %Y')}"
        f" | #{1} {results_df.iloc[0]['ticker'].replace('.AX','')} "
        f"Score {results_df.iloc[0]['composite_score']:.0f}"
    )

    # Content-IDs: safe string derived from ticker
    def _cid(ticker: str) -> str:
        return ticker.replace(".", "_").lower()

    chart_cids = [(ticker, _cid(ticker)) for ticker, _ in chart_paths]

    # Build message
    msg            = MIMEMultipart("related")
    msg["Subject"] = subject
    msg["From"]    = from_addr
    msg["To"]      = to_addr

    html_body = build_html(results_df, run_date, chart_cids)
    msg.attach(MIMEText(html_body, "html", "utf-8"))

    # Attach chart images inline
    for (ticker, filepath), (_, cid) in zip(chart_paths, chart_cids):
        try:
            with open(filepath, "rb") as fh:
                img = MIMEImage(fh.read(), _subtype="png")
            img.add_header("Content-ID", f"<{cid}>")
            img.add_header(
                "Content-Disposition", "inline",
                filename=os.path.basename(filepath),
            )
            msg.attach(img)
        except FileNotFoundError:
            print(f"  Warning: chart file not found — {filepath}")

    # Send
    print(f"Connecting to {smtp_host}:{smtp_port} …")
    with smtplib.SMTP(smtp_host, smtp_port, timeout=30) as server:
        server.ehlo()
        server.starttls()
        server.login(from_addr, password)
        server.sendmail(from_addr, [to_addr], msg.as_string())

    print(f"✓ Email sent → {to_addr}")
