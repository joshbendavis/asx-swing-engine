"""
run_daily.py
------------
Daily orchestrator for the ASX Swing Trade Engine.

Pipeline:
    1. Run screener  → results/screener_output.csv
    2. Generate charts for top 10  → results/charts/YYYYMMDD/
    3. Send HTML email with results table + inline charts

Credentials are read from .env (copy .env.example → .env and fill in):
    EMAIL_FROM      your email address
    EMAIL_PASSWORD  app password (NOT your regular login password)
    EMAIL_TO        recipient (defaults to EMAIL_FROM)
    SMTP_HOST       default: smtp-mail.outlook.com
    SMTP_PORT       default: 587

Usage:
    python run_daily.py            # full run (screener + charts + email)
    python run_daily.py --no-email # screener + charts only
    python run_daily.py --no-charts --no-email  # screener only
"""

import argparse
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# ── logging ───────────────────────────────────────────────────────────────────
log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)
log_file = log_dir / f"run_{datetime.today().strftime('%Y%m%d_%H%M%S')}.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.FileHandler(log_file, encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger("asx-swing")


# ── imports (after logging is configured) ─────────────────────────────────────
from screener import run_screener
from utils.charts import generate_charts
from utils.emailer import send_email


# ── main ──────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="ASX Swing Engine — daily run")
    p.add_argument("--no-charts", action="store_true", help="Skip chart generation")
    p.add_argument("--no-email",  action="store_true", help="Skip sending email")
    p.add_argument("--top-n",     type=int, default=10, help="Charts for top N (default 10)")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    log.info("=" * 62)
    log.info("  ASX SWING ENGINE — daily run  %s", datetime.today().strftime("%d %b %Y %H:%M"))
    log.info("=" * 62)

    # ── 1. Screener ───────────────────────────────────────────────────────────
    log.info("Step 1/3 — Running screener …")
    try:
        results = run_screener()
    except Exception as exc:
        log.error("Screener raised an exception: %s", exc, exc_info=True)
        sys.exit(1)

    if results.empty:
        log.warning("Screener returned no results. Nothing to report.")
        sys.exit(0)

    log.info("Screener complete: %d stocks passed all filters.", len(results))

    # Always save fresh CSV
    csv_path = Path("results") / "screener_output.csv"
    results.to_csv(csv_path)
    log.info("Results saved → %s", csv_path)

    # Print top 10 to log
    log.info("\n%s", results.head(10)[
        ["ticker", "last_price", "composite_score", "rs_vs_xjo",
         "momentum_20d", "atr_pct", "rsi_14"]
    ].to_string())

    # ── 2. Charts ─────────────────────────────────────────────────────────────
    chart_paths: list[tuple[str, str]] = []

    if not args.no_charts:
        log.info("Step 2/3 — Generating charts for top %d …", args.top_n)
        try:
            chart_paths = generate_charts(results, top_n=args.top_n)
            log.info("Generated %d chart(s).", len(chart_paths))
        except Exception as exc:
            log.error("Chart generation failed: %s", exc, exc_info=True)
            # Non-fatal — continue to email with no charts
    else:
        log.info("Step 2/3 — Charts skipped (--no-charts).")

    # ── 3. Email ──────────────────────────────────────────────────────────────
    if not args.no_email:
        log.info("Step 3/3 — Sending email …")

        email_from = os.getenv("EMAIL_FROM", "").strip()
        email_pass = os.getenv("EMAIL_PASSWORD", "").strip()
        email_to   = os.getenv("EMAIL_TO",   email_from).strip()
        smtp_host  = os.getenv("SMTP_HOST",  "smtp-mail.outlook.com").strip()
        smtp_port  = int(os.getenv("SMTP_PORT", "587"))

        if not email_from or not email_pass:
            log.warning(
                "EMAIL_FROM or EMAIL_PASSWORD not set in .env — skipping email.\n"
                "  Copy .env.example → .env and add your credentials."
            )
        else:
            try:
                send_email(
                    results_df  = results,
                    chart_paths = chart_paths,
                    smtp_host   = smtp_host,
                    smtp_port   = smtp_port,
                    from_addr   = email_from,
                    password    = email_pass,
                    to_addr     = email_to,
                )
                log.info("Email sent successfully → %s", email_to)
            except Exception as exc:
                log.error("Email failed: %s", exc, exc_info=True)
    else:
        log.info("Step 3/3 — Email skipped (--no-email).")

    log.info("=" * 62)
    log.info("  Daily run complete.  Log → %s", log_file)
    log.info("=" * 62)


if __name__ == "__main__":
    main()
