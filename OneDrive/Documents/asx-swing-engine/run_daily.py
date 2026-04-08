"""
run_daily.py
------------
Daily orchestrator for the ASX Swing Trade Engine.

Pipeline:
    1. Run screener   -> results/screener_output.csv
    2. Run signals    -> results/signals_output.csv
    3. Generate charts for top 10 -> results/charts/YYYYMMDD/
    4. Send HTML email with results table + inline charts
    5. Submit bracket orders to IBKR TWS via ibkr_executor.py

Credentials are read from .env (copy .env.example -> .env and fill in):
    EMAIL_FROM          your email address
    EMAIL_PASSWORD      app password (NOT your regular login password)
    EMAIL_TO            recipient (defaults to EMAIL_FROM)
    SMTP_HOST           default: smtp-mail.outlook.com
    SMTP_PORT           default: 587
    IBKR_ACCOUNT        TWS account ID (default: DUK913437)
    IBKR_LIVE_CONFIRMED set to "yes" only when flipping ibkr_executor.py to live

Usage:
    python run_daily.py                          # full pipeline
    python run_daily.py --no-email               # skip email
    python run_daily.py --no-ibkr                # skip order submission
    python run_daily.py --ibkr-dry-run           # print IBKR orders without submitting
    python run_daily.py --no-charts --no-email   # screener + signals only
"""

import argparse
import logging
import os
import subprocess
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
from signals import run_signals
from ibkr_executor import run_executor as ibkr_run
from utils.charts import generate_charts
from utils.emailer import send_email


# ── main ──────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="ASX Swing Engine - daily run")
    p.add_argument("--no-charts",       action="store_true", help="Skip chart generation")
    p.add_argument("--no-email",        action="store_true", help="Skip sending email")
    p.add_argument("--no-signals",      action="store_true", help="Skip signal scanner")
    p.add_argument("--no-ibkr",         action="store_true", help="Skip IBKR order submission")
    p.add_argument("--ibkr-dry-run",    action="store_true", help="IBKR dry-run: print orders, do not submit")
    p.add_argument("--no-exit-logger",  action="store_true", help="Skip launching exit_logger in background")
    p.add_argument("--top-n",           type=int, default=10, help="Charts for top N (default 10)")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    log.info("=" * 62)
    log.info("  ASX SWING ENGINE - daily run  %s", datetime.today().strftime("%d %b %Y %H:%M"))
    log.info("=" * 62)

    # ── 1. Screener ───────────────────────────────────────────────────────────
    log.info("Step 1/5 - Running screener ...")
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
    log.info("Results saved -> %s", csv_path)

    # Print top 10 to log
    log.info("\n%s", results.head(10)[
        ["ticker", "last_price", "composite_score", "rs_vs_xjo",
         "momentum_20d", "atr_pct", "rsi_14"]
    ].to_string())

    # ── 2. Signals ────────────────────────────────────────────────────────────
    if not args.no_signals:
        log.info("Step 2/5 - Running signal scanner ...")
        try:
            signals = run_signals()
            if signals.empty:
                log.info("No entry signals today.")
            else:
                log.info("%d signal(s) found.", len(signals))
                sig_path = Path("results") / "signals_output.csv"
                signals.to_csv(sig_path)
                log.info("Signals saved -> %s", sig_path)
                log.info("\n%s", signals.head(10).to_string())
        except Exception as exc:
            log.error("Signal scanner failed: %s", exc, exc_info=True)
            # Non-fatal - continue to charts and email
    else:
        log.info("Step 2/5 - Signals skipped (--no-signals).")

    # ── 3. Charts ─────────────────────────────────────────────────────────────
    chart_paths: list[tuple[str, str]] = []

    if not args.no_charts:
        log.info("Step 3/5 - Generating charts for top %d ...", args.top_n)
        try:
            chart_paths = generate_charts(results, top_n=args.top_n)
            log.info("Generated %d chart(s).", len(chart_paths))
        except Exception as exc:
            log.error("Chart generation failed: %s", exc, exc_info=True)
            # Non-fatal - continue to email with no charts
    else:
        log.info("Step 3/5 - Charts skipped (--no-charts).")

    # ── 4. Email ──────────────────────────────────────────────────────────────
    if not args.no_email:
        log.info("Step 4/5 - Sending email ...")

        email_from = os.getenv("EMAIL_FROM", "").strip()
        email_pass = os.getenv("EMAIL_PASSWORD", "").strip()
        email_to   = os.getenv("EMAIL_TO",   email_from).strip()
        smtp_host  = os.getenv("SMTP_HOST",  "smtp-mail.outlook.com").strip()
        smtp_port  = int(os.getenv("SMTP_PORT", "587"))

        if not email_from or not email_pass:
            log.warning(
                "EMAIL_FROM or EMAIL_PASSWORD not set in .env - skipping email.\n"
                "  Copy .env.example -> .env and add your credentials."
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
                log.info("Email sent successfully -> %s", email_to)
            except Exception as exc:
                log.error("Email failed: %s", exc, exc_info=True)
    else:
        log.info("Step 4/5 - Email skipped (--no-email).")

    # ── 5. IBKR order submission ───────────────────────────────────────────────
    if args.no_ibkr:
        log.info("Step 5/5 - IBKR skipped (--no-ibkr).")
    else:
        sig_path = Path("results") / "signals_output.csv"
        if not sig_path.exists():
            log.warning("Step 5/5 - signals_output.csv not found, skipping IBKR.")
        else:
            dry = args.ibkr_dry_run
            mode_label = "DRY-RUN" if dry else "LIVE SUBMIT"
            log.info("Step 5/5 - IBKR order submission [%s] ...", mode_label)
            try:
                n = ibkr_run(sig_path, dry_run=dry)
                log.info("IBKR: %d bracket order(s) submitted.", n)
            except Exception as exc:
                log.error("IBKR executor failed: %s", exc, exc_info=True)
                # Non-fatal - pipeline still completes

    # ── 6. Exit logger (background) ───────────────────────────────────────────
    if args.no_exit_logger:
        log.info("Step 6/6 - Exit logger skipped (--no-exit-logger).")
    else:
        log.info("Step 6/6 - Starting exit logger in background ...")
        try:
            pid_file = Path("logs/exit_logger.pid")

            # Check if already running via PID file
            already_up = False
            if pid_file.exists():
                try:
                    pid = int(pid_file.read_text().strip())
                    if sys.platform == "win32":
                        import ctypes
                        handle = ctypes.windll.kernel32.OpenProcess(0x0400, False, pid)
                        if handle:
                            ctypes.windll.kernel32.CloseHandle(handle)
                            already_up = True
                    else:
                        os.kill(pid, 0)
                        already_up = True
                except (ValueError, OSError, ProcessLookupError):
                    pass

            if already_up:
                log.info("Exit logger already running (PID %s) - skipping launch.", pid)
            else:
                el_log = open(
                    log_dir / f"exit_logger_{datetime.today().strftime('%Y%m%d_%H%M%S')}.log",
                    "a", encoding="utf-8",
                )
                kwargs = dict(
                    args=[sys.executable, "exit_logger.py"],
                    stdout=el_log,
                    stderr=subprocess.STDOUT,
                    cwd=Path(__file__).parent,
                )
                if sys.platform == "win32":
                    kwargs["creationflags"] = (
                        subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP
                    )
                else:
                    kwargs["start_new_session"] = True

                proc = subprocess.Popen(**kwargs)
                log.info("Exit logger launched  PID=%d  log -> %s", proc.pid, el_log.name)

        except Exception as exc:
            log.error("Failed to launch exit logger: %s", exc, exc_info=True)
            # Non-fatal — pipeline still completes

    log.info("=" * 62)
    log.info("  Daily run complete.  Log -> %s", log_file)
    log.info("=" * 62)


if __name__ == "__main__":
    main()
