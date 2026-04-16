"""
run_daily.py
------------
Daily orchestrator for the ASX Swing Trade Engine.

Pipeline:
    0. Regime detector -> results/regime.json
    1. Screener        -> results/screener_output.csv
    2. Signals         -> results/signals_output.csv
    3. Risk engine     -> results/signals_final.csv
    4. Charts          -> results/charts/YYYYMMDD/
    5. Email
    6. IBKR execution  <- reads signals_final.csv
    7. Exit logger     (background process)

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
from regime_detector import run_regime_detector
from screener import run_screener
from signals import run_signals
from risk_engine import run_risk_engine
from ibkr_executor import run_executor as ibkr_run
from utils.charts import generate_charts
from utils.emailer import send_email
from utils.telegram_bot import send_daily_summary


# ── main ──────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="ASX Swing Engine - daily run")
    p.add_argument("--no-charts",        action="store_true", help="Skip chart generation")
    p.add_argument("--no-email",         action="store_true", help="Skip sending email")
    p.add_argument("--no-signals",       action="store_true", help="Skip signal scanner")
    p.add_argument("--no-risk-engine",   action="store_true", help="Skip risk engine (use signals_output.csv directly)")
    p.add_argument("--no-ibkr",          action="store_true", help="Skip IBKR order submission")
    p.add_argument("--ibkr-dry-run",     action="store_true", help="IBKR dry-run: print orders, do not submit")
    p.add_argument("--no-exit-logger",   action="store_true", help="Skip launching exit_logger in background")
    p.add_argument("--top-n",            type=int, default=10, help="Charts for top N (default 10)")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    log.info("=" * 62)
    log.info("  ASX SWING ENGINE - daily run  %s", datetime.today().strftime("%d %b %Y %H:%M"))
    log.info("=" * 62)

    # Context dict accumulated through the pipeline for Telegram summary at the end
    _tg: dict = {
        "run_date": datetime.today().strftime("%Y-%m-%d"),
        "run_time": datetime.today().strftime("%H:%M"),
        "steps":    {k: None for k in ["regime","screener","signals","risk","ibkr","exit_logger"]},
        "screener_n": None,
        "signals_n":  None,
        "orders_n":   None,
        "regime":     None,
        "f_regime":   None,
        "confidence": None,
        "psm":        None,
    }

    # ── 0. Regime detector ────────────────────────────────────────────────────
    log.info("Step 0/7 - Detecting market regime ...")
    try:
        regime_result = run_regime_detector()
        regime        = regime_result["regime"]
        confidence    = regime_result["confidence"]
        psm           = regime_result["position_size_multiplier"]
        log.info(
            "Regime: %s  |  confidence=%d%%  |  position_size_multiplier=%.2f",
            regime, confidence, psm,
        )
        _tg["regime"]     = regime
        _tg["f_regime"]   = regime_result.get("f_regime", regime)
        _tg["confidence"] = confidence
        _tg["psm"]        = psm
        _tg["steps"]["regime"] = "ok"
        if regime == "BEAR":
            log.warning("BEAR regime — screener and signals will run but no orders will be placed.")
        elif regime == "HIGH_VOL":
            log.warning("HIGH_VOL regime — position sizes will be reduced to %.0f%%.", psm * 100)
        elif regime == "CHOPPY":
            log.warning("CHOPPY regime — signals require all 3 triggers.")
    except Exception as exc:
        log.error("Regime detector failed: %s — defaulting to BULL (no restrictions).", exc)
        regime = "BULL"
        _tg["steps"]["regime"] = "error"
        # Non-fatal: pipeline continues. regime.json may be stale or absent;
        # signals.py will fall back to BULL defaults automatically.

    # ── 1. Screener ───────────────────────────────────────────────────────────
    log.info("Step 1/7 - Running screener ...")
    try:
        results = run_screener()
    except Exception as exc:
        log.error("Screener raised an exception: %s", exc, exc_info=True)
        sys.exit(1)

    if results.empty:
        log.warning("Screener returned no results. Nothing to report.")
        sys.exit(0)

    log.info("Screener complete: %d stocks passed all filters.", len(results))
    _tg["screener_n"] = len(results)
    _tg["steps"]["screener"] = "ok"

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
        log.info("Step 2/7 - Running signal scanner ...")
        try:
            signals = run_signals()
            if signals.empty:
                log.info("No entry signals today.")
                _tg["signals_n"] = 0
            else:
                log.info("%d signal(s) found.", len(signals))
                _tg["signals_n"] = len(signals)
                sig_path = Path("results") / "signals_output.csv"
                signals.to_csv(sig_path)
                log.info("Signals saved -> %s", sig_path)
                log.info("\n%s", signals.head(10).to_string())
            _tg["steps"]["signals"] = "ok"
        except Exception as exc:
            log.error("Signal scanner failed: %s", exc, exc_info=True)
            _tg["steps"]["signals"] = "error"
            # Non-fatal - continue to charts and email
    else:
        log.info("Step 2/7 - Signals skipped (--no-signals).")

    # ── 3. Risk engine ────────────────────────────────────────────────────────
    if args.no_risk_engine:
        log.info("Step 3/7 - Risk engine skipped (--no-risk-engine).")
        log.info("          ibkr_executor will use signals_output.csv directly.")
    else:
        log.info("Step 3/7 - Running risk engine ...")
        try:
            risk_summary = run_risk_engine()
            if risk_summary["blocked"]:
                log.warning(
                    "Risk engine BLOCKED all orders: %s", risk_summary["block_reason"]
                )
                _tg["steps"]["risk"] = "warn"
            else:
                log.info(
                    "Risk engine approved: %d/%d signal(s)  |  "
                    "heat=%.1f%%  |  risk_multiplier=%.2f  |  slots=%d",
                    risk_summary["signals_out"],
                    risk_summary["signals_in"],
                    risk_summary["current_heat_pct"],
                    risk_summary["risk_multiplier"],
                    risk_summary["slots_available"],
                )
                _tg["steps"]["risk"] = "ok"
        except Exception as exc:
            log.error("Risk engine failed: %s", exc, exc_info=True)
            _tg["steps"]["risk"] = "error"
            # Non-fatal — IBKR step will still run with whatever signals_final.csv contains

    # ── 4. Charts ─────────────────────────────────────────────────────────────
    chart_paths: list[tuple[str, str]] = []

    if not args.no_charts:
        log.info("Step 4/7 - Generating charts for top %d ...", args.top_n)
        try:
            chart_paths = generate_charts(results, top_n=args.top_n)
            log.info("Generated %d chart(s).", len(chart_paths))
        except Exception as exc:
            log.error("Chart generation failed: %s", exc, exc_info=True)
            # Non-fatal - continue to email with no charts
    else:
        log.info("Step 4/7 - Charts skipped (--no-charts).")

    # ── 4. Email ──────────────────────────────────────────────────────────────
    if not args.no_email:
        log.info("Step 5/7 - Sending email ...")

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
        log.info("Step 5/7 - Email skipped (--no-email).")

    # ── 5. IBKR order submission ───────────────────────────────────────────────
    if args.no_ibkr:
        log.info("Step 6/7 - IBKR skipped (--no-ibkr).")
    else:
        # Prefer risk-engine output; fall back to raw signals if risk engine was skipped
        sig_path = Path("results") / "signals_final.csv"
        if not sig_path.exists():
            sig_path = Path("results") / "signals_output.csv"
            log.warning(
                "Step 6/7 - signals_final.csv not found, falling back to signals_output.csv."
            )
        if not sig_path.exists():
            log.warning("Step 6/7 - no signals file found, skipping IBKR.")
        else:
            dry = args.ibkr_dry_run
            mode_label = "DRY-RUN" if dry else "LIVE SUBMIT"
            log.info("Step 6/7 - IBKR order submission [%s] ...", mode_label)
            try:
                n = ibkr_run(sig_path, dry_run=dry)
                log.info("IBKR: %d bracket order(s) submitted.", n)
                _tg["orders_n"]       = n
                _tg["steps"]["ibkr"] = "ok"
            except Exception as exc:
                log.error("IBKR executor failed: %s", exc, exc_info=True)
                _tg["steps"]["ibkr"] = "error"
                # Non-fatal - pipeline still completes

    # ── 6. Exit logger (background) ───────────────────────────────────────────
    if args.no_exit_logger:
        log.info("Step 7/7 - Exit logger skipped (--no-exit-logger).")
    else:
        log.info("Step 7/7 - Starting exit logger in background ...")
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
                _tg["steps"]["exit_logger"] = "ok"

        except Exception as exc:
            log.error("Failed to launch exit logger: %s", exc, exc_info=True)
            _tg["steps"]["exit_logger"] = "error"
            # Non-fatal — pipeline still completes

    # ── 8. Telegram summary ───────────────────────────────────────────────────
    log.info("Step 8/8 - Sending Telegram daily summary ...")
    try:
        send_daily_summary(_tg)
    except Exception as exc:
        log.warning("Telegram summary failed (non-fatal): %s", exc)

    log.info("=" * 62)
    log.info("  Daily run complete.  Log -> %s", log_file)
    log.info("=" * 62)


if __name__ == "__main__":
    main()
