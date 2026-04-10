"""
risk_engine.py
--------------
Portfolio-level risk gate that sits between signal generation and IBKR execution.

Reads:
    results/signals_output.csv   — raw signals from signals.py
    results/regime.json          — market context (informational only, no sizing effect)
    logs/trades.csv              — open bracket orders for exposure/heat
    results/sector_cache.json    — persistent sector cache (auto-populated)

Writes:
    results/signals_final.csv    — gated signals for ibkr_executor
    results/sector_cache.json    — updated sector cache after any yfinance lookups

PRODUCTION CONFIGURATION (validated 8-year backtest, 2017-2026):
    Risk per trade : 1.5% of account — FLAT, no regime scaling
    Regime         : informational only — displayed on dashboard, never adjusts sizing

Five checks applied per signal (after portfolio-level gates pass):

  1. EXPOSURE GATE  — if >= MAX_POSITIONS already open, block all new orders.
  2. HEAT CHECK     — if open risk >= HEAT_LIMIT_PCT, scale or block new sizes.
  3. SCORE RANKING  — signals sorted by composite_score desc before slot trimming.
  4. MIN RISK FLOOR — skip any trade where risk < MIN_RISK_AUD ($75).
  5. SECTOR CAP     — skip if the same GICS sector already has >= SECTOR_CAP (2)
                      open positions.  Unknown sectors are not capped.

Output columns added to signals_final.csv:
    risk_multiplier   — always 1.0 (flat sizing); kept for ibkr_executor compatibility.
    adjusted_risk_aud — pre-computed dollar risk for the position.
    sector            — GICS sector string used for concentration check.

Usage:
    python risk_engine.py                  # run standalone
    python risk_engine.py --dry-run        # print decisions without writing CSV
    from risk_engine import run_risk_engine
    summary = run_risk_engine()
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Config  (mirror values from ibkr_executor.py)
# ---------------------------------------------------------------------------
RISK_PCT          = 0.015          # base risk per trade as fraction of account
MAX_POSITIONS     = 6              # maximum concurrent open positions
HEAT_LIMIT_PCT    = 9.0            # max portfolio heat % before scaling kicks in
MIN_SHARES        = 1              # minimum viable share parcel

ACCOUNT_BALANCE   = float(os.getenv("PAPER_BALANCE_OVERRIDE", "20000"))

MIN_RISK_AUD      = 75.0           # skip trades below this dollar risk
SECTOR_CAP        = 2              # max open positions per GICS sector

SIGNALS_CSV       = Path("results/signals_output.csv")
FINAL_CSV         = Path("results/signals_final.csv")
REGIME_JSON       = Path("results/regime.json")
TRADES_CSV        = Path("logs/trades.csv")
SECTOR_CACHE      = Path("results/sector_cache.json")

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
log = logging.getLogger("risk-engine")
if not log.handlers:
    _h = logging.StreamHandler(sys.stdout)
    _h.setFormatter(logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s", "%H:%M:%S"
    ))
    log.addHandler(_h)
    log.setLevel(logging.INFO)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_regime() -> dict:
    """Load regime.json; return BULL defaults if missing."""
    default = {"regime": "BULL", "position_size_multiplier": 1.0, "confidence": 100}
    if not REGIME_JSON.exists():
        log.warning("regime.json not found — using BULL defaults.")
        return default
    try:
        with open(REGIME_JSON) as fh:
            return json.load(fh)
    except Exception as exc:
        log.warning("Could not parse regime.json (%s) — using BULL defaults.", exc)
        return default


def _load_open_trades() -> pd.DataFrame:
    """
    Return the canonical open positions from logs/trades.csv.

    A row is considered open when:
      - tp_id > 0  (a valid bracket was submitted)
      - exit_timestamp is NaN / missing  (not yet closed)

    Deduplicates by order_ref, keeping the latest row.
    """
    if not TRADES_CSV.exists():
        return pd.DataFrame()

    df = pd.read_csv(TRADES_CSV, parse_dates=["timestamp"])

    # Add exit columns if absent (old schema)
    for col in ["exit_timestamp", "tp_id"]:
        if col not in df.columns:
            df[col] = np.nan

    df["tp_id"] = pd.to_numeric(df["tp_id"], errors="coerce").fillna(0)

    # Only bracket orders that haven't closed
    open_df = df[(df["tp_id"] > 0) & (df["exit_timestamp"].isna())].copy()
    open_df = open_df.sort_values("timestamp").drop_duplicates("order_ref", keep="last")
    return open_df.reset_index(drop=True)


def _write_final(df: pd.DataFrame, dry_run: bool) -> None:
    """Write (or log-only in dry-run) the final signals CSV."""
    FINAL_CSV.parent.mkdir(parents=True, exist_ok=True)
    if dry_run:
        log.info("[DRY-RUN] Would write %d signal(s) to %s", len(df), FINAL_CSV)
    else:
        df.to_csv(FINAL_CSV, index=True)
        log.info("signals_final.csv written -> %s  (%d signal(s))", FINAL_CSV, len(df))


def _write_empty(reason: str, dry_run: bool) -> None:
    """Write an empty signals_final.csv and log why."""
    log.warning("BLOCKED: %s — writing empty signals_final.csv.", reason)
    if not dry_run:
        FINAL_CSV.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame().to_csv(FINAL_CSV, index=False)


def _load_sectors(tickers: list[str]) -> dict[str, str]:
    """
    Return {ticker: sector_string} for the given list.

    Reads SECTOR_CACHE first; fetches missing tickers from yfinance
    (yf.Ticker().info['sector']) and persists results back to cache.
    Tickers whose sector cannot be determined are stored as 'Unknown'.
    """
    # Load existing cache
    cache: dict[str, str] = {}
    if SECTOR_CACHE.exists():
        try:
            with open(SECTOR_CACHE) as fh:
                cache = json.load(fh)
        except Exception as exc:
            log.warning("Could not read sector cache (%s) — starting fresh.", exc)

    missing = [t for t in tickers if t not in cache]
    if missing:
        log.info("Fetching sector for %d uncached ticker(s) via yfinance ...", len(missing))
        for ticker in missing:
            try:
                import yfinance as yf
                info   = yf.Ticker(ticker).info
                sector = info.get("sector") or "Unknown"
            except Exception:
                sector = "Unknown"
            cache[ticker] = sector
            log.info("  %-12s  sector=%s", ticker, sector)

        # Persist updated cache
        try:
            SECTOR_CACHE.parent.mkdir(parents=True, exist_ok=True)
            with open(SECTOR_CACHE, "w") as fh:
                json.dump(cache, fh, indent=2)
        except Exception as exc:
            log.warning("Could not save sector cache: %s", exc)

    return {t: cache.get(t, "Unknown") for t in tickers}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_risk_engine(dry_run: bool = False) -> dict:
    """
    Apply portfolio risk gates and write results/signals_final.csv.

    Returns a summary dict with all decisions made.
    """
    log.info("=" * 60)
    log.info("  Risk Engine — %s", datetime.now().strftime("%d %b %Y %H:%M"))
    log.info("=" * 60)

    summary = {
        "signals_in":        0,
        "signals_out":       0,
        "skipped_floor":     0,
        "skipped_sector":    0,
        "n_open":            0,
        "slots_available":   0,
        "current_heat_pct":  0.0,
        "heat_budget_aud":   0.0,
        "risk_multiplier":   1.0,
        "psm":               1.0,
        "regime":            "BULL",
        "blocked":           False,
        "block_reason":      None,
    }

    # ── 1. Load raw signals ───────────────────────────────────────────────────
    if not SIGNALS_CSV.exists():
        log.warning("signals_output.csv not found — nothing to process.")
        _write_empty("no signals file", dry_run)
        summary["blocked"] = True
        summary["block_reason"] = "no signals file"
        return summary

    signals = pd.read_csv(SIGNALS_CSV, index_col=0)   # rank is the index
    summary["signals_in"] = len(signals)
    log.info("Signals loaded: %d candidate(s)", len(signals))

    if signals.empty:
        log.info("No signals today — nothing to gate.")
        _write_empty("no signals", dry_run)
        return summary

    # ── 2. Load regime ────────────────────────────────────────────────────────
    regime_data = _load_regime()
    regime      = regime_data.get("regime", "BULL")
    confidence  = regime_data.get("confidence", 100)
    summary["regime"] = regime
    summary["psm"]    = 1.0   # flat sizing — regime is informational only
    log.info("Regime: %s  |  confidence=%s  |  sizing=FLAT 1.0x (regime informational only)",
             regime, confidence)

    # ── 3. Exposure gate ──────────────────────────────────────────────────────
    open_trades = _load_open_trades()
    n_open      = len(open_trades)
    summary["n_open"] = n_open
    log.info(
        "Open positions: %d / %d  (from %s)",
        n_open, MAX_POSITIONS, TRADES_CSV,
    )

    if n_open >= MAX_POSITIONS:
        _write_empty(
            f"portfolio full ({n_open}/{MAX_POSITIONS} positions open)", dry_run
        )
        summary["blocked"]      = True
        summary["block_reason"] = f"portfolio full ({n_open}/{MAX_POSITIONS})"
        return summary

    slots_available = MAX_POSITIONS - n_open
    summary["slots_available"] = slots_available
    log.info("Slots available: %d", slots_available)

    # ── 4. Portfolio heat check ───────────────────────────────────────────────
    base_risk_aud   = ACCOUNT_BALANCE * RISK_PCT   # $300 at default settings

    if not open_trades.empty and "risk_aud" in open_trades.columns:
        total_open_risk  = float(open_trades["risk_aud"].sum())
    else:
        total_open_risk  = 0.0

    current_heat_pct = total_open_risk / ACCOUNT_BALANCE * 100
    heat_remaining   = max(0.0, HEAT_LIMIT_PCT - current_heat_pct)
    heat_budget_aud  = ACCOUNT_BALANCE * heat_remaining / 100

    summary["current_heat_pct"] = round(current_heat_pct, 2)
    summary["heat_budget_aud"]  = round(heat_budget_aud,  2)

    log.info(
        "Portfolio heat: %.1f%% used / %.1f%% limit  "
        "(open risk $%.0f / account $%.0f)  |  budget remaining: $%.0f",
        current_heat_pct, HEAT_LIMIT_PCT,
        total_open_risk, ACCOUNT_BALANCE, heat_budget_aud,
    )

    if heat_budget_aud <= 0:
        _write_empty(
            f"heat limit reached ({current_heat_pct:.1f}% >= {HEAT_LIMIT_PCT}%)",
            dry_run,
        )
        summary["blocked"]      = True
        summary["block_reason"] = (
            f"heat limit reached ({current_heat_pct:.1f}% >= {HEAT_LIMIT_PCT}%)"
        )
        return summary

    # ── 5. Compute effective risk per trade ───────────────────────────────────
    # Flat 1.5% sizing — regime does NOT adjust position size.
    # Heat cap is the only runtime constraint (prevents exceeding 9% portfolio heat).
    n_new          = min(slots_available, len(signals))
    heat_risk_cap  = heat_budget_aud / n_new       # fair share of remaining budget
    final_risk_aud = min(base_risk_aud, heat_risk_cap)
    risk_multiplier = round(final_risk_aud / base_risk_aud, 4) if base_risk_aud > 0 else 0.0

    summary["risk_multiplier"] = risk_multiplier

    log.info(
        "Sizing: base=$%.0f  heat_cap=$%.0f  final=$%.0f  multiplier=%.4f",
        base_risk_aud, heat_risk_cap, final_risk_aud, risk_multiplier,
    )

    if final_risk_aud < 1.0:
        _write_empty(
            f"final risk ${final_risk_aud:.2f} too small to place any trade", dry_run
        )
        summary["blocked"]      = True
        summary["block_reason"] = "final risk < $1 after heat adjustment"
        return summary

    # ── 6. Sort by composite_score desc (best signals first) ─────────────────
    if "composite_score" in signals.columns:
        signals = signals.sort_values("composite_score", ascending=False)
    log.info("Signals ranked by composite_score (highest first).")

    # ── 7. Build sector counts from open positions ────────────────────────────
    open_tickers: list[str] = []
    if not open_trades.empty and "ticker" in open_trades.columns:
        open_tickers = open_trades["ticker"].tolist()

    # Gather all tickers we need sectors for (open + candidates)
    candidate_tickers = signals["ticker"].tolist() if "ticker" in signals.columns else []
    all_need_sectors  = list(set(open_tickers + candidate_tickers))
    sector_map        = _load_sectors(all_need_sectors)

    # Count open positions per sector (skip Unknown)
    sector_open_count: dict[str, int] = {}
    for t in open_tickers:
        sec = sector_map.get(t, "Unknown")
        if sec != "Unknown":
            sector_open_count[sec] = sector_open_count.get(sec, 0) + 1

    log.info("Open sector counts: %s", sector_open_count if sector_open_count else "none")

    # ── 8. Per-signal filtering: floor + sector cap + slot limit ──────────────
    approved: list[pd.Series] = []
    skipped_floor:  list[str] = []
    skipped_sector: list[str] = []
    skipped_slots:  list[str] = []
    sector_session: dict[str, int] = dict(sector_open_count)  # mutable copy

    for rank, row in signals.iterrows():
        if len(approved) >= slots_available:
            ticker = str(row.get("ticker", "???"))
            skipped_slots.append(ticker)
            continue

        ticker = str(row.get("ticker", "???"))

        # Min risk floor
        if final_risk_aud < MIN_RISK_AUD:
            log.info("  SKIP %-6s — risk $%.0f < floor $%.0f",
                     ticker.replace(".AX", ""), final_risk_aud, MIN_RISK_AUD)
            skipped_floor.append(ticker)
            continue

        # Sector cap
        sector = sector_map.get(ticker, "Unknown")
        if sector != "Unknown" and sector_session.get(sector, 0) >= SECTOR_CAP:
            log.info("  SKIP %-6s — sector '%s' already has %d/%d open",
                     ticker.replace(".AX", ""), sector,
                     sector_session[sector], SECTOR_CAP)
            skipped_sector.append(ticker)
            continue

        # Passed — approve
        row = row.copy()
        row["risk_multiplier"]   = risk_multiplier   # always 1.0 or heat-capped
        row["adjusted_risk_aud"] = round(final_risk_aud, 2)
        row["regime"]            = regime            # informational only
        row["sector"]            = sector
        approved.append(row)

        # Update sector count for this session
        if sector != "Unknown":
            sector_session[sector] = sector_session.get(sector, 0) + 1

    signals_out = pd.DataFrame(approved)
    summary["signals_out"] = len(signals_out)

    # ── 9. Log the decision for each approved signal ──────────────────────────
    log.info("--- Approved signals (%d) ---", len(signals_out))
    for _, row in signals_out.iterrows():
        ticker = str(row.get("ticker", "???"))
        entry  = float(row.get("entry", 0))
        sl     = float(row.get("stop_loss", 0))
        tgt    = float(row.get("target", 0))
        sector = str(row.get("sector", "Unknown"))
        rps    = entry - sl
        shares = int(final_risk_aud / rps) if rps > 0 else 0
        log.info(
            "  OK  %-6s  entry=%.3f  stop=%.3f  target=%.3f  "
            "risk=$%.0f  shares~%d  sector=%s",
            ticker.replace(".AX", ""),
            entry, sl, tgt,
            final_risk_aud, shares, sector,
        )

    if skipped_floor:
        log.info("  FLOOR skip (%d): %s", len(skipped_floor),
                 ", ".join(t.replace(".AX", "") for t in skipped_floor))
    if skipped_sector:
        log.info("  SECTOR skip (%d): %s", len(skipped_sector),
                 ", ".join(t.replace(".AX", "") for t in skipped_sector))
    if skipped_slots:
        log.info("  SLOTS skip (%d): %s", len(skipped_slots),
                 ", ".join(t.replace(".AX", "") for t in skipped_slots))

    summary["skipped_floor"]  = len(skipped_floor)
    summary["skipped_sector"] = len(skipped_sector)

    # ── 10. Write output ──────────────────────────────────────────────────────
    _write_final(signals_out, dry_run)

    log.info("=" * 60)
    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="ASX Swing Engine — portfolio risk gate")
    p.add_argument("--dry-run", action="store_true",
                   help="Print decisions without writing signals_final.csv")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    summary = run_risk_engine(dry_run=args.dry_run)
    print()
    import pprint
    pprint.pprint(summary)
