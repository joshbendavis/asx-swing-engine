# ASX Swing Engine

A systematic ASX swing trading engine with relative-strength entry filtering, multi-trigger confirmation, and regime-conditional position sizing. Validated over 8 years (2017–2026).

---

## Production Configuration — Variant F + Trail + Gap Risk (locked 2026-04-18)

| Parameter | Value |
|---|---|
| Universe | 105 ASX200 stocks |
| Entry filter | RS top 20% vs XJO (cross-sectional, 63-day) |
| Entry triggers | ≥2 of: RSI bounce, MACD bullish cross, volume breakout (≥1.5× avg) |
| ATR floor | ≥3.5% at signal date |
| Entry timing | Signal-day close (3:30pm proxy) |
| Stop loss | Entry − 2×ATR(14) |
| Exit method | **2×ATR trailing stop**, activates after +1R profit |
| Trail logic | Phase 1: hard 2×ATR stop until close ≥ entry + 1R · Phase 2: trail_high − 2×ATR (EWM ATR, causal) |
| Time stop | Exit on bar 21 if trail not triggered |
| Max positions | 6 |
| Heat budget | 9% total open risk |
| Unpaid heat cap | 3% max open risk in positions below +1R |
| Sector cap | Max 2 positions per GICS sector |
| Min risk floor | $75 per trade |
| Commission | IBKR $6/leg ($12 round-trip) |
| **Gap Risk Engine** | Step 3.5 in pipeline: scores overnight exposure, exits/reduces/tightens before next open |

### Variant F — Regime-Conditional Momentum Filter

The regime detector classifies each trading day into one of three F-regime states based on XJO vs its 200 EMA and ASX200 breadth (% of sample above their 50 EMA):

| F-Regime | Condition | Behaviour |
|---|---|---|
| **STRONG_BULL** | XJO > 200 EMA AND breadth > 55% | Full 1.0× risk, no momentum filter |
| **WEAK_BULL** | XJO > 200 EMA AND breadth 40–55% | Reduced 0.8× risk, no momentum filter |
| **CHOPPY_BEAR** | breadth < 40% OR XJO < 200 EMA | Hard filter: stock ROC(20) > 0 AND EMA-50 slope > 0 required; 1.0× risk if passed |

Over the 8-year validation period: STRONG_BULL = 54.6% of trading days, WEAK_BULL = 12.0%, CHOPPY_BEAR = 33.4%.

---

## 8-Year Backtest Results — Final Production Config (2017–2026)

**Setup:** $20,000 start, 1.5% risk/trade (regime-conditional), max 6 positions, IBKR $6/leg, 0.2% slippage.

Three progressive configurations validated in sequence:

| Metric | F-Fixed (2:1 target) | F-Trail (2×ATR trail) | **F-Trail + Gap Risk** ✅ |
|---|---|---|---|
| Period | Apr 2017 – Apr 2026 | Apr 2017 – Apr 2026 | Apr 2017 – Apr 2026 |
| Trades (records) | 98 | 101 | 155 (incl. 48 partial reduces) |
| Full exits | 98 | 101 | 107 |
| Win rate | 58.0% | 62.4% | **60.0%** |
| **Profit factor** | 1.84 | 1.97 | **1.67** |
| Avg R / record | 0.322 | 0.364 | **0.187** |
| Expectancy | $103/trade | $120/trade | **$41/record** |
| Gap stops | 17 | 24 | **14** (−42%) |
| Best trade R | — | 5.344R | **5.676R (LYC.AX)** |
| Total return | +47.5% | +57.5% | **+28.9%** |
| CAGR | +5.0% | +5.8% | **+3.2%** |
| **Max drawdown** | −9.0% | −9.0% | **−6.3%** |
| Calmar ratio | 0.56 | 0.65 | **0.51** |
| Sharpe ratio | 1.16 | 1.12 | **1.03** |
| Avg hold days | 14.8 | 15.5 | **8.7** |

> **Gap risk engine trade-off:** Reduces return in exchange for materially lower drawdown (−9.0% → −6.3%), 42% fewer gap stops, and a hard cap on unpaid overnight exposure. The engine fires only on genuinely stalled or deteriorating positions — the best trade (LYC +5.676R) survived intact.

### Year-by-Year — F-Trail + Gap Risk Engine (Production)

| Year | Return |
|---|---|
| 2018 | +$149 (+0.6%) |
| 2019 | +$1,144 (+10.5%) |
| 2020 | +$2,303 (+5.4%) |
| 2021 | +$750 (+2.6%) |
| 2022 | −$255 (−0.1%) |
| 2023 | +$125 (−1.1%) |
| 2024 | −$154 (−0.8%) |
| 2025 | +$2,439 (+10.0%) |
| 2026 | −$88 (−0.5%) |

---

## Why Regime-Conditional (Not Flat, Not Fully-Scaled)

Four approaches were compared across 8 years with identical realistic costs:

| Approach | Trades | Return | CAGR | Max DD | Sharpe | Calmar |
|---|---|---|---|---|---|---|
| **Variant F** (regime-conditional) | 101 | +57.5% | 5.8% | −9.0% | **1.12** | **0.65** |
| Control A (flat, no filter) | 113 | +39.8% | 4.3% | −12.3% | 0.77 | 0.35 |
| C (hard block: ROC20>0 & EMA50>0) | 68 | +16.6% | 1.9% | −9.0% | 0.71 | 0.22 |
| E (soft scaler: 1.0×/0.6× by quality) | 111 | +28.6% | 3.2% | −11.1% | 0.85 | 0.29 |

**The key insight:** The CHOPPY_BEAR momentum filter is correctly targeted. Over 8 years it blocked 30 of 79 raw CHOPPY_BEAR signals — specifically those where the stock's own ROC-20 or EMA-50 slope was already negative heading into the trade. These are the trades that most frequently gap through stops and generate time-stop exits. By filtering them (not blocking the whole regime), STRONG_BULL trades in trending markets fire normally — which is why 2021 is preserved.

---

## Architecture

```
run_daily.py           # Orchestrates daily pipeline (8 steps)
  regime_detector.py   # Step 0: XJO regime + Variant F f_regime → regime.json
  screener.py          # Step 1: Universe filter: price >$0.50, vol >200K, mcap >$100M, ATR ≥3.5%
  signals.py           # Step 2: RS + trigger scan → Variant F momentum filter (CHOPPY_BEAR)
  risk_engine.py       # Step 3: Position sizing: 1.5% × f_regime_size, sector cap, heat budget
  gap_risk_engine.py   # Step 3.5: Overnight gap-risk scoring → EXIT/REDUCE/TIGHTEN/HOLD
  ibkr_executor.py     # Step 6: Interactive Brokers order submission
  exit_logger.py       # Step 7: Tracks open positions, logs exits
  dashboard.py         # Streamlit monitoring UI
backtest_trail.py      # Trail stop backtest: F-Fixed vs F-Trail comparison
backtest_gap.py        # Gap risk backtest: F-Trail vs F-Trail + Gap Risk Engine
backtest_full.py       # Multi-scenario comparison engine (regime variants)
```

### Gap Risk Engine — Scoring Model

Each open position is scored nightly (end-of-day, before the next open):

| Factor | Score |
|---|---|
| Bars held ≥ 5 | +15 |
| Bars held ≥ 8 | +15 (stacks) |
| ATR% > 4% | +15 |
| ATR% > 6% | +15 (stacks) |
| Regime: CHOPPY/BEAR | +20 |
| ROC-20 < −2% | +15 |
| Position UNPAID (<1R) | +25 |
| Position PARTIALLY_PAID (1–2R) | +10 |
| Event risk proxy (vol >2.5× avg AND range >2×ATR) | +20 |

**Actions by classification × score** (grace period: first 5 bars exempt):

| Classification | Threshold | Action |
|---|---|---|
| UNPAID | score > 80 | EXIT at close |
| UNPAID | score ≥ 70 | REDUCE 50% (once per entry) |
| UNPAID | days ≥ 12 | DAY-12 forced exit (non-negotiable) |
| PARTIALLY_PAID | score > 60 | REDUCE 50% |
| PARTIALLY_PAID | score ≥ 40 | TIGHTEN trail to 1.8×ATR |
| PAID (≥2R) | score > 60 | TIGHTEN trail to 1.8×ATR |
| Portfolio | unpaid heat > 3% | Force-exit highest-scored unpaid position |

---

## Running the Pipeline

### Prerequisites

```bash
pip install -r requirements.txt
```

### Daily pipeline

```bash
python run_daily.py
```

Runs: regime detection → screener → signal scan (with Variant F filter) → risk sizing → IBKR submission → logs output to `logs/`.

### Dashboard

```bash
streamlit run dashboard.py
```

Opens at `http://localhost:8501`. Shows F-regime state, open positions, recent signals, last-run status.

### Backtest

```bash
# Trail stop vs fixed 2:1 target comparison (F-Fixed vs F-Trail)
python backtest_trail.py

# Gap risk engine impact (F-Trail vs F-Trail + Gap Risk Engine)
python backtest_gap.py

# Multi-scenario regime variant comparison
python backtest_full.py
```

Results saved to `results/backtest/`.

---

## Key Design Decisions

- **RS cross-sectional percentile** (not absolute threshold): adapts automatically to market conditions
- **2+ triggers always required**: reduces false entries — each trigger (RSI bounce, MACD cross, volume) independently confirms momentum
- **ATR ≥3.5% hard floor**: filters low-volatility setups that don't have enough range for a 2R trail
- **Variant F regime-conditional sizing**: STRONG_BULL (full size) / WEAK_BULL (0.8×) / CHOPPY_BEAR (momentum gate)
- **2×ATR trailing stop (activates at +1R)**: lets winners run past fixed 2:1 targets; Phase 1 uses hard stop until +1R, Phase 2 trails peak − 2×ATR using causal EWM ATR
- **Gap risk engine (Step 3.5)**: nightly scoring of overnight exposure — forces exits/reduces on stalled UNPAID positions before they gap through stops; cuts gap stops by 42%, reduces max drawdown from 9% to 6.3%
- **Unpaid heat cap (3%)**: portfolio-level cap on risk from positions below +1R — the highest-risk overnight exposure category
- **Sector cap (2/sector)**: prevents concentration in one sector rotational theme
- **Heat budget (9%)**: caps total open risk regardless of how many signals fire simultaneously
- **No leverage, no shorting**: long-only, 100% cash when no signals
