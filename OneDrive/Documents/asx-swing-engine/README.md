# ASX Swing Engine

A systematic ASX swing trading engine with relative-strength entry filtering, multi-trigger confirmation, and regime-conditional position sizing. Validated over 8 years (2017–2026).

---

## Production Configuration — Variant F (locked 2026-04-14)

| Parameter | Value |
|---|---|
| Universe | 105 ASX200 stocks |
| Entry filter | RS top 20% vs XJO (cross-sectional, 63-day) |
| Entry triggers | ≥2 of: RSI bounce, MACD bullish cross, volume breakout (≥1.5× avg) |
| ATR floor | ≥3.5% at signal date |
| Entry timing | Signal-day close (3:30pm proxy) |
| Stop loss | Entry − 2×ATR(14) |
| Target | 2:1 R (entry + 4×ATR) |
| Breakeven move | Trail stop to entry at +1R |
| Time stop | Exit on bar 21 if target/stop not hit |
| Max positions | 6 |
| Heat budget | 9% total open risk |
| Sector cap | Max 2 positions per GICS sector |
| Min risk floor | $75 per trade |
| Commission | IBKR $6/leg ($12 round-trip) |

### Variant F — Regime-Conditional Momentum Filter

The regime detector classifies each trading day into one of three F-regime states based on XJO vs its 200 EMA and ASX200 breadth (% of sample above their 50 EMA):

| F-Regime | Condition | Behaviour |
|---|---|---|
| **STRONG_BULL** | XJO > 200 EMA AND breadth > 55% | Full 1.0× risk, no momentum filter |
| **WEAK_BULL** | XJO > 200 EMA AND breadth 40–55% | Reduced 0.8× risk, no momentum filter |
| **CHOPPY_BEAR** | breadth < 40% OR XJO < 200 EMA | Hard filter: stock ROC(20) > 0 AND EMA-50 slope > 0 required; 1.0× risk if passed |

Over the 8-year validation period: STRONG_BULL = 54.6% of trading days, WEAK_BULL = 12.0%, CHOPPY_BEAR = 33.4%.

---

## 8-Year Backtest Results — Variant F (2017–2026)

**Setup:** $20,000 start, 1.5% risk/trade (regime-conditional), max 6 positions, IBKR $6/leg, 0.2% slippage, gap risk model.

| Metric | Variant F | Control A (no filter) |
|---|---|---|
| Period | Apr 2017 – Apr 2026 | Apr 2017 – Apr 2026 |
| Trades | 102 | 113 |
| Win rate | **15.7%** | 13.3% |
| Targets hit | **16** | 15 |
| Stops hit | 36 | 45 |
| **Gap stops** | **17** | 20 |
| Time stops | 50 | 53 |
| Profit factor | **0.91** | 0.67 |
| Avg R / trade | **0.293** | 0.212 |
| Expectancy | **$89 / trade** | $63 / trade |
| Total P&L | **$9,079** | $8,659 |
| Final account | **$28,467** | $27,951 |
| Total return | **+42.3%** | +39.8% |
| CAGR | **+4.5%** | +4.3% |
| Max drawdown | **−$1,863 (−9.0%)** | −$2,544 (−12.3%) |
| Calmar ratio | **0.50** | 0.35 |
| Max consec. losses | **4** | 5 |
| **Sharpe ratio** | **1.06** | 0.77 |
| Avg hold days | 14.8 | 14.4 |

### Year-by-Year

| Year | Variant F | Control A |
|---|---|---|
| 2018 | −$417 (−2.3%) | −$490 (−2.8%) |
| 2019 | +$693 (+3.3%) | +$719 (+3.5%) |
| 2020 | +$1,803 (+8.5%) | +$1,124 (+5.0%) |
| **2021** | **+$3,324 (+14.6%)** | +$3,529 (+16.0%) |
| 2022 | **+$527 (+2.8%)** | −$269 (−0.6%) |
| 2023 | **−$366 (−1.8%)** | −$879 (−4.0%) |
| 2024 | −$3 (−1.0%) | +$52 (−0.8%) |
| 2025 | +$2,323 (+9.0%) | +$2,441 (+10.2%) |
| 2026 | +$1,195 (+4.2%) | +$2,433 (+9.5%) |

Variant F retains 94% of 2021's bull-run profit while halving drawdown in 2022/2023.

---

## Why Regime-Conditional (Not Flat, Not Fully-Scaled)

Four approaches were compared across 8 years with identical realistic costs:

| Approach | Trades | Return | CAGR | Max DD | Sharpe | Calmar |
|---|---|---|---|---|---|---|
| **Variant F** (regime-conditional) | 102 | +42.3% | 4.5% | −9.0% | **1.06** | **0.50** |
| Control A (flat, no filter) | 113 | +39.8% | 4.3% | −12.3% | 0.77 | 0.35 |
| C (hard block: ROC20>0 & EMA50>0) | 68 | +16.6% | 1.9% | −9.0% | 0.71 | 0.22 |
| E (soft scaler: 1.0×/0.6× by quality) | 111 | +28.6% | 3.2% | −11.1% | 0.85 | 0.29 |

**The key insight:** The CHOPPY_BEAR momentum filter is correctly targeted. Over 8 years it blocked 30 of 79 raw CHOPPY_BEAR signals — specifically those where the stock's own ROC-20 or EMA-50 slope was already negative heading into the trade. These are the trades that most frequently gap through stops and generate time-stop exits. By filtering them (not blocking the whole regime), STRONG_BULL trades in trending markets fire normally — which is why 2021 is preserved.

---

## Architecture

```
run_daily.py          # Orchestrates daily pipeline
  regime_detector.py  # Market context: XJO regime + Variant F f_regime → regime.json
  screener.py         # Universe filter: price >$0.50, vol >200K, mcap >$100M, ATR ≥3.5%
  signals.py          # RS + trigger scan → Variant F momentum filter (CHOPPY_BEAR)
  risk_engine.py      # Position sizing: 1.5% × f_regime_size, sector cap, heat budget
  exit_logger.py      # Tracks open positions, logs exits
  ibkr_executor.py    # Interactive Brokers order submission
  dashboard.py        # Streamlit monitoring UI
backtest.py           # Original single-scenario backtest
backtest_full.py      # 8-year validation backtest (multi-scenario comparison engine)
```

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
# Full 8-year validation (takes ~5 min for data download)
python backtest_full.py

# Original quick backtest
python backtest.py
```

Results saved to `results/backtest/`.

---

## Key Design Decisions

- **RS cross-sectional percentile** (not absolute threshold): adapts automatically to market conditions
- **2+ triggers always required**: reduces false entries — each trigger (RSI bounce, MACD cross, volume) independently confirms momentum
- **ATR ≥3.5% hard floor**: filters low-volatility setups that don't have enough range for a clean 2R move
- **Variant F regime-conditional sizing**: STRONG_BULL (full size) / WEAK_BULL (0.8×) / CHOPPY_BEAR (momentum gate)
- **ATR-based stops**: volatility-normalised, consistent 2R exit structure
- **Sector cap (2/sector)**: prevents concentration in one sector rotational theme
- **Heat budget (9%)**: caps total open risk regardless of how many signals fire simultaneously
- **No leverage, no shorting**: long-only, 100% cash when no signals
