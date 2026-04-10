# ASX Swing Engine

A systematic ASX swing trading engine with relative-strength entry filtering, multi-trigger confirmation, and risk-managed position sizing. Validated over 8 years (2017–2026).

---

## Production Configuration (Locked 2026-04-10)

| Parameter | Value |
|---|---|
| Universe | 105 ASX200 stocks |
| Entry filter | RS top 20% vs XJO (cross-sectional, 63-day) |
| Entry triggers | ≥2 of: RSI bounce, MACD bullish cross, volume breakout (≥1.5× avg) |
| Risk per trade | 1.5% of account (flat — no regime adjustment) |
| Stop loss | Entry − 2×ATR(14) |
| Target | 2:1 R (entry + 4×ATR) |
| Breakeven move | Trail stop to entry at +1R |
| Time stop | Exit at open of bar 21 if target/stop not hit |
| Max positions | 6 |
| Heat budget | 9% total open risk |
| Sector cap | Max 2 positions per GICS sector |
| Min risk floor | $75 per trade |
| Commission | $10 per side |
| Regime detector | Runs daily — market context display only, does NOT affect sizing |

---

## 8-Year Backtest Results (2017–2026)

**Setup:** $20,000 start, 1.5% risk/trade, max 6 positions, 105-ticker ASX200 universe.

| Metric | Result |
|---|---|
| Period | Apr 2017 – Apr 2026 (9 years) |
| Trades | 166 |
| Win rate | 18.7% |
| Targets hit | 31 |
| Stops hit | 70 |
| Time stops | 65 |
| Profit factor | 0.92 |
| Avg R / trade | 0.238 |
| Expectancy | $74 / trade |
| Total P&L | $14,394 |
| Final account | $34,394 |
| Total return | +72.0% |
| CAGR | +7.0% |
| Max drawdown | −$5,551 (−18.7%) |
| Calmar ratio | 0.38 |
| Sharpe ratio | 0.77 |
| Max consec. losses | 5 |
| Avg hold days | 13.6 |

### Year-by-Year

| Year | P&L | Return |
|---|---|---|
| 2018 | +$69 | +2.4% |
| 2019 | +$1,819 | +6.9% |
| 2020 | +$3,036 | +13.9% |
| 2021 | +$1,642 | +9.2% |
| 2022 | +$1,449 | +4.8% |
| 2023 | −$2,783 | −10.7% |
| 2024 | +$351 | +0.4% |
| 2025 | +$5,871 | +22.3% |
| 2026 | +$2,939 | +10.1% |

---

## Why Flat Sizing (Not Regime-Adjusted)

The definitive 8-year test compared identical 166 trades at full size vs regime-scaled size (BULL=1.0×, WEAK\_BULL=0.75×, CHOPPY=0.6×, BEAR=0.5×, HIGH\_VOL=0.25×):

| | Full size | Regime scaled |
|---|---|---|
| Total return | +72.0% | +37.0% |
| CAGR | +7.0% | +4.0% |
| Max drawdown | −18.7% | −16.4% |
| Sharpe | 0.77 | 0.77 |
| Calmar | 0.38 | 0.24 |

Regime scaling gave up 48% of absolute return to save 2.3% of max drawdown — a 5.7:1 sacrifice ratio with zero improvement in Sharpe. **The RS top-20% filter already selects stocks demonstrating relative strength against the index; the XJO regime is the wrong proxy for individual stock quality.**

The regime detector still runs daily and is displayed on the dashboard as market context — it does not affect position sizing.

---

## Architecture

```
run_daily.py          # Orchestrates daily pipeline
  screener.py         # Universe filter: price >$0.50, vol >200K, mcap >$100M, above 200 EMA
  signals.py          # RS + trigger scan → signals list
  risk_engine.py      # Position sizing: 1.5% risk, sector cap, heat budget, $75 floor
  regime_detector.py  # Market context classifier (display only)
  exit_logger.py      # Tracks open positions, logs exits
  ibkr_executor.py    # Interactive Brokers order submission
  dashboard.py        # Streamlit monitoring UI
backtest.py           # Original single-scenario backtest
backtest_full.py      # 8-year validation backtest (dual-scenario comparison)
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

Runs screener → signal scan → risk sizing → regime detection → logs output to `logs/`.

### Dashboard

```bash
streamlit run dashboard.py
```

Opens at `http://localhost:8501`. Shows regime context, open positions, recent signals.

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
- **2+ triggers required**: reduces false entries — each trigger (RSI bounce, MACD cross, volume) independently confirms momentum
- **ATR-based stops**: volatility-normalised, consistent 2R exit structure
- **Sector cap (2/sector)**: prevents concentration in one sector rotational theme
- **Heat budget (9%)**: caps total open risk regardless of how many signals fire simultaneously
- **No leverage, no shorting**: long-only, 100% cash when no signals
