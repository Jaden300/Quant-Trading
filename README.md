# Trading Strategies

A Python backtesting and parameter-optimization framework for 5 trading strategies, each with a matching TradingView Pine Script. Run the optimizer to find the best parameters, use the daily scanner to find live signals, and visualize any backtest with a single command.

---

## Strategies

| Strategy | Type | Logic | Win Rate | Avg Hold |
|---|---|---|---|---|
| **Elektro BB** | Mean reversion | Enter at extreme oversold (BB + RSI), exit when overbought | ~80–100% | Months–years |
| **DCA Long** | Always-in | Base order + up to 5 safety orders averaging down, exit at TP | ~100% | Weeks–months |
| **Bollinger WMA** | Breakout | Enter when price breaks above WMA, exit on deep pullback | ~87% | Months–years |
| **RMA ATR Bands** | Trend | Asymmetric ATR channel — enter on upper band breakout | ~64% | Weeks–months |
| **EMA Trail** | Trend | EMA crossover entry with ratcheting trailing stop | ~50–60% | Weeks |

All strategies are long-only, optimized across 10 large-cap US stocks (NVDA, GOOG, AAPL, MSFT, AMZN, AVGO, META, TSLA, AMD, NFLX), backtested 2021–2026 on daily bars with $1,500 capital, 33% risk per trade, and 1.5% commission per leg.

---

## Quick Start

```bash
# 1. Create the conda environment
conda env create -f requirements_conda.yml
conda activate trading

# 2. Visualize a backtest
python visualize.py --strategy elektro --ticker NVDA

# 3. Compare all strategies on one ticker
python compare.py --ticker NVDA

# 4. Run the live signal scanner
python strategy/strategy_elektro/scanner.py
```

---

## Workflows

### Visualize a backtest
Plot price + indicator overlay, equity curve, and drawdown for any strategy and ticker.

```bash
python visualize.py --strategy <name> --ticker <TICKER> [--start YYYY-MM-DD] [--save]
```

Strategies: `rma_atr` · `ema_trail` · `bb_wma` · `elektro` · `dca`

```bash
python visualize.py --strategy rma_atr   --ticker NVDA
python visualize.py --strategy bb_wma    --ticker AVGO --start 2022-01-01
python visualize.py --strategy elektro   --ticker AAPL --save   # saves PNG
```

### Compare all strategies on one ticker
Shows all 5 equity curves on a single chart with a summary metrics table.

```bash
python compare.py --ticker NVDA
python compare.py --ticker MSFT --save
```

### Run the daily scanner
Run after market close (4pm ET) to find fresh signals across the S&P 500.

```bash
python strategy/strategy_rma_atr/scanner.py
python strategy/strategy_ema_trail/scanner.py
python strategy/strategy_bb_wma/scanner.py
python strategy/strategy_elektro/scanner.py
python strategy/strategy_dca/scanner.py
```

The scanner reports **FRESH** signals (fired within the last few days — actionable entries) and **ACTIVE** positions (already in a trade), ranked by confidence.

Scan a specific list of tickers:
```bash
python strategy/strategy_elektro/scanner.py --tickers NVDA AAPL MSFT AVGO
```

### Re-optimize parameters
Run the cross-asset grid search to find better parameters. Results are saved as a CSV.

```bash
python strategy/strategy_rma_atr/tests/test_largecap.py
python strategy/strategy_elektro/tests/test_largecap.py --jobs 4
```

### TradingView Pine Scripts
Each strategy has a ready-to-paste Pine Script in its folder. Open TradingView, create a new indicator, paste the script, and the default inputs are set to the optimized parameters.

```
strategy/strategy_rma_atr/pine_rma_atr_largecap.pine
strategy/strategy_ema_trail/pine_ema_trail_largecap.pine
strategy/strategy_bb_wma/pine_bb_wma_largecap.pine
strategy/strategy_elektro/pine_elektro_largecap.pine
strategy/strategy_dca/pine_dca_largecap.pine
```

---

## Folder Structure

```
strategy/
  strategy_<name>/
    strategy_<name>.py     # backtest engine (run_backtest)
    tests/
      test_largecap.py     # cross-asset parameter optimizer
    scanner.py             # live S&P 500 signal scanner
    pine_<name>_largecap.pine  # TradingView Pine Script
visualize.py               # single-strategy chart
compare.py                 # fleet comparison chart
requirements_conda.yml     # conda environment spec
```

---

## Install

```bash
conda env create -f requirements_conda.yml
```

Requires Python 3.10+. Key dependencies: `numpy`, `pandas`, `yfinance`, `matplotlib`, `seaborn`, `tqdm`.

> **Note:** This project uses [yfinance](https://github.com/ranaroussi/yfinance) for data, which is free for personal and educational use. Commercial use requires a licensed data provider.
