# Trading Strategy Testing Grounds

This folder is the **backend / testing layer** for trading strategies.
TradingView is the frontend — run the optimizer here, then paste the best parameters into the Pine Script on TradingView.

## Strategies

### Keltner Channel (`strategy_keltner/`)
A breakout strategy. Draws a volatility channel around a moving average. Enters long when price breaks above the upper band, short when it breaks below. Exits when price crosses back through the MA.
- Long + short
- Standard commissions (~0.1% per leg)
- Run: `python strategy_keltner/test_keltner.py`

### Pullback / Buy the Dip (`strategy_pullback/`)
Looks for stocks that trended up over ~1 month then pulled back, and buys the expected bounce via a buy-stop. Long only.
- Long only
- High forex commissions (1.5% per leg, 3% round-trip)
- Cross-asset optimized across: NVDA, GOOG, AAPL, MSFT, AMZN, AVGO
- Run: `python strategy_pullback/test_pullback.py`

## Setup

**pip:**
```bash
pip install -r requirements.txt
```

**conda:**
```bash
conda env create -f requirements_conda.yml
conda activate trading
```

## Workflow
1. Run the `test_<name>.py` optimizer for a strategy
2. Check the top result printed in the terminal (or open `result_<name>.csv`)
3. Plug the best parameters into `pine_<name>.pine`
4. Copy the Pine Script into TradingView

## Project Structure

```
Trading/
├── strategy_keltner/
│   ├── strategy_keltner.py   # backtest engine
│   ├── test_keltner.py       # optimizer (run this)
│   ├── pine_keltner.pine     # TradingView script
│   └── result_keltner.csv    # latest optimizer output
├── strategy_pullback/
│   ├── strategy_pullback.py  # backtest engine
│   ├── test_pullback.py      # optimizer (run this)
│   ├── pine_pullback.pine    # TradingView script
│   └── result_pullback.csv   # latest optimizer output
├── requirements.txt
├── requirements_conda.yml
├── CONVENTIONS.md            # naming rules + how to add new strategies
└── README.md
```

See [CONVENTIONS.md](CONVENTIONS.md) for naming rules and how to add new strategies.
