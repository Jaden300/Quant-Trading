# Project Conventions

## Purpose
This folder is the testing/backend layer for trading strategies.
TradingView is the frontend — optimized parameters flow from here into Pine Scripts there.

## The Pipeline
```
strategy_<name>.py  →  tests/test_<testname>.py  →  result_<name>_<testname>.csv  →  pine_<name>_<testname>.pine
     (engine)               (run this)                    (output)                      (paste into TradingView)
```

## Folder Structure

```
strategy_<name>/
├── strategy_<name>.py              — backtest engine and indicator logic
├── tests/
│   └── test_<testname>.py          — optimizer for a specific ticker set / date range
├── result_<name>_<testname>.csv    — output of that test run (overwritten on re-run)
└── pine_<name>_<testname>.pine     — TradingView Pine Script with those best params
```

A strategy can have multiple tests (e.g. test_SPY, test_largecap, test_smallcap).
Each test gets its own result CSV and pine file.

## Current Strategies & Tests

| Strategy folder | Test file | Result CSV | Pine Script |
|---|---|---|---|
| `strategy_keltner/` | `tests/test_SPY.py` | `result_keltner_SPY.csv` | `pine_keltner_SPY.pine` |
| `strategy_pullback/` | `tests/test_largecap.py` | `result_pullback_largecap.csv` | `pine_pullback_largecap.pine` |
| `strategy_rma_atr/` | `tests/test_largecap.py` | `result_rma_atr_largecap.csv` | `pine_rma_atr_largecap.pine` |
| `strategy_momentum_rotation/` | `tests/test_largecap.py` | `result_momentum_rotation_largecap.csv` | `pine_momentum_rotation_largecap.pine` |

## Naming Rules
| Thing | Pattern | Example |
|---|---|---|
| Strategy folder | `strategy_<name>/` | `strategy_keltner/` |
| Backtest engine | `strategy_<name>.py` | `strategy_keltner.py` |
| Tests subfolder | `tests/` | `strategy_keltner/tests/` |
| Test/optimizer | `test_<testname>.py` | `test_SPY.py` |
| Results CSV | `result_<name>_<testname>.csv` | `result_keltner_SPY.csv` |
| Pine Script | `pine_<name>_<testname>.pine` | `pine_keltner_SPY.pine` |

## Setup Files
- `requirements.txt` — pip install list
- `requirements_conda.yml` — conda environment (same packages, conda format)
- `README.md` — project overview

## How to Add a New Strategy
1. Create `strategy_<name>/` and `strategy_<name>/tests/` folders
2. Write `strategy_<name>.py` with a `run_backtest(data, params, ...) -> dict` function
3. Write `tests/test_<testname>.py` — set `sys.path.insert(0, Path(__file__).parent.parent)`, define PARAM_GRID, grid_search, main
4. Run the test to generate `result_<name>_<testname>.csv`
5. Write `pine_<name>_<testname>.pine` with the best params from the CSV plugged in

## How to Add a New Test to an Existing Strategy
1. Create a new `tests/test_<newtestname>.py` in the strategy folder
2. It auto-saves to `result_<name>_<newtestname>.csv` in the strategy folder root
3. Create `pine_<name>_<newtestname>.pine` with the new best params

## Cross-Testing
All backtest engines share the same interface:
  `run_backtest(data: pd.DataFrame, params: dict, initial_capital, commission) -> dict`
You can swap the import in any test file to run a different strategy's engine
through the same optimizer loop.
