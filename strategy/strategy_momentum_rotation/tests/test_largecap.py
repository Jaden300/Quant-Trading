"""
test_largecap.py — Momentum rotation optimizer across large-cap tech stocks.

Unlike the per-stock strategies, this runs ONE portfolio backtest per combo:
all 10 stocks are visible simultaneously and the strategy picks which to hold.

Usage:
    python strategy_momentum_rotation/tests/test_largecap.py
    python strategy_momentum_rotation/tests/test_largecap.py --jobs 4
"""

from __future__ import annotations

import argparse
import itertools
import multiprocessing as mp
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from strategy_momentum_rotation import run_backtest

RESULTS_FILE = Path(__file__).parent.parent / "result_momentum_rotation_largecap.csv"

DEFAULT_TICKERS = ["NVDA", "GOOG", "AAPL", "MSFT", "AMZN", "AVGO",
                   "META", "TSLA", "AMD", "NFLX"]
WARMUP_START    = "2018-07-01"   # 6-month warmup before test start
DEFAULT_START   = "2019-01-01"   # ~7 years of live data for statistical confidence
DEFAULT_END     = "2026-04-30"
INITIAL_CAPITAL = 1500.0
COMMISSION      = 0.015

PARAM_GRID: dict[str, list] = {
    "lookback":    [20, 40, 63, 126],   # 1M, 2M, 3M, 6M in trading days
    "top_n":       [1, 2, 3],           # how many stocks to hold simultaneously
    "hold_buffer": [0, 1, 2],           # extra rank tolerance before selling
    "rebal_every": [5, 10, 21],         # check for rotation every N bars
}

METRIC_COLS = [
    "sharpe_ratio", "total_return", "max_drawdown",
    "win_rate", "num_trades", "mean_hold_bars", "profit_factor", "final_equity",
]


def download_all(tickers: list[str], warmup_start: str, trade_start: str, end: str) -> tuple[dict, int]:
    print(f"Downloading {tickers}  [{warmup_start} → {end}]  (trades start {trade_start}) …")
    all_data: dict[str, pd.DataFrame] = {}
    trade_start_idx = 0

    for ticker in tickers:
        raw = yf.download(ticker, start=warmup_start, end=end,
                          interval="1d", auto_adjust=True, progress=False)
        if raw.empty:
            print(f"  WARNING: no data for {ticker}, skipping.")
            continue
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        df = raw[["Open", "High", "Low", "Close", "Volume"]].dropna()
        all_data[ticker] = df

    if not all_data:
        return all_data, 0

    # All US stocks share the same trading calendar — use first ticker as reference
    ref = all_data[next(iter(all_data))]
    trade_start_idx = int(ref.index.searchsorted(pd.Timestamp(trade_start)))

    for ticker, df in all_data.items():
        live_date = df.index[trade_start_idx].date() if trade_start_idx < len(df) else "N/A"
        print(f"  {ticker}: {len(df)} bars  warmup={df.index[0].date()}  live={live_date}")

    return all_data, trade_start_idx


def _worker(args: tuple) -> dict:
    all_data, params, trade_start_idx = args
    try:
        m = run_backtest(all_data, params, INITIAL_CAPITAL, COMMISSION, trade_start_idx)
        return {**params, **m}
    except Exception as e:
        return {**params, "_error": str(e)}


def grid_search(all_data: dict, trade_start_idx: int, n_jobs: int = -1) -> pd.DataFrame:
    keys   = list(PARAM_GRID.keys())
    combos = [dict(zip(keys, vals)) for vals in itertools.product(*PARAM_GRID.values())]
    total  = len(combos)
    print(f"\nGrid search: {total} combinations (1 portfolio backtest each) …")

    arg_list = [(all_data, p, trade_start_idx) for p in combos]

    if n_jobs == 1:
        results = [_worker(a) for a in tqdm(arg_list, unit="combo")]
    else:
        workers = mp.cpu_count() if n_jobs == -1 else max(1, n_jobs)
        with mp.Pool(workers) as pool:
            results = list(tqdm(pool.imap(_worker, arg_list, chunksize=8),
                                total=total, unit="combo"))

    df = pd.DataFrame(results)
    if "_error" in df.columns:
        n_err = df["_error"].notna().sum()
        if n_err:
            print(f"Warning: {n_err} combo(s) errored.")
        df = df[df["_error"].isna()].drop(columns=["_error"])

    df = (df[keys + METRIC_COLS]
          .sort_values("sharpe_ratio", ascending=False)
          .reset_index(drop=True))
    return df


def print_top(df: pd.DataFrame, all_data: dict, trade_start_idx: int, n: int = 10) -> None:
    print(f"\n{'─' * 120}")
    print(f"  TOP {n} MOMENTUM ROTATION COMBINATIONS  (ranked by Sharpe)")
    print(f"{'─' * 120}")
    cols = [c for c in [
        "lookback", "top_n", "hold_buffer", "rebal_every",
        "sharpe_ratio", "total_return", "max_drawdown",
        "win_rate", "num_trades", "mean_hold_bars", "final_equity",
    ] if c in df.columns]
    top = df.head(n)[cols].copy()
    top.index = range(1, len(top) + 1)
    print(top.to_string(float_format=lambda x: f"{x:7.2f}", max_colwidth=10))
    print(f"{'─' * 120}\n")

    best = df.iloc[0]
    best_params = {
        "lookback":    int(best["lookback"]),
        "top_n":       int(best["top_n"]),
        "hold_buffer": int(best["hold_buffer"]),
        "rebal_every": int(best["rebal_every"]),
    }
    print(f"  Best params: {best_params}")

    m = run_backtest(all_data, best_params, INITIAL_CAPITAL, COMMISSION,
                     trade_start_idx, return_equity_curve=False)
    print(f"\n  Portfolio result:")
    print(f"  Sharpe {m['sharpe_ratio']:.3f}  |  Return {m['total_return']:.1f}%  "
          f"|  MaxDD {m['max_drawdown']:.1f}%  |  WinRate {m['win_rate']:.0f}%  "
          f"|  Trades {m['num_trades']}  |  AvgHold {m['mean_hold_bars']:.0f}d  "
          f"|  Final ${m['final_equity']:.2f}\n")


def main() -> None:
    p = argparse.ArgumentParser(description="Momentum rotation — large-cap optimizer")
    p.add_argument("--tickers", nargs="+", default=DEFAULT_TICKERS)
    p.add_argument("--start",   default=DEFAULT_START)
    p.add_argument("--end",     default=DEFAULT_END)
    p.add_argument("--jobs",    default=-1, type=int)
    p.add_argument("--top",     default=10, type=int)
    args = p.parse_args()

    all_data, trade_start_idx = download_all(args.tickers, WARMUP_START, args.start, args.end)
    if not all_data:
        sys.exit("ERROR: no data downloaded.")

    print(f"Capital: ${INITIAL_CAPITAL:,.0f}  |  Commission: {COMMISSION*100:.1f}% per leg")

    results = grid_search(all_data, trade_start_idx, n_jobs=args.jobs)
    results.to_csv(RESULTS_FILE, index=False)
    print(f"Results saved → {RESULTS_FILE}  ({len(results)} rows)")

    print_top(results, all_data, trade_start_idx, n=args.top)
    print("Done.")


if __name__ == "__main__":
    main()
