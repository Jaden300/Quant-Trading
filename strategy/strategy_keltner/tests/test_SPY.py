"""
test_SPY.py — Keltner Channel optimizer, single-ticker, defaulting to SPY.

Usage:
    python strategy_keltner/tests/test_SPY.py
    python strategy_keltner/tests/test_SPY.py --ticker QQQ --start 2015-01-01 --end 2024-12-31
    python strategy_keltner/tests/test_SPY.py --ticker SPY --jobs 4 --capital 50000
"""

from __future__ import annotations

import argparse
import itertools
import multiprocessing as mp
import sys
from pathlib import Path

import pandas as pd
import yfinance as yf
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from strategy_keltner import run_backtest

RESULTS_FILE = Path(__file__).parent.parent / "result_keltner_SPY.csv"

PARAM_GRID: dict[str, list] = {
    "length":      [10, 15, 20, 25, 30],
    "mult":        [1.5, 2.0, 2.5, 3.0],
    "atr_length":  [7, 10, 14, 20],
    "use_ema":     [True, False],
    "bands_style": ["ATR", "TR", "Range"],
}

METRIC_COLS = [
    "total_return", "sharpe_ratio", "max_drawdown",
    "win_rate", "num_trades", "profit_factor", "final_equity",
]


def _worker(args: tuple) -> dict:
    data, params, initial_capital, commission = args
    try:
        metrics = run_backtest(data, params, initial_capital, commission)
        return {**params, **metrics}
    except Exception as exc:
        return {**params, "_error": str(exc)}


def grid_search(
    data: pd.DataFrame,
    n_jobs: int = -1,
    initial_capital: float = 100_000.0,
    commission: float = 0.001,
) -> pd.DataFrame:
    keys   = list(PARAM_GRID.keys())
    combos = [dict(zip(keys, vals)) for vals in itertools.product(*PARAM_GRID.values())]
    total  = len(combos)
    print(f"Grid search: {total} combinations over {len(data)} bars …")

    arg_list = [(data, p, initial_capital, commission) for p in combos]

    if n_jobs == 1:
        results = [_worker(a) for a in tqdm(arg_list, total=total, unit="combo")]
    else:
        workers = mp.cpu_count() if n_jobs == -1 else max(1, n_jobs)
        with mp.Pool(processes=workers) as pool:
            results = list(
                tqdm(pool.imap(_worker, arg_list, chunksize=16), total=total, unit="combo")
            )

    df = pd.DataFrame(results)
    if "_error" in df.columns:
        n_failed = df["_error"].notna().sum()
        if n_failed:
            print(f"Warning: {n_failed} combination(s) failed and are excluded.")
        df = df[df["_error"].isna()].drop(columns=["_error"])

    df = (
        df[keys + METRIC_COLS]
        .sort_values("sharpe_ratio", ascending=False)
        .reset_index(drop=True)
    )
    return df


def print_top(df: pd.DataFrame, n: int = 10) -> None:
    print(f"\n{'─' * 110}")
    print(f"  TOP {n} PARAMETER COMBINATIONS  (ranked by Sharpe Ratio)")
    print(f"{'─' * 110}")
    display_cols = [
        "length", "mult", "atr_length", "use_ema", "bands_style",
        "sharpe_ratio", "total_return", "max_drawdown", "win_rate",
        "num_trades", "profit_factor",
    ]
    cols = [c for c in display_cols if c in df.columns]
    top = df.head(n)[cols].copy()
    top.index = range(1, len(top) + 1)
    print(top.to_string(float_format=lambda x: f"{x:8.3f}", max_colwidth=12))
    print(f"{'─' * 110}\n")


def download_data(ticker: str, start: str, end: str, interval: str) -> pd.DataFrame:
    print(f"Downloading {ticker}  [{start} → {end}]  interval={interval} …")
    raw = yf.download(ticker, start=start, end=end, interval=interval,
                      auto_adjust=True, progress=False)
    if raw.empty:
        sys.exit(f"ERROR: yfinance returned no data for {ticker!r}.")
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    needed = {"Open", "High", "Low", "Close", "Volume"}
    missing = needed - set(raw.columns)
    if missing:
        sys.exit(f"ERROR: Missing columns: {missing}")
    data = raw[["Open", "High", "Low", "Close", "Volume"]].dropna()
    print(f"  {len(data)} bars  ({data.index[0].date()} → {data.index[-1].date()})")
    return data


def main() -> None:
    p = argparse.ArgumentParser(
        description="Keltner Channel strategy — SPY grid-search optimizer",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--ticker",     default="SPY",        help="Ticker symbol")
    p.add_argument("--start",      default="2018-01-01", help="Start date (YYYY-MM-DD)")
    p.add_argument("--end",        default="2024-12-31", help="End date (YYYY-MM-DD)")
    p.add_argument("--interval",   default="1d",         help="yfinance interval")
    p.add_argument("--capital",    default=100_000.0,    type=float, help="Initial capital ($)")
    p.add_argument("--commission", default=0.001,        type=float, help="Per-leg commission rate")
    p.add_argument("--jobs",       default=-1,           type=int,   help="Parallel workers (-1 = all CPUs)")
    p.add_argument("--top",        default=10,           type=int,   help="Top-N results to print")
    args = p.parse_args()

    data    = download_data(args.ticker, args.start, args.end, args.interval)
    results = grid_search(data, n_jobs=args.jobs,
                          initial_capital=args.capital, commission=args.commission)

    results.to_csv(RESULTS_FILE, index=False)
    print(f"Results saved → {RESULTS_FILE}  ({len(results)} rows)")

    print_top(results, n=args.top)
    print("Done.")


if __name__ == "__main__":
    main()
