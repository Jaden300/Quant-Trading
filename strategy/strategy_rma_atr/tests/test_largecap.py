"""
test_largecap.py — RMA ATR Bands optimizer, cross-asset across large-cap tech stocks.

Tickers: NVDA, GOOG, AAPL, MSFT, AMZN, AVGO
Ranks by score = mean_sharpe − 0.5 × std_sharpe  (balances performance + consistency).

Usage:
    python strategy_rma_atr/tests/test_largecap.py
    python strategy_rma_atr/tests/test_largecap.py --jobs 4
    python strategy_rma_atr/tests/test_largecap.py --tickers TSLA NVDA AAPL
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
from strategy_rma_atr import run_backtest

RESULTS_FILE = Path(__file__).parent.parent / "result_rma_atr_largecap.csv"

DEFAULT_TICKERS = ["NVDA", "GOOG", "AAPL", "MSFT", "AMZN", "AVGO",
                   "META", "TSLA", "AMD", "NFLX"]
WARMUP_START    = "2020-07-01"
DEFAULT_START   = "2021-01-01"
DEFAULT_END     = "2026-04-30"
INITIAL_CAPITAL = 1500.0
COMMISSION      = 0.015   # 1.5% per leg  (3% round-trip)
RISK_PCT        = 0.33    # 33% of available cash deployed per trade

LONG_ONLY = True   # never enter short positions

PARAM_GRID: dict[str, list] = {
    "ma_src":     ["high", "close"],
    "ma_length":  [8, 10, 14, 20],
    "atr_length": [10, 14, 17, 20],
    "upper_mult": [0.8, 1.0, 1.2, 1.5],
    "lower_mult": [1.5, 2.0, 2.5],
    "ma_exit":    [False],
}

METRIC_COLS = [
    "mean_sharpe", "min_sharpe", "std_sharpe", "score",
    "mean_return", "mean_drawdown", "mean_win_rate",
    "total_trades", "mean_hold_bars", "mean_pf", "tickers_tested",
]


def download_all(
    tickers: list[str], warmup_start: str, trade_start: str, end: str
) -> dict[str, pd.DataFrame]:
    """Downloads from warmup_start so trend state builds before trade_start."""
    print(f"Downloading {tickers}  [{warmup_start} → {end}]  "
          f"(trades start {trade_start}) …")
    all_data: dict[str, pd.DataFrame] = {}
    for ticker in tickers:
        raw = yf.download(ticker, start=warmup_start, end=end,
                          interval="1d", auto_adjust=True, progress=False)
        if raw.empty:
            print(f"  WARNING: no data for {ticker}, skipping.")
            continue
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        df = raw[["Open", "High", "Low", "Close", "Volume"]].dropna()
        # Compute index of first bar on or after trade_start
        trade_start_dt = pd.Timestamp(trade_start)
        idx = int(df.index.searchsorted(trade_start_dt))
        all_data[ticker] = {"df": df, "trade_start_idx": idx}
        print(f"  {ticker}: {len(df)} bars  warmup={df.index[0].date()}  "
              f"live={df.index[idx].date() if idx < len(df) else 'N/A'}")
    return all_data


def _worker(args: tuple) -> dict:
    all_data, params = args
    sharpes, returns, drawdowns, win_rates, n_trades, pf_list, hold_list = \
        [], [], [], [], [], [], []

    for entry in all_data.values():
        try:
            m = run_backtest(
                entry["df"], params, INITIAL_CAPITAL, COMMISSION, RISK_PCT,
                trade_start_idx=entry["trade_start_idx"],
                long_only=LONG_ONLY,
            )
            sharpes.append(m["sharpe_ratio"])
            returns.append(m["total_return"])
            drawdowns.append(m["max_drawdown"])
            win_rates.append(m["win_rate"])
            n_trades.append(m["num_trades"])
            hold_list.append(m["mean_hold_bars"])
            pf = m["profit_factor"]
            pf_list.append(pf if pf != float("inf") else 10.0)
        except Exception:
            pass

    if not sharpes:
        return {**params, "_error": "all tickers failed"}

    mean_sharpe = float(np.mean(sharpes))
    std_sharpe  = float(np.std(sharpes))
    mean_wr     = float(np.mean(win_rates))

    return {
        **params,
        "mean_sharpe":    round(mean_sharpe,                4),
        "min_sharpe":     round(float(np.min(sharpes)),     4),
        "std_sharpe":     round(std_sharpe,                 4),
        "score":          round(mean_sharpe - 0.5 * std_sharpe + 4.0 * (mean_wr / 100.0), 4),
        "mean_return":    round(float(np.mean(returns)),    4),
        "mean_drawdown":  round(float(np.mean(drawdowns)),  4),
        "mean_win_rate":  round(mean_wr,                    4),
        "total_trades":   int(sum(n_trades)),
        "mean_hold_bars": round(float(np.mean(hold_list)),  1),
        "mean_pf":        round(float(np.mean(pf_list)),    4),
        "tickers_tested": len(sharpes),
    }


def grid_search(all_data: dict[str, pd.DataFrame], n_jobs: int = -1) -> pd.DataFrame:
    keys   = list(PARAM_GRID.keys())
    combos = [dict(zip(keys, vals)) for vals in itertools.product(*PARAM_GRID.values())]
    total  = len(combos)
    print(f"\nGrid search: {total} combinations × {len(all_data)} tickers "
          f"= {total * len(all_data):,} backtests …")

    arg_list = [(all_data, p) for p in combos]

    if n_jobs == 1:
        results = [_worker(a) for a in tqdm(arg_list, unit="combo")]
    else:
        workers = mp.cpu_count() if n_jobs == -1 else max(1, n_jobs)
        with mp.Pool(workers) as pool:
            results = list(
                tqdm(pool.imap(_worker, arg_list, chunksize=16),
                     total=total, unit="combo")
            )

    df = pd.DataFrame(results)
    if "_error" in df.columns:
        n_err = df["_error"].notna().sum()
        if n_err:
            print(f"Warning: {n_err} combo(s) errored and were excluded.")
        df = df[df["_error"].isna()].drop(columns=["_error"])

    df = (
        df[keys + METRIC_COLS]
        .sort_values("score", ascending=False)
        .reset_index(drop=True)
    )
    return df


def print_top(df: pd.DataFrame, all_data: dict, n: int = 10) -> None:
    print(f"\n{'─' * 140}")
    print(f"  TOP {n} CROSS-ASSET COMBINATIONS  "
          f"(score = mean_sharpe − 0.5×std_sharpe + 4×win_rate)")
    print(f"{'─' * 140}")
    cols = [c for c in [
        "ma_src", "ma_length", "atr_length", "upper_mult", "lower_mult", "ma_exit",
        "score", "mean_sharpe", "min_sharpe", "std_sharpe",
        "mean_return", "mean_drawdown", "mean_win_rate", "total_trades", "mean_hold_bars",
    ] if c in df.columns]
    top = df.head(n)[cols].copy()
    top.index = range(1, len(top) + 1)
    print(top.to_string(float_format=lambda x: f"{x:7.3f}", max_colwidth=8))
    print(f"{'─' * 140}\n")

    best        = df.iloc[0]
    best_params = _row_to_params(best)
    print("  Per-ticker breakdown for best params:")
    print(f"  {'Ticker':<8} {'Sharpe':>8} {'Return%':>9} {'MaxDD%':>8} "
          f"{'WinRate%':>10} {'Trades':>7} {'AvgHold':>8} {'FinalEq':>9}")
    print(f"  {'─' * 75}")
    for ticker, entry in all_data.items():
        m = run_backtest(
            entry["df"], best_params, INITIAL_CAPITAL, COMMISSION, RISK_PCT,
            trade_start_idx=entry["trade_start_idx"],
            long_only=LONG_ONLY,
        )
        print(f"  {ticker:<8} {m['sharpe_ratio']:>8.3f} {m['total_return']:>9.2f} "
              f"{m['max_drawdown']:>8.2f} {m['win_rate']:>10.1f} "
              f"{m['num_trades']:>7} {m['mean_hold_bars']:>7.0f}d   ${m['final_equity']:>7.2f}")
    print()


def _row_to_params(row: pd.Series) -> dict:
    return {
        "ma_src":     str(row["ma_src"]),
        "ma_length":  int(row["ma_length"]),
        "atr_length": int(row["atr_length"]),
        "upper_mult": float(row["upper_mult"]),
        "lower_mult": float(row["lower_mult"]),
        "ma_exit":    bool(row["ma_exit"]),
    }


def main() -> None:
    p = argparse.ArgumentParser(description="RMA ATR Bands — large-cap cross-asset optimizer")
    p.add_argument("--tickers", nargs="+", default=DEFAULT_TICKERS, help="Ticker symbols")
    p.add_argument("--start",   default=DEFAULT_START, help="Start date (YYYY-MM-DD)")
    p.add_argument("--end",     default=DEFAULT_END,   help="End date (YYYY-MM-DD)")
    p.add_argument("--jobs",    default=-1,  type=int, help="Parallel workers (-1 = all CPUs)")
    p.add_argument("--top",     default=10,  type=int, help="Top-N to print")
    args = p.parse_args()

    all_data = download_all(args.tickers, WARMUP_START, args.start, args.end)
    if not all_data:
        sys.exit("ERROR: no data downloaded.")

    print(f"Capital: ${INITIAL_CAPITAL:,.0f}  |  Risk: {RISK_PCT*100:.0f}% per trade  |  "
          f"Commission: {COMMISSION*100:.1f}% per leg  ({COMMISSION*200:.0f}% round-trip)")

    results = grid_search(all_data, n_jobs=args.jobs)

    results.to_csv(RESULTS_FILE, index=False)
    print(f"Results saved → {RESULTS_FILE}  ({len(results)} rows)")

    print_top(results, all_data, n=args.top)
    print("Done.")


if __name__ == "__main__":
    main()
