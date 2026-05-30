"""
test_largecap.py — DCA Long optimizer, cross-asset large-cap tech.

Usage:
    python strategy_dca/tests/test_largecap.py
    python strategy_dca/tests/test_largecap.py --jobs 4
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
from strategy_dca import run_backtest

RESULTS_FILE = Path(__file__).parent.parent / "result_dca_largecap.csv"

DEFAULT_TICKERS = ["NVDA", "GOOG", "AAPL", "MSFT", "AMZN", "AVGO",
                   "META", "TSLA", "AMD", "NFLX"]
WARMUP_START    = "2020-07-01"
DEFAULT_START   = "2021-01-01"
DEFAULT_END     = "2026-04-30"
INITIAL_CAPITAL = 1500.0
COMMISSION      = 0.015
RISK_PCT        = 0.33

PARAM_GRID: dict[str, list] = {
    "tp_pct":  [22.0, 26.0, 30.0, 35.0],
    "so_step": [15.0, 18.0, 22.0],
    "max_so":  [5, 6, 7],
}

METRIC_COLS = [
    "mean_sharpe", "min_sharpe", "std_sharpe", "score",
    "mean_return", "mean_drawdown", "mean_win_rate",
    "total_trades", "mean_hold_bars", "mean_pf", "tickers_tested",
]


def download_all(
    tickers: list[str], warmup_start: str, trade_start: str, end: str
) -> dict:
    print(f"Downloading {tickers}  [{warmup_start} → {end}]  "
          f"(trades start {trade_start}) …")
    all_data: dict = {}
    for ticker in tickers:
        raw = yf.download(ticker, start=warmup_start, end=end,
                          interval="1d", auto_adjust=True, progress=False)
        if raw.empty:
            print(f"  WARNING: no data for {ticker}, skipping.")
            continue
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        df = raw[["Open", "High", "Low", "Close", "Volume"]].dropna()
        idx = int(df.index.searchsorted(pd.Timestamp(trade_start)))
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

    return {
        **params,
        "mean_sharpe":    round(mean_sharpe,                    4),
        "min_sharpe":     round(float(np.min(sharpes)),         4),
        "std_sharpe":     round(std_sharpe,                     4),
        "score":          round(mean_sharpe - 0.5 * std_sharpe, 4),
        "mean_return":    round(float(np.mean(returns)),        4),
        "mean_drawdown":  round(float(np.mean(drawdowns)),      4),
        "mean_win_rate":  round(float(np.mean(win_rates)),      4),
        "total_trades":   int(sum(n_trades)),
        "mean_hold_bars": round(float(np.mean(hold_list)),      1),
        "mean_pf":        round(float(np.mean(pf_list)),        4),
        "tickers_tested": len(sharpes),
    }


def grid_search(all_data: dict, n_jobs: int = -1) -> pd.DataFrame:
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
                tqdm(pool.imap(_worker, arg_list, chunksize=8),
                     total=total, unit="combo")
            )

    df = pd.DataFrame(results)
    if "_error" in df.columns:
        n_err = df["_error"].notna().sum()
        if n_err:
            print(f"Warning: {n_err} combo(s) errored and were excluded.")
        df = df[df["_error"].isna()].drop(columns=["_error"])

    return (
        df[keys + METRIC_COLS]
        .sort_values("score", ascending=False)
        .reset_index(drop=True)
    )


def print_top(df: pd.DataFrame, all_data: dict, n: int = 10) -> None:
    print(f"\n{'─' * 140}")
    print(f"  TOP {n} DCA LONG COMBINATIONS  (score = mean_sharpe − 0.5×std_sharpe)")
    print(f"{'─' * 140}")
    cols = [c for c in [
        "tp_pct", "so_step", "max_so",
        "score", "mean_sharpe", "min_sharpe", "std_sharpe",
        "mean_return", "mean_drawdown", "mean_win_rate", "total_trades", "mean_hold_bars",
    ] if c in df.columns]
    top = df.head(n)[cols].copy()
    top.index = range(1, len(top) + 1)
    print(top.to_string(float_format=lambda x: f"{x:7.3f}", max_colwidth=12))
    print(f"{'─' * 140}\n")

    best = df.iloc[0]
    best_params = {
        "tp_pct":  float(best["tp_pct"]),
        "so_step": float(best["so_step"]),
        "max_so":  int(best["max_so"]),
    }
    print("  Per-ticker breakdown for best params:")
    print(f"  {'Ticker':<8} {'Sharpe':>8} {'Return%':>9} {'MaxDD%':>8} "
          f"{'WinRate%':>10} {'Trades':>7} {'AvgHold':>8} {'FinalEq':>9}")
    print(f"  {'─' * 75}")
    for ticker, entry in all_data.items():
        m = run_backtest(
            entry["df"], best_params, INITIAL_CAPITAL, COMMISSION, RISK_PCT,
            trade_start_idx=entry["trade_start_idx"],
        )
        print(f"  {ticker:<8} {m['sharpe_ratio']:>8.3f} {m['total_return']:>9.2f} "
              f"{m['max_drawdown']:>8.2f} {m['win_rate']:>10.1f} "
              f"{m['num_trades']:>7} {m['mean_hold_bars']:>7.0f}d   ${m['final_equity']:>7.2f}")
    print()


def main() -> None:
    p = argparse.ArgumentParser(description="DCA Long — large-cap optimizer")
    p.add_argument("--tickers", nargs="+", default=DEFAULT_TICKERS)
    p.add_argument("--start",   default=DEFAULT_START)
    p.add_argument("--end",     default=DEFAULT_END)
    p.add_argument("--jobs",    default=-1, type=int)
    p.add_argument("--top",     default=10, type=int)
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
