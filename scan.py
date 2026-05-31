"""
scan.py - Unified fleet scanner across all 5 strategies.

Downloads data once per ticker, runs all 5 strategies, and outputs a single
confidence-ranked list of fresh signals and active positions.

Usage:
    conda run -n trading python scan.py
    conda run -n trading python scan.py --tickers NVDA AAPL MSFT AVGO
    conda run -n trading python scan.py --top 15
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import yfinance as yf

ROOT = Path(__file__).parent

STRATEGY_NAMES = {
    "rma_atr":   "RMA ATR Bands",
    "ema_trail": "EMA Trail",
    "bb_wma":    "BB WMA",
    "elektro":   "Elektro BB",
    "dca":       "DCA Long",
}

# Minimum bars each strategy needs to produce a valid signal
MIN_BARS = {
    "rma_atr":   60,
    "ema_trail": 60,
    "bb_wma":    160,
    "elektro":   400,
    "dca":       30,
}


def _load_scanner(name: str):
    path = ROOT / "featured" / f"strategy_{name}" / "scanner.py"
    spec = importlib.util.spec_from_file_location(f"scanner_{name}", str(path))
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _load_all() -> dict:
    mods = {}
    for name in STRATEGY_NAMES:
        try:
            mods[name] = _load_scanner(name)
        except Exception as e:
            print(f"  WARNING: could not load {name} scanner: {e}")
    return mods


def get_sp500_tickers(scanners: dict) -> list[str]:
    for mod in scanners.values():
        return mod.get_sp500_tickers()
    return []


def download_batch(tickers: list[str], start: str, end: str) -> dict[str, pd.DataFrame]:
    print(f"  Downloading {len(tickers)} tickers [{start} to {end}] ...")
    raw = yf.download(
        tickers, start=start, end=end,
        interval="1d", auto_adjust=True,
        progress=False, group_by="ticker",
    )
    result: dict[str, pd.DataFrame] = {}
    for ticker in tickers:
        try:
            df = raw if len(tickers) == 1 else raw[ticker]
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
            # Need enough bars for the slowest strategy (Elektro: 400 bars)
            if len(df) >= max(MIN_BARS.values()):
                result[ticker] = df
        except Exception:
            pass
    return result


def scan_ticker(ticker: str, df: pd.DataFrame, scanners: dict) -> list[dict]:
    signals = []
    for name, mod in scanners.items():
        try:
            if len(df) < MIN_BARS[name]:
                continue
            result = mod.scan_ticker(ticker, df)
            if result is not None:
                result["strategy"]     = STRATEGY_NAMES[name]
                result["strategy_key"] = name
                signals.append(result)
        except Exception:
            pass
    return signals


def print_results(signals: list[dict], top: int) -> None:
    fresh  = [s for s in signals if s["fresh"]]
    active = [s for s in signals if not s["fresh"]]

    # DCA is always in the market - only show active DCA positions with conf >= 60
    # to avoid flooding the active list with every S&P 500 ticker
    active = [s for s in active
              if not (s["strategy_key"] == "dca" and s["confidence"] < 60)]

    fresh.sort( key=lambda s: -s["confidence"])
    active.sort(key=lambda s: -s["confidence"])

    def _row(s: dict) -> str:
        tag = "★ FRESH" if s["fresh"] else "  active"
        ret = f"+{s['ret_pct']:.1f}%" if s["ret_pct"] >= 0 else f"{s['ret_pct']:.1f}%"
        return (f"  {tag}   {s['strategy']:<16}  {s['ticker']:<6}  "
                f"conf {s['confidence']:>3}/100  "
                f"  {s['bars_ago']:>4}d   {ret:>8}")

    W = 76
    print(f"\n{'─' * W}")
    print(f"  FLEET SCAN   {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"  All 5 strategies - signals ranked by confidence")
    print(f"{'─' * W}")

    if fresh:
        print(f"\n  FRESH SIGNALS  (entry window open - act within a few days)\n")
        for s in fresh[:top]:
            print(_row(s))
    else:
        print(f"\n  No fresh signals right now.\n")

    if active:
        print(f"\n  ACTIVE POSITIONS  (already in trade - monitor only)\n")
        for s in active[:top]:
            print(_row(s))

    print(f"\n{'─' * W}")
    print(f"  {len(fresh)} fresh  |  {len(active)} active  |  "
          f"{len(fresh) + len(active)} total signals\n")


def main() -> None:
    p = argparse.ArgumentParser(description="Fleet scanner - all 5 strategies unified")
    p.add_argument("--tickers", nargs="+", default=None,
                   help="Specific tickers (default: full S&P 500)")
    p.add_argument("--top",     default=20, type=int,
                   help="Max results per section")
    args = p.parse_args()

    print("Loading scanners ...")
    scanners = _load_all()
    print(f"  {len(scanners)} strategies loaded.\n")

    if args.tickers:
        tickers = [t.upper() for t in args.tickers]
        print(f"Scanning {len(tickers)} specified tickers ...")
    else:
        print("Fetching S&P 500 ticker list ...")
        tickers = get_sp500_tickers(scanners)
        print(f"  {len(tickers)} tickers.\n")

    end   = datetime.today().strftime("%Y-%m-%d")
    start = (datetime.today() - timedelta(days=730)).strftime("%Y-%m-%d")

    all_data = download_batch(tickers, start, end)
    print(f"  {len(all_data)} tickers downloaded.\n")
    print("Scanning ...")

    signals: list[dict] = []
    for ticker, df in all_data.items():
        signals.extend(scan_ticker(ticker, df, scanners))

    print_results(signals, top=args.top)


if __name__ == "__main__":
    main()
