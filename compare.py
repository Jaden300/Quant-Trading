"""
compare.py — Compare all 5 fleet strategies on a single ticker.

Runs each strategy's backtest with the optimized parameters and shows
equity curves side-by-side, plus a summary metrics table below.

Usage:
    conda run -n trading python compare.py --ticker NVDA
    conda run -n trading python compare.py --ticker AAPL --start 2022-01-01
    conda run -n trading python compare.py --ticker MSFT --save
"""

from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf

# ── Fleet config (same params as scanners + Pine Scripts) ─────────────────────

FLEET_PARAMS: dict[str, dict] = {
    "rma_atr":   {"ma_src": "close", "ma_length": 8, "atr_length": 20,
                  "upper_mult": 1.5, "lower_mult": 2.5, "ma_exit": False},
    "ema_trail": {"fast_len": 10, "slow_len": 50, "tp_pct": 0.30, "sl_pct": 0.10,
                  "trail_trigger_pct": 0.06, "trail_step_pct": 0.04},
    "bb_wma":    {"bb_length": 150, "std_buy": 1.0, "std_sell": 3.0},
    "elektro":   {"bb_length": 350, "bb_mult": 1.5, "rsi_length": 35,
                  "rsi_oversold": 40.0, "rsi_exit": 80.0},
    "dca":       {"tp_pct": 30.0, "so_step": 18.0, "max_so": 5},
}

FLEET_NAMES: dict[str, str] = {
    "rma_atr":   "RMA ATR Bands",
    "ema_trail": "EMA Trail",
    "bb_wma":    "Bollinger WMA",
    "elektro":   "Elektro BB",
    "dca":       "DCA Long",
}

FLEET_COLORS: dict[str, str] = {
    "rma_atr":   "#f0883e",
    "ema_trail": "#58a6ff",
    "bb_wma":    "#3fb950",
    "elektro":   "#bc8cff",
    "dca":       "#ffa657",
}

WARMUP_START    = "2020-07-01"
INITIAL_CAPITAL = 1500.0
COMMISSION      = 0.015
RISK_PCT        = 0.33

_BG     = "#0d1117"
_PANEL  = "#161b22"
_BORDER = "#30363d"
_MUTED  = "#8b949e"
_TEXT   = "#e6edf3"
_GREEN  = "#3fb950"
_RED    = "#da3633"


def _load(strategy: str):
    mod_dir = Path(__file__).parent / "strategy" / f"strategy_{strategy}"
    if str(mod_dir) not in sys.path:
        sys.path.insert(0, str(mod_dir))
    return importlib.import_module(f"strategy_{strategy}").run_backtest


def main() -> None:
    p = argparse.ArgumentParser(description="Compare all 5 strategies on one ticker.")
    p.add_argument("--ticker", required=True, help="e.g. NVDA")
    p.add_argument("--start",  default="2021-01-01")
    p.add_argument("--end",    default="2026-04-30")
    p.add_argument("--save",   action="store_true",
                   help="Save PNG to current directory instead of showing")
    args = p.parse_args()

    ticker = args.ticker.upper()
    print(f"Downloading {ticker} …")

    raw = yf.download(ticker, start=WARMUP_START, end=args.end,
                      interval="1d", auto_adjust=True, progress=False)
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    df = raw[["Open", "High", "Low", "Close", "Volume"]].dropna()
    if df.empty:
        raise SystemExit(f"No data returned for {ticker}.")

    tsi = int(df.index.searchsorted(pd.Timestamp(args.start)))

    results: dict[str, dict] = {}
    for name in FLEET_PARAMS:
        print(f"  {FLEET_NAMES[name]} …")
        run_bt = _load(name)
        results[name] = run_bt(
            df, FLEET_PARAMS[name],
            initial_capital=INITIAL_CAPITAL,
            commission=COMMISSION,
            risk_pct=RISK_PCT,
            return_equity_curve=True,
            trade_start_idx=tsi,
        )

    # ── Figure ────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(14, 9))
    fig.patch.set_facecolor(_BG)

    gs     = fig.add_gridspec(2, 1, height_ratios=[62, 38], hspace=0.08)
    ax_eq  = fig.add_subplot(gs[0])
    ax_tbl = fig.add_subplot(gs[1])

    for ax in (ax_eq, ax_tbl):
        ax.set_facecolor(_PANEL)
        ax.tick_params(colors=_MUTED, labelsize=8)
        for sp in ax.spines.values():
            sp.set_edgecolor(_BORDER)

    ax_tbl.axis("off")

    fig.suptitle(
        f"Fleet Comparison  —  {ticker}  |  {args.start} → {args.end}",
        color=_TEXT, fontsize=12, y=0.985,
    )

    # ── Equity curves ─────────────────────────────────────────────────────
    for name, m in results.items():
        eq       = np.asarray(m["equity_curve"])
        eq_dates = pd.DatetimeIndex(m.get("eq_dates", df.index[tsi:]))
        n_min    = min(len(eq), len(eq_dates))
        ax_eq.plot(eq_dates[:n_min], eq[:n_min],
                   color=FLEET_COLORS[name], linewidth=1.6,
                   label=FLEET_NAMES[name])

    ax_eq.axhline(INITIAL_CAPITAL, color=_MUTED, linewidth=0.7,
                  linestyle="--", alpha=0.4)
    ax_eq.set_ylabel("Equity ($)", fontsize=9, color=_MUTED)
    ax_eq.legend(fontsize=9, facecolor="#21262d", edgecolor=_BORDER,
                 labelcolor=_TEXT, loc="upper left")
    ax_eq.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax_eq.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    plt.setp(ax_eq.get_xticklabels(), rotation=30, ha="right",
             fontsize=8, color=_MUTED)

    # ── Metrics table ─────────────────────────────────────────────────────
    rows = []
    for name, m in results.items():
        ret = m["total_return"]
        rows.append([
            FLEET_NAMES[name],
            f"{'+' if ret >= 0 else ''}{ret:.1f}%",
            f"{m['sharpe_ratio']:.3f}",
            f"{m['win_rate']:.1f}%",
            f"{m['max_drawdown']:.1f}%",
            str(m["num_trades"]),
            f"${m['final_equity']:,.0f}",
        ])

    col_labels = ["Strategy", "Return", "Sharpe", "Win Rate",
                  "Max DD", "Trades", "Final Equity"]

    tbl = ax_tbl.table(
        cellText=rows,
        colLabels=col_labels,
        loc="center",
        cellLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1, 2.0)

    for (row, col), cell in tbl.get_celld().items():
        cell.set_edgecolor(_BORDER)
        if row == 0:
            cell.set_facecolor("#21262d")
            cell.set_text_props(color=_TEXT, fontweight="bold")
        else:
            cell.set_facecolor(_PANEL)
            strategy_key = list(FLEET_PARAMS.keys())[row - 1]
            cell.set_text_props(color=FLEET_COLORS[strategy_key])

    fig.subplots_adjust(top=0.94, bottom=0.04, left=0.07, right=0.98, hspace=0.08)

    if args.save:
        fname = f"{ticker}_fleet_comparison.png"
        plt.savefig(fname, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        print(f"Saved → {fname}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
