"""
visualize.py — Backtest visualization for a single strategy + ticker.

Downloads historical data, runs the backtest with the fleet's optimized
parameters, and produces a 3-panel chart:
  • Price + indicator overlay + entry/exit markers
  • Equity curve
  • Drawdown

Usage:
    conda run -n trading python visualize.py --strategy rma_atr --ticker NVDA
    conda run -n trading python visualize.py --strategy elektro --ticker AAPL
    conda run -n trading python visualize.py --strategy dca --ticker MSFT --save
    conda run -n trading python visualize.py --strategy bb_wma --ticker AVGO --start 2022-01-01

Strategies: rma_atr | ema_trail | bb_wma | elektro | dca
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

# ── Fleet params (optimized, matches scanners + Pine Scripts) ─────────────────

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

WARMUP_START    = "2020-07-01"
INITIAL_CAPITAL = 1500.0
COMMISSION      = 0.015
RISK_PCT        = 0.33

# ── Colour palette ────────────────────────────────────────────────────────────

_BG     = "#0d1117"
_PANEL  = "#161b22"
_BORDER = "#30363d"
_MUTED  = "#8b949e"
_TEXT   = "#e6edf3"
_BLUE   = "#58a6ff"
_GREEN  = "#3fb950"
_RED    = "#da3633"
_ORANGE = "#f0883e"
_PURPLE = "#bc8cff"
_YELLOW = "#e3b341"


# ── Helpers ───────────────────────────────────────────────────────────────────

def _load(strategy: str):
    """Add strategy directory to path and return its run_backtest function."""
    mod_dir = Path(__file__).parent / "strategy" / f"strategy_{strategy}"
    if str(mod_dir) not in sys.path:
        sys.path.insert(0, str(mod_dir))
    mod = importlib.import_module(f"strategy_{strategy}")
    return mod.run_backtest


def _download(ticker: str, end: str) -> pd.DataFrame:
    raw = yf.download(ticker, start=WARMUP_START, end=end,
                      interval="1d", auto_adjust=True, progress=False)
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    df = raw[["Open", "High", "Low", "Close", "Volume"]].dropna()
    if df.empty:
        raise SystemExit(f"No data returned for {ticker}.")
    return df


def _overlay(strategy: str, df: pd.DataFrame, params: dict) -> dict[str, pd.Series]:
    """Return named indicator series to plot on the price chart."""
    if strategy == "rma_atr":
        src = df["Close"].ewm(alpha=1.0 / params["ma_length"], adjust=False).mean()
        tr  = pd.concat([
            df["High"] - df["Low"],
            (df["High"] - df["Close"].shift(1)).abs(),
            (df["Low"]  - df["Close"].shift(1)).abs(),
        ], axis=1).max(axis=1)
        atr   = tr.ewm(alpha=1.0 / params["atr_length"], adjust=False).mean()
        upper = src + atr * params["upper_mult"]
        lower = src - atr * params["lower_mult"]
        return {"MA": src, "Upper band": upper, "Lower band": lower}

    if strategy == "ema_trail":
        fast = df["Close"].ewm(span=params["fast_len"], adjust=False).mean()
        slow = df["Close"].ewm(span=params["slow_len"], adjust=False).mean()
        return {f"EMA {params['fast_len']}": fast,
                f"EMA {params['slow_len']}": slow}

    if strategy == "bb_wma":
        n     = params["bb_length"]
        hlc3  = (df["High"] + df["Low"] + df["Close"]) / 3.0
        w     = np.arange(1, n + 1, dtype=float)
        basis = hlc3.rolling(n).apply(lambda x: np.dot(x, w) / w.sum(), raw=True)
        dev   = hlc3.rolling(n).std(ddof=0)
        return {"WMA": basis,
                "Upper band": basis + params["std_buy"]  * dev,
                "Lower band": basis - params["std_sell"] * dev}

    if strategy == "elektro":
        basis = df["Close"].rolling(params["bb_length"]).mean()
        dev   = df["Close"].rolling(params["bb_length"]).std(ddof=0) * params["bb_mult"]
        return {"Basis": basis, "Upper band": basis + dev, "Lower band": basis - dev}

    return {}   # DCA: no fixed overlay


def _style(ax) -> None:
    ax.set_facecolor(_PANEL)
    ax.tick_params(colors=_MUTED, labelsize=8)
    for sp in ax.spines.values():
        sp.set_edgecolor(_BORDER)
    ax.yaxis.label.set_color(_MUTED)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(description="Visualize a strategy backtest.")
    p.add_argument("--strategy", required=True, choices=list(FLEET_PARAMS),
                   metavar="STRATEGY",
                   help="rma_atr | ema_trail | bb_wma | elektro | dca")
    p.add_argument("--ticker",   required=True, help="e.g. NVDA")
    p.add_argument("--start",    default="2021-01-01")
    p.add_argument("--end",      default="2026-04-30")
    p.add_argument("--save",     action="store_true",
                   help="Save PNG to current directory instead of showing")
    args = p.parse_args()

    strategy = args.strategy
    ticker   = args.ticker.upper()
    params   = FLEET_PARAMS[strategy]

    print(f"Downloading {ticker} …")
    df  = _download(ticker, args.end)
    tsi = int(df.index.searchsorted(pd.Timestamp(args.start)))

    run_bt = _load(strategy)
    print(f"Running {FLEET_NAMES[strategy]} backtest …")
    m = run_bt(
        df, params,
        initial_capital=INITIAL_CAPITAL,
        commission=COMMISSION,
        risk_pct=RISK_PCT,
        return_equity_curve=True,
        return_trades=True,
        trade_start_idx=tsi,
    )

    # ── Data prep ─────────────────────────────────────────────────────────
    equity   = np.asarray(m["equity_curve"])
    eq_dates = pd.DatetimeIndex(m.get("eq_dates", df.index[tsi:]))
    n_min    = min(len(equity), len(eq_dates))
    equity, eq_dates = equity[:n_min], eq_dates[:n_min]

    peak     = np.maximum.accumulate(equity)
    drawdown = (equity - peak) / np.where(peak != 0, peak, 1.0) * 100.0

    trades    = m.get("trades", [])
    all_dates = pd.DatetimeIndex(m.get("all_dates", df.index))
    live_from = pd.Timestamp(args.start)

    entry_dates, entry_prices, exit_dates, exit_prices = [], [], [], []
    for t in trades:
        for key, d_list, p_list in [("bar_in",  entry_dates, entry_prices),
                                    ("bar_out", exit_dates,  exit_prices)]:
            idx = t.get(key)
            if idx is not None and idx < len(all_dates):
                d = all_dates[idx]
                if d >= live_from and d in df.index:
                    d_list.append(d)
                    p_list.append(float(df["Close"].loc[d]))

    df_live  = df.loc[df.index >= live_from]
    overlays = _overlay(strategy, df, params)

    # ── Plot ──────────────────────────────────────────────────────────────
    fig, (ax_p, ax_e, ax_d) = plt.subplots(
        3, 1, figsize=(14, 10),
        gridspec_kw={"height_ratios": [55, 25, 20]},
    )
    fig.patch.set_facecolor(_BG)
    for ax in (ax_p, ax_e, ax_d):
        _style(ax)

    title = (
        f"{ticker}  —  {FLEET_NAMES[strategy]}  |  "
        f"{args.start} → {args.end}  |  "
        f"{'+' if m['total_return'] >= 0 else ''}{m['total_return']:.1f}%  "
        f"·  Sharpe {m['sharpe_ratio']:.2f}  "
        f"·  {m['win_rate']:.0f}% win rate  ·  {m['num_trades']} trades"
    )
    fig.suptitle(title, color=_TEXT, fontsize=11, y=0.985)

    # — Price panel —
    ax_p.plot(df_live.index, df_live["Close"],
              color=_BLUE, linewidth=1.1, label="Close", zorder=2)

    ov_colors = [_ORANGE, _GREEN, _RED, _PURPLE, _YELLOW]
    for (label, series), color in zip(overlays.items(), ov_colors):
        s = series.loc[series.index >= live_from].dropna()
        ax_p.plot(s.index, s, color=color, linewidth=0.9, alpha=0.85, label=label)

    if entry_dates:
        ax_p.scatter(entry_dates, entry_prices, marker="^",
                     color=_GREEN, s=70, zorder=5, label="Entry")
    if exit_dates:
        ax_p.scatter(exit_dates, exit_prices, marker="v",
                     color=_RED, s=70, zorder=5, label="Exit")

    ax_p.legend(fontsize=8, facecolor="#21262d", edgecolor=_BORDER,
                labelcolor=_TEXT, loc="upper left")
    ax_p.set_ylabel("Price ($)", fontsize=9)
    plt.setp(ax_p.get_xticklabels(), visible=False)

    # — Equity panel —
    ax_e.plot(eq_dates, equity, color=_BLUE, linewidth=1.5)
    ax_e.axhline(INITIAL_CAPITAL, color=_MUTED, linewidth=0.7,
                 linestyle="--", alpha=0.5)
    ax_e.fill_between(eq_dates, INITIAL_CAPITAL, equity,
                      where=equity >= INITIAL_CAPITAL,
                      alpha=0.15, color=_GREEN)
    ax_e.fill_between(eq_dates, INITIAL_CAPITAL, equity,
                      where=equity < INITIAL_CAPITAL,
                      alpha=0.15, color=_RED)
    ax_e.set_ylabel("Equity ($)", fontsize=9)
    plt.setp(ax_e.get_xticklabels(), visible=False)

    # — Drawdown panel —
    ax_d.fill_between(eq_dates, drawdown, 0, alpha=0.55, color=_RED)
    ax_d.plot(eq_dates, drawdown, color=_RED, linewidth=0.8)
    ax_d.set_ylabel("Drawdown %", fontsize=9)
    ax_d.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax_d.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    plt.setp(ax_d.get_xticklabels(), rotation=30, ha="right",
             fontsize=8, color=_MUTED)

    plt.tight_layout(rect=[0, 0, 1, 0.97])

    if args.save:
        fname = f"{ticker}_{strategy}.png"
        plt.savefig(fname, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        print(f"Saved → {fname}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
