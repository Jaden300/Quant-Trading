"""
strategy_pullback.py — "Buy the dip in an uptrend" backtest engine.

Signal conditions (checked on each bar's close):
  1. Price trended UP over the window from long_period → short_period bars ago
  2. Price dipped DOWN over the last short_period bars
  → Place a buy-stop at close * (1 + stop_buffer)

Entry: filled next bar when high >= buy_stop  (fill = max(open, buy_stop))
Exit:  TP at entry * (1 + tp_pct)  OR  SL at entry * (1 − sl_pct)
       Both checked via bar high/low. If both hit same bar → SL wins (conservative).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def run_backtest(
    data: pd.DataFrame,
    params: dict,
    initial_capital: float = 1500.0,
    commission: float = 0.015,
    risk_pct: float = 0.33,
    return_equity_curve: bool = False,
    trade_start_idx: int = 0,
) -> dict:
    long_period     = int(params["long_period"])
    short_period    = int(params["short_period"])
    trend_threshold = float(params["trend_threshold"])
    dip_threshold   = float(params["dip_threshold"])
    stop_buffer     = float(params["stop_buffer"])
    tp_pct          = float(params["tp_pct"])
    sl_pct          = float(params["sl_pct"])

    close  = data["Close"].values
    high   = data["High"].values
    low    = data["Low"].values
    open_  = data["Open"].values
    n      = len(close)

    warmup = max(long_period + 5, trade_start_idx)

    cash         = initial_capital
    position     = 0
    shares       = 0.0
    entry_price  = 0.0
    tp_level     = 0.0
    sl_level     = 0.0
    cost_basis   = 0.0
    notional     = 0.0
    pending_stop: float | None = None

    equity_curve = np.empty(n)
    equity_curve[0] = initial_capital
    trades: list[dict] = []

    for i in range(1, n):

        # ── 1. Manage open position: check TP / SL ──────────────────────
        if position == 1:
            hit_tp = high[i] >= tp_level
            hit_sl = low[i]  <= sl_level

            if hit_tp or hit_sl:
                exit_raw  = sl_level if hit_sl else tp_level
                reason    = "SL"     if hit_sl else "TP"
                exit_eff  = exit_raw * (1.0 - commission)
                proceeds  = shares * exit_eff
                pnl       = proceeds - cost_basis
                cash     += proceeds
                if trades and "pnl" not in trades[-1]:
                    trades[-1].update({"exit": exit_raw, "pnl": pnl,
                                       "bar_out": i, "reason": reason})
                shares   = 0.0
                notional = 0.0
                position = 0

        # ── 2. Process pending buy-stop ──────────────────────────────────
        if position == 0 and pending_stop is not None:
            long_idx  = i - long_period
            short_idx = i - short_period
            if long_idx >= 0 and short_idx >= 0:
                trend_ok = (close[short_idx] / close[long_idx] - 1.0) >= trend_threshold
                dip_ok   = (close[i]         / close[short_idx] - 1.0) <= -dip_threshold
            else:
                trend_ok = dip_ok = False

            if not (trend_ok and dip_ok):
                pending_stop = None
            elif high[i] >= pending_stop:
                fill        = max(open_[i], pending_stop)
                eff_entry   = fill * (1.0 + commission)
                notional    = cash * risk_pct
                shares      = notional / eff_entry
                cost_basis  = notional
                cash       -= notional
                entry_price = fill
                tp_level    = fill * (1.0 + tp_pct)
                sl_level    = fill * (1.0 - sl_pct)
                position    = 1
                pending_stop = None
                trades.append({"dir": 1, "entry": fill, "cost_basis": cost_basis, "bar_in": i})

        # ── 3. Generate new signal (flat, no pending order) ──────────────
        if position == 0 and pending_stop is None and i >= warmup:
            long_idx  = i - long_period
            short_idx = i - short_period
            if long_idx >= 0 and short_idx >= 0:
                trend_ok = (close[short_idx] / close[long_idx] - 1.0) >= trend_threshold
                dip_ok   = (close[i]         / close[short_idx] - 1.0) <= -dip_threshold
                if trend_ok and dip_ok:
                    pending_stop = close[i] * (1.0 + stop_buffer)

        equity_curve[i] = cash + shares * close[i]

    # Force-close at last bar
    if position == 1:
        exit_raw  = close[-1]
        exit_eff  = exit_raw * (1.0 - commission)
        proceeds  = shares * exit_eff
        pnl       = proceeds - cost_basis
        cash     += proceeds
        if trades and "pnl" not in trades[-1]:
            trades[-1].update({"exit": exit_raw, "pnl": pnl,
                               "bar_out": n - 1, "reason": "EOD"})
        shares   = 0.0

    equity_curve[-1] = cash

    metrics = _calc_metrics(equity_curve, trades, initial_capital)
    if return_equity_curve:
        metrics["equity_curve"] = equity_curve
    return metrics


def _calc_metrics(
    equity: np.ndarray, trades: list[dict], initial_capital: float
) -> dict:
    total_return = (equity[-1] / initial_capital - 1.0) * 100.0

    rets  = np.diff(equity) / np.where(equity[:-1] != 0, equity[:-1], 1.0)
    std   = rets.std()
    sharpe = float(rets.mean() / std * np.sqrt(252)) if std > 0 else 0.0

    peak  = np.maximum.accumulate(equity)
    dd    = (equity - peak) / np.where(peak != 0, peak, 1.0)
    max_dd = float(dd.min() * 100.0)

    closed = [t for t in trades if "pnl" in t]
    n_trades = len(closed)
    if n_trades:
        wins          = [t["pnl"] for t in closed if t["pnl"] > 0]
        losses        = [t["pnl"] for t in closed if t["pnl"] <= 0]
        win_rate      = len(wins) / n_trades * 100.0
        gross_profit  = sum(wins)
        gross_loss    = abs(sum(losses))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")
        hold_bars     = [t["bar_out"] - t["bar_in"]
                         for t in closed if "bar_out" in t and "bar_in" in t]
        mean_hold = float(np.mean(hold_bars)) if hold_bars else 0.0
    else:
        win_rate = profit_factor = mean_hold = 0.0

    return {
        "total_return":   round(total_return,  4),
        "sharpe_ratio":   round(sharpe,         4),
        "max_drawdown":   round(max_dd,          4),
        "win_rate":       round(win_rate,         4),
        "num_trades":     n_trades,
        "profit_factor":  round(profit_factor,   4),
        "mean_hold_bars": round(mean_hold,        1),
        "final_equity":   round(float(equity[-1]), 2),
    }
