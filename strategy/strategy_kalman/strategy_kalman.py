"""
strategy_kalman.py — Dual Kalman Filter crossover backtest engine (long-only, daily bars).

Adapted from "Kalman Trend Levels v9".

Kalman filter (recursive Bayesian estimator):
  prediction  = estimate[i-1]
  kalman_gain = error_est / (error_est + error_meas)
  estimate    = prediction + gain * (src[i] - prediction)
  error_est   = (1 - gain) * error_est + Q / length

  where error_meas = R * length  (fixed per config)

Signals (fire at bar close → execute at next open):
  Long  entry: short_kalman crosses above long_kalman
  Long  exit:  short_kalman crosses below long_kalman
  Cooldown:    3 bars minimum between entry and re-entry (anti-chop)
"""

from __future__ import annotations

import numpy as np
import pandas as pd


COOLDOWN_BARS = 3


def _kalman_filter(src: np.ndarray, length: int, R: float, Q: float) -> np.ndarray:
    n = len(src)
    result = np.full(n, np.nan)
    estimate = np.nan
    error_est = 1.0
    error_meas = R * length

    for i in range(1, n):
        if np.isnan(src[i]) or np.isnan(src[i - 1]):
            continue
        if np.isnan(estimate):
            estimate = src[i - 1]
        prediction = estimate
        kalman_gain = error_est / (error_est + error_meas)
        estimate = prediction + kalman_gain * (src[i] - prediction)
        error_est = (1 - kalman_gain) * error_est + Q / length
        result[i] = estimate

    return result


class _Portfolio:
    def __init__(self, initial_capital: float, commission: float, risk_pct: float):
        self._comm    = commission
        self._risk    = risk_pct
        self.cash     = initial_capital
        self.position = 0
        self.entry_price = 0.0
        self.notional    = 0.0

    def enter_long(self, price: float) -> None:
        fill             = price * (1.0 + self._comm)
        self.notional    = self.cash * self._risk
        self.cash       -= self.notional
        self.entry_price = fill
        self.position    = 1

    def exit_long(self, price: float) -> float:
        fill     = price * (1.0 - self._comm)
        pnl_pct  = (fill - self.entry_price) / self.entry_price
        proceeds = self.notional * (1.0 + pnl_pct)
        self.cash    += proceeds
        pnl           = proceeds - self.notional
        self.position = 0
        self.notional = 0.0
        return pnl

    def mtm(self, price: float) -> float:
        if self.position == 0:
            return self.cash
        return self.cash + self.notional * (1.0 + (price - self.entry_price) / self.entry_price)


def run_backtest(
    data: pd.DataFrame,
    params: dict,
    initial_capital: float = 1500.0,
    commission: float = 0.015,
    risk_pct: float = 0.33,
    return_equity_curve: bool = False,
    trade_start_idx: int = 0,
) -> dict:
    short_len = int(params["short_len"])
    long_len  = int(params["long_len"])
    kalman_r  = float(params["kalman_r"])
    kalman_q  = float(params["kalman_q"])

    close_a = data["Close"].values
    open_a  = data["Open"].values
    n       = len(close_a)

    short_k = _kalman_filter(close_a, short_len, kalman_r, kalman_q)
    long_k  = _kalman_filter(close_a, long_len,  kalman_r, kalman_q)

    port      = _Portfolio(initial_capital, commission, risk_pct)
    pending   = 0   # 1 = enter long,  -1 = exit long
    last_exit = -999

    live_start   = max(trade_start_idx, 1)
    live_len     = n - live_start
    equity_curve = np.empty(max(live_len, 1))
    equity_curve[0] = initial_capital
    trades: list[dict] = []

    for i in range(1, n):
        if np.isnan(short_k[i]) or np.isnan(long_k[i]) or \
           np.isnan(short_k[i - 1]) or np.isnan(long_k[i - 1]):
            if i >= live_start:
                equity_curve[i - live_start] = port.mtm(close_a[i])
            continue

        # ── 1. Execute pending signal at this bar's open ──────────────────
        if pending != 0 and i >= live_start:
            exec_price = open_a[i]
            if pending == 1 and port.position == 0:
                port.enter_long(exec_price)
                trades.append({"entry": port.entry_price,
                                "notional": port.notional, "bar_in": i})
            elif pending == -1 and port.position == 1:
                pnl = port.exit_long(exec_price)
                if trades and "pnl" not in trades[-1]:
                    trades[-1].update({"exit": exec_price, "pnl": pnl, "bar_out": i})
                last_exit = i
            pending = 0

        # ── 2. Evaluate signals from this bar's close ─────────────────────
        if i >= live_start:
            cross_above = short_k[i] > long_k[i] and short_k[i - 1] <= long_k[i - 1]
            cross_below = short_k[i] < long_k[i] and short_k[i - 1] >= long_k[i - 1]

            if port.position == 0 and pending == 0:
                if cross_above and (i - last_exit) >= COOLDOWN_BARS:
                    pending = 1
            elif port.position == 1:
                if cross_below:
                    pending = -1

            equity_curve[i - live_start] = port.mtm(close_a[i])

    # Force-close at last bar
    if port.position == 1:
        pnl = port.exit_long(close_a[-1])
        if trades and "pnl" not in trades[-1]:
            trades[-1].update({"exit": close_a[-1], "pnl": pnl,
                               "bar_out": n - 1, "reason": "EOD"})

    equity_curve[-1] = port.cash if port.position == 0 else port.mtm(close_a[-1])

    metrics = _calc_metrics(equity_curve, trades, initial_capital)
    if return_equity_curve:
        metrics["equity_curve"] = equity_curve
    return metrics


def _calc_metrics(
    equity: np.ndarray, trades: list[dict], initial_capital: float
) -> dict:
    total_return = (equity[-1] / initial_capital - 1.0) * 100.0

    rets   = np.diff(equity) / np.where(equity[:-1] != 0, equity[:-1], 1.0)
    std    = rets.std()
    sharpe = float(rets.mean() / std * np.sqrt(252)) if std > 0 else 0.0

    peak   = np.maximum.accumulate(equity)
    dd     = (equity - peak) / np.where(peak != 0, peak, 1.0)
    max_dd = float(dd.min() * 100.0)

    closed   = [t for t in trades if "pnl" in t]
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
