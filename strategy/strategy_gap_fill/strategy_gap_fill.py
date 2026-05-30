"""
strategy_gap_fill.py — Gap Fill backtest engine.

HOW IT WORKS:
  When a stock gaps DOWN at the open (opens below the previous day's low
  with no candle overlap), it buys expecting the gap to fill back up.
  Target is fixed at the gap fill level (previous low). Pure mean-reversion.

Gap condition (checked at bar open):
  open < low[1]                              — gapped below previous low
  gap_pct = (low[1] - open) / open >= min_gap_pct  — gap is large enough

Entry:
  → go long at that bar's open

Exit (whichever comes first, intraday):
  high >= gap_fill_level (prev low)          — gap filled, take profit
  low  <= entry * (1 - sl_pct)              — stop loss
  bars held >= max_hold                      — time-based exit at close

Adapted from "Gap Filling Strategy" by alexgrover.
Long-only (down-gap fills only). Entry and TP/SL all on daily bars.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


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
        pnl_pct = (price - self.entry_price) / self.entry_price
        return self.cash + self.notional * (1.0 + pnl_pct)


def run_backtest(
    data: pd.DataFrame,
    params: dict,
    initial_capital: float = 1500.0,
    commission: float = 0.015,
    risk_pct: float = 0.33,
    return_equity_curve: bool = False,
    trade_start_idx: int = 0,
) -> dict:
    min_gap_pct = float(params["min_gap_pct"])
    sl_pct      = float(params["sl_pct"])
    max_hold    = int(params["max_hold"])

    close_a = data["Close"].values
    open_a  = data["Open"].values
    high_a  = data["High"].values
    low_a   = data["Low"].values
    n       = len(close_a)

    live_start = max(trade_start_idx, 2)
    port       = _Portfolio(initial_capital, commission, risk_pct)
    sl_level   = 0.0
    tp_level   = 0.0
    bars_held  = 0

    equity_curve = np.full(n, float(initial_capital))
    trades: list[dict] = []

    for i in range(1, n):
        # ── 1. Manage open position ───────────────────────────────────────
        if port.position == 1 and i >= live_start:
            bars_held += 1
            hit_tp = high_a[i] >= tp_level
            hit_sl = low_a[i]  <= sl_level
            timed_out = bars_held >= max_hold

            if hit_sl or hit_tp or timed_out:
                if hit_sl and not hit_tp:
                    fill   = open_a[i] if open_a[i] <= sl_level else sl_level
                    reason = "stop"
                elif hit_tp:
                    fill   = tp_level
                    reason = "fill"
                else:
                    fill   = close_a[i]
                    reason = "timeout"
                pnl = port.exit_long(fill)
                if trades and "pnl" not in trades[-1]:
                    trades[-1].update({"exit": fill, "pnl": pnl,
                                       "bar_out": i, "reason": reason})
                sl_level = tp_level = 0.0
                bars_held = 0

        # ── 2. Detect down gap and enter ──────────────────────────────────
        if port.position == 0 and i >= live_start:
            gap_down = open_a[i] < low_a[i - 1]
            if gap_down:
                gap_pct = (low_a[i - 1] - open_a[i]) / open_a[i]
                if gap_pct >= min_gap_pct:
                    port.enter_long(open_a[i])
                    tp_level  = low_a[i - 1]             # gap fill = prev low
                    sl_level  = port.entry_price * (1.0 - sl_pct)
                    bars_held = 0
                    trades.append({"entry": port.entry_price,
                                   "notional": port.notional, "bar_in": i})

                    # Check if gap filled or stopped out on same bar
                    hit_tp = high_a[i] >= tp_level
                    hit_sl = low_a[i]  <= sl_level
                    if hit_sl or hit_tp:
                        if hit_sl and not hit_tp:
                            fill   = open_a[i] if open_a[i] <= sl_level else sl_level
                            reason = "stop"
                        else:
                            fill   = tp_level
                            reason = "fill"
                        pnl = port.exit_long(fill)
                        trades[-1].update({"exit": fill, "pnl": pnl,
                                           "bar_out": i, "reason": reason})
                        sl_level = tp_level = 0.0
                        bars_held = 0

        equity_curve[i] = port.mtm(close_a[i])

    # Force-close at last bar
    if port.position == 1:
        pnl = port.exit_long(close_a[-1])
        if trades and "pnl" not in trades[-1]:
            trades[-1].update({"exit": close_a[-1], "pnl": pnl,
                               "bar_out": n - 1, "reason": "EOD"})

    equity_curve[-1] = port.cash if port.position == 0 else port.mtm(close_a[-1])
    live_eq = equity_curve[live_start:]

    metrics = _calc_metrics(live_eq, trades, initial_capital)
    if return_equity_curve:
        metrics["equity_curve"] = live_eq
    return metrics


def _calc_metrics(equity: np.ndarray, trades: list[dict], initial_capital: float) -> dict:
    if len(equity) < 2:
        return {"total_return": 0.0, "sharpe_ratio": 0.0, "max_drawdown": 0.0,
                "win_rate": 0.0, "num_trades": 0, "profit_factor": 0.0,
                "mean_hold_bars": 0.0, "final_equity": float(equity[-1])}

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
        hold_bars     = [t["bar_out"] - t["bar_in"] for t in closed
                         if "bar_out" in t and "bar_in" in t]
        mean_hold     = float(np.mean(hold_bars)) if hold_bars else 0.0
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
