"""
strategy_ema_trail.py — EMA Crossover + Trailing Stop backtest engine.

HOW IT WORKS:
  Enters when the fast EMA crosses above the slow EMA (uptrend confirmed).
  Sets an initial TP and SL. As price moves in your favour, both the TP
  and SL ratchet upward — locking in profit while letting winners run.

Entry:
  fast_ema crosses above slow_ema   — trend just turned bullish
  → go long at that bar's close (process_orders_on_close behaviour)

Exit (whichever comes first, checked at each bar's close):
  close >= tp_level                 — take profit hit
  close <= sl_level                 — stop loss hit

Trailing (fires once per bar, after price moves trail_trigger_pct in favour):
  tp_level  *= (1 + trail_step_pct)   — target moves up
  sl_level  *= (1 + trail_step_pct)   — floor moves up
  ref_price *= (1 + trail_trigger_pct) — step reference up for next check

Adapted from community Pine Script "Trailing.SL.Target" by Sharad_Gaikwad.
Long-only. Exits and entries both at bar close (faithful to original).
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


def _ema(close: pd.Series, length: int) -> np.ndarray:
    return close.ewm(span=length, adjust=False).mean().values


def run_backtest(
    data: pd.DataFrame,
    params: dict,
    initial_capital: float = 1500.0,
    commission: float = 0.015,
    risk_pct: float = 0.33,
    return_equity_curve: bool = False,
    return_trades: bool = False,
    trade_start_idx: int = 0,
) -> dict:
    fast_len          = int(params["fast_len"])
    slow_len          = int(params["slow_len"])
    tp_pct            = float(params["tp_pct"])
    sl_pct            = float(params["sl_pct"])
    trail_trigger_pct = float(params["trail_trigger_pct"])
    trail_step_pct    = float(params["trail_step_pct"])

    close_a  = data["Close"].values
    fast_ema = _ema(data["Close"], fast_len)
    slow_ema = _ema(data["Close"], slow_len)
    n        = len(close_a)

    live_start = max(trade_start_idx, slow_len + 5)
    port       = _Portfolio(initial_capital, commission, risk_pct)
    tp_level   = 0.0
    sl_level   = 0.0
    ref_price  = 0.0

    equity_curve = np.full(n, float(initial_capital))
    trades: list[dict] = []

    for i in range(1, n):
        if np.isnan(fast_ema[i]) or np.isnan(slow_ema[i]):
            equity_curve[i] = port.mtm(close_a[i])
            continue

        # ── 1. Update trailing levels ─────────────────────────────────────
        if port.position == 1 and i >= live_start and ref_price > 0:
            if close_a[i] >= ref_price * (1.0 + trail_trigger_pct):
                tp_level  *= (1.0 + trail_step_pct)
                sl_level  *= (1.0 + trail_step_pct)
                ref_price *= (1.0 + trail_trigger_pct)

        # ── 2. Check exit at bar close ────────────────────────────────────
        if port.position == 1 and i >= live_start:
            if close_a[i] <= sl_level or close_a[i] >= tp_level:
                reason = "stop" if close_a[i] <= sl_level else "tp"
                pnl    = port.exit_long(close_a[i])
                if trades and "pnl" not in trades[-1]:
                    trades[-1].update({"exit": close_a[i], "pnl": pnl,
                                       "bar_out": i, "reason": reason})
                tp_level = sl_level = ref_price = 0.0

        # ── 3. Check entry signal at bar close ────────────────────────────
        if port.position == 0 and i >= live_start:
            crossover = fast_ema[i - 1] < slow_ema[i - 1] and fast_ema[i] >= slow_ema[i]
            if crossover:
                port.enter_long(close_a[i])
                tp_level  = port.entry_price * (1.0 + tp_pct)
                sl_level  = port.entry_price * (1.0 - sl_pct)
                ref_price = port.entry_price
                trades.append({"entry": port.entry_price,
                               "notional": port.notional, "bar_in": i})

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
        metrics["eq_dates"] = data.index[live_start:].tolist()
    if return_trades:
        metrics["trades"]    = trades
        metrics["all_dates"] = data.index.tolist()
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
