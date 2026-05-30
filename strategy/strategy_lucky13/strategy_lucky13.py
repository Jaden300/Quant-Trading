"""
strategy_lucky13.py — Lucky 13 EMA backtest engine.

HOW IT WORKS:
  Enters when price crosses above EMA(13) on a green bar with above-average
  volume — requiring trend flip, bullish momentum, and institutional interest
  all on the same bar.

Entry:
  close > open                          — green bar (bullish momentum)
  close > ema(ema_length)               — price above EMA
  close[1] <= ema(ema_length)[1]        — was below EMA last bar (crossover)
  volume > vol_mult × SMA(volume, 20)   — volume spike confirms move
  → go long at next bar's open

Exit (whichever comes first, checked intraday):
  high >= entry * (1 + tp_pct)          — take profit
  low  <= entry * (1 - sl_pct)          — stop loss
  If both hit same bar → stop wins (conservative).
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
    ema_length = int(params["ema_length"])
    vol_mult   = float(params["vol_mult"])
    tp_pct     = float(params["tp_pct"])
    sl_pct     = float(params["sl_pct"])

    close_a  = data["Close"].values
    open_a   = data["Open"].values
    high_a   = data["High"].values
    low_a    = data["Low"].values
    volume_a = data["Volume"].values
    n        = len(close_a)

    ema_a    = data["Close"].ewm(span=ema_length, adjust=False).mean().values
    vol_sma  = data["Volume"].rolling(20).mean().values

    live_start = max(trade_start_idx, ema_length + 25)
    port       = _Portfolio(initial_capital, commission, risk_pct)
    tp_level   = 0.0
    sl_level   = 0.0
    pending    = 0

    equity_curve = np.full(n, float(initial_capital))
    trades: list[dict] = []

    for i in range(1, n):
        if np.isnan(ema_a[i]) or np.isnan(vol_sma[i]):
            equity_curve[i] = port.mtm(close_a[i])
            continue

        # ── 1. Execute pending entry at open ──────────────────────────────
        if pending == 1 and port.position == 0 and i >= live_start:
            port.enter_long(open_a[i])
            tp_level = port.entry_price * (1.0 + tp_pct)
            sl_level = port.entry_price * (1.0 - sl_pct)
            trades.append({"entry": port.entry_price,
                           "notional": port.notional, "bar_in": i})
            pending = 0

        # ── 2. Check intraday TP / SL ─────────────────────────────────────
        if port.position == 1 and i >= live_start:
            hit_tp = high_a[i] >= tp_level
            hit_sl = low_a[i]  <= sl_level

            if hit_sl or hit_tp:
                if hit_sl:
                    fill   = open_a[i] if open_a[i] <= sl_level else sl_level
                    reason = "stop"
                else:
                    fill   = tp_level
                    reason = "tp"
                pnl = port.exit_long(fill)
                if trades and "pnl" not in trades[-1]:
                    trades[-1].update({"exit": fill, "pnl": pnl,
                                       "bar_out": i, "reason": reason})
                tp_level = sl_level = 0.0

        # ── 3. Generate signal from bar close ─────────────────────────────
        if port.position == 0 and pending == 0 and i >= live_start:
            is_green     = close_a[i] > open_a[i]
            above_ema    = close_a[i] > ema_a[i]
            was_below    = close_a[i - 1] <= ema_a[i - 1]
            vol_ok       = volume_a[i] > vol_mult * vol_sma[i]

            if is_green and above_ema and was_below and vol_ok:
                pending = 1

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
