"""
strategy_psar.py — Parabolic SAR backtest engine.

HOW IT WORKS:
  Parabolic SAR places a trailing stop that accelerates as the trend matures.
  The acceleration factor (AF) starts small and increases each time a new
  extreme price is set, pulling the SAR closer to price over time.

  When SAR flips from bearish to bullish → go long.
  When SAR flips back to bearish → exit long.

Entry:
  SAR flips from downtrend to uptrend   — SAR just crossed below price
  → go long at next bar's open

Exit (whichever comes first):
  SAR flips back to downtrend           — signal → exit at next bar's open
  high >= entry * (1 + tp_pct)          — optional TP hit intraday (0 = disabled)

SAR algorithm faithful to TradingView's built-in implementation:
  - AF starts at `start`, increments each time a new EP is set, capped at `maximum`
  - Uptrend SAR is capped by the two prior lows (prevents premature stops)
  - Downtrend SAR is capped by the two prior highs
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _calc_psar(
    high_a: np.ndarray,
    low_a: np.ndarray,
    close_a: np.ndarray,
    start: float,
    increment: float,
    maximum: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Returns (sar, uptrend) arrays, faithful to TradingView's implementation."""
    n = len(close_a)
    sar      = np.full(n, np.nan)
    uptrend  = np.zeros(n, dtype=bool)
    ep       = np.zeros(n)
    af       = np.full(n, start)
    next_sar = np.full(n, np.nan)

    if n < 3:
        return sar, uptrend

    # Bar 1 initialisation (matches Pine Script bar_index == 1 block)
    if close_a[1] > close_a[0]:
        uptrend[1] = True
        ep[1]      = high_a[1]
        prev_sar   = low_a[0]
        prev_ep    = high_a[1]
    else:
        uptrend[1] = False
        ep[1]      = low_a[1]
        prev_sar   = high_a[0]
        prev_ep    = low_a[1]

    sar[1]      = prev_sar + start * (prev_ep - prev_sar)
    af[1]       = start
    next_sar[1] = sar[1] + af[1] * (ep[1] - sar[1])

    for i in range(2, n):
        cur_sar  = next_sar[i - 1]
        cur_up   = uptrend[i - 1]
        cur_ep   = ep[i - 1]
        cur_af   = af[i - 1]
        first_tb = False

        # Check for trend flip
        if cur_up:
            if cur_sar > low_a[i]:
                first_tb = True
                cur_up   = False
                cur_sar  = max(cur_ep, high_a[i])
                cur_ep   = low_a[i]
                cur_af   = start
        else:
            if cur_sar < high_a[i]:
                first_tb = True
                cur_up   = True
                cur_sar  = min(cur_ep, low_a[i])
                cur_ep   = high_a[i]
                cur_af   = start

        # Update EP / AF when no flip
        if not first_tb:
            if cur_up:
                if high_a[i] > cur_ep:
                    cur_ep = high_a[i]
                    cur_af = min(cur_af + increment, maximum)
            else:
                if low_a[i] < cur_ep:
                    cur_ep = low_a[i]
                    cur_af = min(cur_af + increment, maximum)

        # Cap SAR by prior two bars (TradingView convention)
        if cur_up:
            cur_sar = min(cur_sar, low_a[i - 1])
            if i > 2:
                cur_sar = min(cur_sar, low_a[i - 2])
        else:
            cur_sar = max(cur_sar, high_a[i - 1])
            if i > 2:
                cur_sar = max(cur_sar, high_a[i - 2])

        sar[i]      = cur_sar
        uptrend[i]  = cur_up
        ep[i]       = cur_ep
        af[i]       = cur_af
        next_sar[i] = cur_sar + cur_af * (cur_ep - cur_sar)

    return sar, uptrend


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
    start     = float(params["start"])
    increment = float(params["increment"])
    maximum   = float(params["maximum"])
    tp_pct    = float(params["tp_pct"])

    close_a = data["Close"].values
    high_a  = data["High"].values
    low_a   = data["Low"].values
    open_a  = data["Open"].values
    n       = len(close_a)

    sar_a, uptrend_a = _calc_psar(high_a, low_a, close_a, start, increment, maximum)

    live_start = max(trade_start_idx, 10)
    port       = _Portfolio(initial_capital, commission, risk_pct)
    tp_level   = 0.0
    pending    = 0   # 1 = enter long, -1 = exit long

    equity_curve = np.full(n, float(initial_capital))
    trades: list[dict] = []

    for i in range(1, n):
        if np.isnan(sar_a[i]):
            equity_curve[i] = port.mtm(close_a[i])
            continue

        # ── 1. Execute pending signal at open ─────────────────────────────
        if i >= live_start:
            if pending == 1 and port.position == 0:
                port.enter_long(open_a[i])
                tp_level = port.entry_price * (1.0 + tp_pct) if tp_pct > 0 else 0.0
                trades.append({"entry": port.entry_price,
                               "notional": port.notional, "bar_in": i})
            elif pending == -1 and port.position == 1:
                pnl = port.exit_long(open_a[i])
                if trades and "pnl" not in trades[-1]:
                    trades[-1].update({"exit": open_a[i], "pnl": pnl,
                                       "bar_out": i, "reason": "sar_flip"})
                tp_level = 0.0
            pending = 0

        # ── 2. Check intraday TP ──────────────────────────────────────────
        if port.position == 1 and tp_pct > 0 and high_a[i] >= tp_level and i >= live_start:
            pnl = port.exit_long(tp_level)
            if trades and "pnl" not in trades[-1]:
                trades[-1].update({"exit": tp_level, "pnl": pnl,
                                   "bar_out": i, "reason": "tp"})
            tp_level = 0.0

        # ── 3. Generate signal from SAR state at bar close ────────────────
        if i >= live_start:
            just_turned_up   = uptrend_a[i] and not uptrend_a[i - 1]
            just_turned_down = not uptrend_a[i] and uptrend_a[i - 1]

            if port.position == 0 and just_turned_up:
                pending = 1
            elif port.position == 1 and just_turned_down:
                pending = -1

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
