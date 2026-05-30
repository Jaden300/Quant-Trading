"""
strategy_rsi_reversion.py — RSI Mean Reversion backtest engine.

HOW IT WORKS:
  Only trades stocks in an uptrend (close > trend MA).
  Waits for RSI to pull back to oversold territory, then enter when RSI
  recovers back above the entry threshold — buying the dip as it bounces.

Entry:
  close > MA(trend_ma)               — uptrend filter
  RSI was below rsi_entry last bar   — stock was oversold
  RSI >= rsi_entry this bar          — RSI is recovering (confirmed bounce)
  → go long at next bar's open

Exit (whichever comes first):
  RSI >= rsi_exit                    — overbought, take profit
  low <= entry_price * (1 - sl_pct)  — stop loss hit intraday

Execution: signal fires at bar close → entry/exit at NEXT bar's open.
Stop loss uses intraday low to check if stop was hit, fills at stop price
(or open if the bar gapped down through it).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _rsi(close: pd.Series, length: int) -> pd.Series:
    delta    = close.diff()
    gain     = delta.clip(lower=0)
    loss     = (-delta).clip(lower=0)
    avg_gain = gain.ewm(alpha=1.0 / length, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / length, adjust=False).mean()
    rs       = avg_gain / avg_loss.replace(0, np.inf)
    return 100.0 - (100.0 / (1.0 + rs))


def _sma(close: pd.Series, length: int) -> pd.Series:
    return close.rolling(length).mean()


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
    trend_ma   = int(params["trend_ma"])
    rsi_length = int(params["rsi_length"])
    rsi_entry  = float(params["rsi_entry"])
    rsi_exit   = float(params["rsi_exit"])
    sl_pct     = float(params["sl_pct"])

    close = data["Close"]
    ma    = _sma(close, trend_ma)
    rsi   = _rsi(close, rsi_length)

    close_a = close.values
    open_a  = data["Open"].values
    low_a   = data["Low"].values
    ma_a    = ma.values
    rsi_a   = rsi.values
    n       = len(close_a)

    live_start = max(trade_start_idx, trend_ma + rsi_length + 5)
    port       = _Portfolio(initial_capital, commission, risk_pct)
    stop_price = 0.0
    pending    = 0   # 1 = enter long, -1 = exit long

    equity_curve = np.full(n, float(initial_capital))
    trades: list[dict] = []

    for i in range(1, n):
        if np.isnan(ma_a[i]) or np.isnan(rsi_a[i]):
            equity_curve[i] = port.mtm(close_a[i])
            continue

        # ── 1. Execute pending signal at this bar's open ──────────────────
        if pending != 0 and i >= live_start:
            if pending == 1 and port.position == 0:
                port.enter_long(open_a[i])
                stop_price = port.entry_price * (1.0 - sl_pct)
                trades.append({"entry": port.entry_price,
                                "notional": port.notional, "bar_in": i})
            elif pending == -1 and port.position == 1:
                pnl = port.exit_long(open_a[i])
                if trades and "pnl" not in trades[-1]:
                    trades[-1].update({"exit": open_a[i], "pnl": pnl,
                                       "bar_out": i, "reason": "signal"})
            pending = 0

        # ── 2. Check intraday stop loss ───────────────────────────────────
        if port.position == 1 and low_a[i] <= stop_price and i >= live_start:
            fill = open_a[i] if open_a[i] <= stop_price else stop_price
            pnl  = port.exit_long(fill)
            if trades and "pnl" not in trades[-1]:
                trades[-1].update({"exit": fill, "pnl": pnl,
                                   "bar_out": i, "reason": "stop"})
            stop_price = 0.0

        # ── 3. Generate signal from this bar's close ──────────────────────
        if i >= live_start:
            in_uptrend   = close_a[i] > ma_a[i]
            rsi_was_low  = rsi_a[i - 1] < rsi_entry
            rsi_recovery = rsi_a[i] >= rsi_entry

            if port.position == 0 and in_uptrend and rsi_was_low and rsi_recovery:
                pending = 1
            elif port.position == 1 and rsi_a[i] >= rsi_exit:
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
