"""
strategy_keltner.py — Keltner Channel indicator and bar-by-bar backtest engine.

Entry logic (stop orders placed the bar the signal fires):
  Long:  close crosses above upper band  → stop-buy  at bar.high + TICK
  Short: close crosses below lower band  → stop-sell at bar.low  − TICK

Cancel / exit:
  Cancel pending long  when close drops below MA.
  Cancel pending short when close rises above MA.
  Exit long  position  when close < MA.
  Exit short position  when close > MA.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TICK_SIZE = 0.01


# ── Indicators ───────────────────────────────────────────────────────────────

def _true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    prev = close.shift(1)
    return pd.concat(
        [high - low, (high - prev).abs(), (low - prev).abs()], axis=1
    ).max(axis=1)


def calculate_keltner(
    data: pd.DataFrame,
    length: int,
    mult: float,
    atr_length: int,
    use_ema: bool,
    bands_style: str,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Return (ma, upper_band, lower_band)."""
    close, high, low = data["Close"], data["High"], data["Low"]

    ma = (
        close.ewm(span=length, adjust=False).mean()
        if use_ema
        else close.rolling(length).mean()
    )

    tr = _true_range(high, low, close)
    if bands_style == "ATR":
        band = tr.ewm(alpha=1.0 / atr_length, adjust=False).mean() * mult
    elif bands_style == "TR":
        band = tr * mult
    else:  # "Range"
        band = (high - low) * mult

    return ma, ma + band, ma - band


# ── Portfolio ─────────────────────────────────────────────────────────────────

class _Portfolio:
    def __init__(self, initial_capital: float, risk_pct: float, commission: float):
        self._comm = commission
        self._risk = risk_pct
        self.cash = initial_capital
        self.position = 0
        self.entry_price = 0.0
        self.notional = 0.0

    def enter_long(self, price: float) -> None:
        fill = price * (1 + self._comm)
        self.notional = self.cash * self._risk
        self.cash -= self.notional
        self.entry_price = fill
        self.position = 1

    def exit_long(self, price: float) -> float:
        fill = price * (1 - self._comm)
        pnl_pct = (fill - self.entry_price) / self.entry_price
        proceeds = self.notional * (1.0 + pnl_pct)
        self.cash += proceeds
        pnl = proceeds - self.notional
        self.position = 0
        self.notional = 0.0
        return pnl

    def enter_short(self, price: float) -> None:
        fill = price * (1 - self._comm)
        self.notional = self.cash * self._risk
        self.cash -= self.notional
        self.entry_price = fill
        self.position = -1

    def exit_short(self, price: float) -> float:
        fill = price * (1 + self._comm)
        pnl_pct = (self.entry_price - fill) / self.entry_price
        proceeds = self.notional * (1.0 + pnl_pct)
        self.cash += proceeds
        pnl = proceeds - self.notional
        self.position = 0
        self.notional = 0.0
        return pnl

    def mtm(self, price: float) -> float:
        if self.position == 0:
            return self.cash
        pnl_pct = (
            (price - self.entry_price) / self.entry_price
            if self.position == 1
            else (self.entry_price - price) / self.entry_price
        )
        return self.cash + self.notional * (1.0 + pnl_pct)


# ── Backtest engine ───────────────────────────────────────────────────────────

def run_backtest(
    data: pd.DataFrame,
    params: dict,
    initial_capital: float = 100_000.0,
    commission: float = 0.001,
    risk_pct: float = 0.95,
    return_equity_curve: bool = False,
) -> dict:
    ma, upper, lower = calculate_keltner(
        data,
        params["length"],
        params["mult"],
        params["atr_length"],
        params["use_ema"],
        params["bands_style"],
    )

    close  = data["Close"].values
    high   = data["High"].values
    low    = data["Low"].values
    open_  = data["Open"].values
    ma_a   = ma.values
    up_a   = upper.values
    lo_a   = lower.values
    n      = len(close)

    warmup = max(params["length"], params["atr_length"]) + 5
    port   = _Portfolio(initial_capital, risk_pct, commission)

    pending_dir: str | None = None
    pending_stop: float = 0.0

    equity_curve = np.empty(n)
    equity_curve[0] = initial_capital
    trades: list[dict] = []

    for i in range(1, n):
        if i < warmup or np.isnan(ma_a[i]) or np.isnan(up_a[i]):
            equity_curve[i] = port.mtm(close[i])
            continue

        if port.position == 0 and pending_dir is not None:
            if pending_dir == "long":
                if close[i] < ma_a[i]:
                    pending_dir = None
                elif high[i] >= pending_stop:
                    fill = max(open_[i], pending_stop)
                    port.enter_long(fill)
                    trades.append({"dir": 1, "entry": port.entry_price, "bar_in": i})
                    pending_dir = None
            else:
                if close[i] > ma_a[i]:
                    pending_dir = None
                elif low[i] <= pending_stop:
                    fill = min(open_[i], pending_stop)
                    port.enter_short(fill)
                    trades.append({"dir": -1, "entry": port.entry_price, "bar_in": i})
                    pending_dir = None

        if port.position == 1 and close[i] < ma_a[i]:
            pnl = port.exit_long(close[i])
            if trades and "pnl" not in trades[-1]:
                trades[-1].update({"exit": close[i], "pnl": pnl, "bar_out": i})

        elif port.position == -1 and close[i] > ma_a[i]:
            pnl = port.exit_short(close[i])
            if trades and "pnl" not in trades[-1]:
                trades[-1].update({"exit": close[i], "pnl": pnl, "bar_out": i})

        if port.position == 0 and pending_dir is None:
            cross_up = close[i - 1] <= up_a[i - 1] and close[i] > up_a[i]
            cross_dn = close[i - 1] >= lo_a[i - 1] and close[i] < lo_a[i]
            if cross_up:
                pending_dir = "long"
                pending_stop = high[i] + TICK_SIZE
            elif cross_dn:
                pending_dir = "short"
                pending_stop = low[i] - TICK_SIZE

        equity_curve[i] = port.mtm(close[i])

    if port.position == 1:
        pnl = port.exit_long(close[-1])
        if trades and "pnl" not in trades[-1]:
            trades[-1].update({"exit": close[-1], "pnl": pnl, "bar_out": n - 1})
    elif port.position == -1:
        pnl = port.exit_short(close[-1])
        if trades and "pnl" not in trades[-1]:
            trades[-1].update({"exit": close[-1], "pnl": pnl, "bar_out": n - 1})

    equity_curve[-1] = port.cash if port.position == 0 else port.mtm(close[-1])

    metrics = _calculate_metrics(equity_curve, trades, initial_capital)
    if return_equity_curve:
        metrics["equity_curve"] = equity_curve
    return metrics


def _calculate_metrics(
    equity_curve: np.ndarray, trades: list[dict], initial_capital: float
) -> dict:
    total_return = (equity_curve[-1] / initial_capital - 1.0) * 100.0

    rets = np.diff(equity_curve) / np.where(equity_curve[:-1] != 0, equity_curve[:-1], 1.0)
    std = rets.std()
    sharpe = float(rets.mean() / std * np.sqrt(252)) if std > 0 else 0.0

    peak = np.maximum.accumulate(equity_curve)
    dd = (equity_curve - peak) / np.where(peak != 0, peak, 1.0)
    max_dd = float(dd.min() * 100.0)

    closed = [t for t in trades if "pnl" in t]
    n_trades = len(closed)
    if n_trades:
        wins   = [t["pnl"] for t in closed if t["pnl"] > 0]
        losses = [t["pnl"] for t in closed if t["pnl"] <= 0]
        win_rate     = len(wins) / n_trades * 100.0
        gross_profit = sum(wins)
        gross_loss   = abs(sum(losses))
        profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else float("inf")
    else:
        win_rate = profit_factor = 0.0

    return {
        "total_return":   round(total_return,   4),
        "sharpe_ratio":   round(sharpe,          4),
        "max_drawdown":   round(max_dd,          4),
        "win_rate":       round(win_rate,         4),
        "num_trades":     n_trades,
        "profit_factor":  round(profit_factor,   4),
        "final_equity":   round(float(equity_curve[-1]), 2),
    }
