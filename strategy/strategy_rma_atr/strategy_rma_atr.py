"""
strategy_rma_atr.py — RMA ATR Bands backtest engine.

Indicator:
  ma    = RMA(ma_src, ma_length)         — Wilder's smoothing of selected price
  atr   = RMA(true_range, atr_length)
  upper = ma + atr * upper_mult
  lower = ma - atr * lower_mult

Signal (fires only on trend CHANGE):
  trend = 1  when close > upper  → long signal
  trend = -1 when close < lower  → short signal
  Trend persists until the opposite band is crossed.

Execution:
  Signal fires at bar close → entry/flip executed at NEXT bar's open.
  No TP/SL — position held until opposite signal flips it.
  Long ↔ Short flips in one step (exit old + enter new at same open price).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ── Indicators ────────────────────────────────────────────────────────────────

def _true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    prev = close.shift(1)
    return pd.concat(
        [high - low, (high - prev).abs(), (low - prev).abs()], axis=1
    ).max(axis=1)


def _select_src(data: pd.DataFrame, ma_src: str) -> pd.Series:
    if ma_src == "high":
        return data["High"]
    if ma_src == "hl2":
        return (data["High"] + data["Low"]) / 2.0
    return data["Close"]   # "close" or fallback


def calculate_bands(
    data: pd.DataFrame,
    ma_length: int,
    atr_length: int,
    upper_mult: float,
    lower_mult: float,
    ma_src: str = "high",
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Return (ma, upper_band, lower_band)."""
    src = _select_src(data, ma_src)
    ma  = src.ewm(alpha=1.0 / ma_length, adjust=False).mean()

    tr  = _true_range(data["High"], data["Low"], data["Close"])
    atr = tr.ewm(alpha=1.0 / atr_length, adjust=False).mean()

    return ma, ma + atr * upper_mult, ma - atr * lower_mult


# ── Portfolio ─────────────────────────────────────────────────────────────────

class _Portfolio:
    """Tracks cash + one open position (long or short)."""

    def __init__(self, initial_capital: float, commission: float, risk_pct: float = 0.33):
        self._comm    = commission
        self._risk    = risk_pct   # fraction of current cash deployed per trade
        self.cash     = initial_capital
        self.position = 0
        self.entry_price = 0.0
        self.notional    = 0.0

    def enter_long(self, price: float) -> None:
        fill = price * (1.0 + self._comm)
        self.notional    = self.cash * self._risk
        self.cash       -= self.notional        # idle cash stays in account
        self.entry_price = fill
        self.position    = 1

    def exit_long(self, price: float) -> float:
        fill    = price * (1.0 - self._comm)
        pnl_pct = (fill - self.entry_price) / self.entry_price
        proceeds = self.notional * (1.0 + pnl_pct)
        self.cash     += proceeds
        pnl            = proceeds - self.notional
        self.position  = 0
        self.notional  = 0.0
        return pnl

    def enter_short(self, price: float) -> None:
        fill = price * (1.0 - self._comm)
        self.notional    = self.cash * self._risk
        self.cash       -= self.notional
        self.entry_price = fill
        self.position    = -1

    def exit_short(self, price: float) -> float:
        fill    = price * (1.0 + self._comm)
        pnl_pct = (self.entry_price - fill) / self.entry_price
        proceeds = self.notional * (1.0 + pnl_pct)
        self.cash    += proceeds
        pnl           = proceeds - self.notional
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
    initial_capital: float = 1500.0,
    commission: float = 0.015,
    risk_pct: float = 0.33,
    return_equity_curve: bool = False,
    return_trades: bool = False,
    trade_start_idx: int = 0,
    long_only: bool = True,
    ma_exit: bool = False,
) -> dict:
    """
    trade_start_idx: first bar index where trades are allowed to execute.
    long_only: skip short entries; on a short signal, just exit any open long.
    ma_exit: exit a long position when close drops below the MA (faster exit than
             waiting for the lower band to be breached; produces shorter holds).
    """
    ma_src     = str(params["ma_src"])
    ma_length  = int(params["ma_length"])
    atr_length = int(params["atr_length"])
    upper_mult = float(params["upper_mult"])
    lower_mult = float(params["lower_mult"])
    ma_exit    = bool(params.get("ma_exit", ma_exit))

    ma, upper, lower = calculate_bands(
        data, ma_length, atr_length, upper_mult, lower_mult, ma_src
    )

    close  = data["Close"].values
    open_  = data["Open"].values
    ma_a   = ma.values
    up_a   = upper.values
    lo_a   = lower.values
    n      = len(close)

    port      = _Portfolio(initial_capital, commission, risk_pct)
    trend_cur = 0
    pending   = 0   # queued signal: 1=long, -1=exit/short

    live_start = max(trade_start_idx, 1)
    live_len   = n - live_start
    equity_curve = np.empty(max(live_len, 1))
    equity_curve[0] = initial_capital
    trades: list[dict] = []

    for i in range(1, n):
        if np.isnan(up_a[i]) or np.isnan(lo_a[i]):
            if i >= live_start:
                equity_curve[i - live_start] = port.mtm(close[i])
            continue

        # ── 1. Execute pending signal at this bar's open ──────────────────
        if pending != 0 and i >= live_start:
            exec_price = open_[i]

            if pending == 1:   # go long
                if port.position == -1:
                    pnl = port.exit_short(exec_price)
                    if trades and "pnl" not in trades[-1]:
                        trades[-1].update({"exit": exec_price, "pnl": pnl,
                                           "bar_out": i, "reason": "flip"})
                if port.position == 0:
                    port.enter_long(exec_price)
                    trades.append({"dir": 1, "entry": port.entry_price,
                                   "notional": port.notional, "bar_in": i})

            else:              # exit / go short
                if port.position == 1:
                    pnl = port.exit_long(exec_price)
                    if trades and "pnl" not in trades[-1]:
                        trades[-1].update({"exit": exec_price, "pnl": pnl,
                                           "bar_out": i, "reason": "flip"})
                if port.position == 0 and not long_only:
                    port.enter_short(exec_price)
                    trades.append({"dir": -1, "entry": port.entry_price,
                                   "notional": port.notional, "bar_in": i})

            pending = 0

        # ── 2. Update trend state from this bar's close ───────────────────
        trend_prev = trend_cur
        if close[i] > up_a[i]:
            trend_cur = 1
        elif close[i] < lo_a[i]:
            trend_cur = -1

        # Only queue signals once we're in the live window
        if i >= live_start:
            if trend_cur == 1 and trend_prev != 1:
                pending = 1
            elif trend_cur == -1 and trend_prev != -1:
                pending = -1

            # MA exit: if long and price drops back to the MA, queue exit
            # (overrides any other pending; fires before lower-band breach)
            if ma_exit and port.position == 1 and close[i] < ma_a[i]:
                pending = -1

            equity_curve[i - live_start] = port.mtm(close[i])

    # Force-close at last bar
    if port.position == 1:
        pnl = port.exit_long(close[-1])
        if trades and "pnl" not in trades[-1]:
            trades[-1].update({"exit": close[-1], "pnl": pnl,
                               "bar_out": n - 1, "reason": "EOD"})
    elif port.position == -1:
        pnl = port.exit_short(close[-1])
        if trades and "pnl" not in trades[-1]:
            trades[-1].update({"exit": close[-1], "pnl": pnl,
                               "bar_out": n - 1, "reason": "EOD"})

    equity_curve[-1] = port.cash if port.position == 0 else port.mtm(close[-1])

    metrics = _calc_metrics(equity_curve, trades, initial_capital)
    if return_equity_curve:
        metrics["equity_curve"] = equity_curve
        metrics["eq_dates"] = data.index[live_start:][:len(equity_curve)].tolist()
    if return_trades:
        metrics["trades"]    = trades
        metrics["all_dates"] = data.index.tolist()
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
        mean_hold     = float(np.mean(hold_bars)) if hold_bars else 0.0
    else:
        win_rate = profit_factor = mean_hold = 0.0

    return {
        "total_return":  round(total_return,  4),
        "sharpe_ratio":  round(sharpe,         4),
        "max_drawdown":  round(max_dd,          4),
        "win_rate":      round(win_rate,         4),
        "num_trades":    n_trades,
        "profit_factor": round(profit_factor,   4),
        "mean_hold_bars": round(mean_hold,       1),
        "final_equity":  round(float(equity[-1]), 2),
    }
