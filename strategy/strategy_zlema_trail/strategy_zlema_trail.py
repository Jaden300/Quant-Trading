"""
strategy_zlema_trail.py — ZLEMA Trailing Stop backtest engine (long-only).

Indicators:
  lag      = floor((length - 1) / 2)
  zlema    = EMA(close + (close - close[lag]), length)   ← zero-lag EMA
  atr_high = rolling max of ATR(length) over length*3 bars
  volatility = atr_high * mult
  upper    = zlema + volatility
  lower    = zlema - volatility
  ema      = EMA(close, ema_length)

Trend state (persists):
  trend → 1  when close > upper AND close > ema
  trend → -1 when close < lower AND close < ema

Long entry signal (fires at bar close → executes at next open):
  close crosses above zlema  AND  trend == 1  AND  close > ema

Exit — trailing stop:
  longStop = max(longStop, lower)   (ratchets upward only, never falls)
  Exit when low[i] <= longStop  → fill at min(open[i], longStop)
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ── Indicators ────────────────────────────────────────────────────────────────

def _zlema(close: pd.Series, length: int) -> pd.Series:
    lag = (length - 1) // 2
    src = close + (close - close.shift(lag))
    return src.ewm(span=length, adjust=False).mean()


def _ema(close: pd.Series, length: int) -> pd.Series:
    return close.ewm(span=length, adjust=False).mean()


def _true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    prev = close.shift(1)
    return pd.concat(
        [high - low, (high - prev).abs(), (low - prev).abs()], axis=1
    ).max(axis=1)


def calculate_bands(
    data: pd.DataFrame,
    length: int,
    mult: float,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Return (zlema, upper, lower)."""
    zl  = _zlema(data["Close"], length)
    tr  = _true_range(data["High"], data["Low"], data["Close"])
    atr = tr.rolling(length).mean()
    vol = atr.rolling(length * 3).max() * mult
    return zl, zl + vol, zl - vol


# ── Portfolio ─────────────────────────────────────────────────────────────────

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


# ── Backtest engine ───────────────────────────────────────────────────────────

def run_backtest(
    data: pd.DataFrame,
    params: dict,
    initial_capital: float = 1500.0,
    commission: float = 0.015,
    risk_pct: float = 0.33,
    return_equity_curve: bool = False,
    trade_start_idx: int = 0,
) -> dict:
    length     = int(params["length"])
    mult       = float(params["mult"])
    ema_length = int(params["ema_length"])

    zlema, upper, lower = calculate_bands(data, length, mult)
    ema = _ema(data["Close"], ema_length)

    close  = data["Close"].values
    open_  = data["Open"].values
    low_   = data["Low"].values
    zl_a   = zlema.values
    up_a   = upper.values
    lo_a   = lower.values
    ema_a  = ema.values
    n      = len(close)

    port      = _Portfolio(initial_capital, commission, risk_pct)
    trend_cur = 0
    pending   = 0        # 1 = enter long next open
    trail_stop = np.nan  # trailing stop level (ratchets up)

    live_start = max(trade_start_idx, 1)
    live_len   = n - live_start
    equity_curve = np.empty(max(live_len, 1))
    equity_curve[0] = initial_capital
    trades: list[dict] = []

    for i in range(1, n):
        if np.isnan(up_a[i]) or np.isnan(lo_a[i]) or np.isnan(zl_a[i]) or np.isnan(ema_a[i]):
            if i >= live_start:
                equity_curve[i - live_start] = port.mtm(close[i])
            continue

        # ── 1. Execute pending long entry at this bar's open ──────────────
        if pending == 1 and i >= live_start and port.position == 0:
            exec_price = open_[i]
            port.enter_long(exec_price)
            trail_stop = lo_a[i]   # initialise trailing stop at lower band
            trades.append({"entry": port.entry_price,
                            "notional": port.notional, "bar_in": i})
            pending = 0

        # ── 2. Manage open position ────────────────────────────────────────
        if port.position == 1 and i >= live_start:
            # Ratchet trailing stop up to today's lower band
            if not np.isnan(lo_a[i]):
                trail_stop = max(trail_stop, lo_a[i])

            # Check if low breached the trailing stop
            if low_[i] <= trail_stop:
                fill_price = min(open_[i], trail_stop)
                pnl = port.exit_long(fill_price)
                if trades and "pnl" not in trades[-1]:
                    trades[-1].update({"exit": fill_price, "pnl": pnl,
                                       "bar_out": i, "reason": "trail"})
                trail_stop = np.nan
                pending    = 0

        # ── 3. Update trend state from this bar's close ───────────────────
        if not np.isnan(up_a[i]) and not np.isnan(ema_a[i]):
            if close[i] > up_a[i] and close[i] > ema_a[i]:
                trend_cur = 1
            elif close[i] < lo_a[i] and close[i] < ema_a[i]:
                trend_cur = -1

        # ── 4. Evaluate long entry signal ─────────────────────────────────
        if i >= live_start and port.position == 0 and pending == 0:
            zlema_cross_up = close[i] > zl_a[i] and close[i - 1] <= zl_a[i - 1]
            if zlema_cross_up and trend_cur == 1 and close[i] > ema_a[i]:
                pending = 1

        if i >= live_start:
            equity_curve[i - live_start] = port.mtm(close[i])

    # Force-close at last bar
    if port.position == 1:
        pnl = port.exit_long(close[-1])
        if trades and "pnl" not in trades[-1]:
            trades[-1].update({"exit": close[-1], "pnl": pnl,
                               "bar_out": n - 1, "reason": "EOD"})

    equity_curve[-1] = port.cash if port.position == 0 else port.mtm(close[-1])

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

    peak  = np.maximum.accumulate(equity)
    dd    = (equity - peak) / np.where(peak != 0, peak, 1.0)
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
