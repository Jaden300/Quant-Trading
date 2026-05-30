"""
strategy_ichimoku.py — Ichimoku Cloud backtest engine.

HOW IT WORKS:
  Three-part confirmation system (all optional, all on by default):
  1. TK Cross:     Tenkan-Sen crosses above Kijun-Sen     → primary entry signal
  2. Cloud Color:  Future cloud is bullish (SpanA > SpanB) → trend regime filter
  3. Chikou:       close > close[kijun_period] bars ago    → momentum confirmation

Entry:
  All enabled conditions met at bar close → go long at next bar's open.

Exit (whichever comes first):
  Opposite TK cross (Tenkan crosses below Kijun)    → exit at next bar's open
  Price falls below visible cloud bottom             → exit at next bar's open
  Optional SL: low <= entry * (1 - sl_pct)          → exit intraday at sl_level

Long-only. No TP — holds until the trend signal reverses.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _calc_ichimoku(
    data: pd.DataFrame,
    tenkan_period: int,
    kijun_period: int,
    senkou_b_period: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    high = data["High"]
    low  = data["Low"]
    tenkan   = (high.rolling(tenkan_period).max() + low.rolling(tenkan_period).min()) / 2
    kijun    = (high.rolling(kijun_period).max()  + low.rolling(kijun_period).min())  / 2
    senkou_a = (tenkan + kijun) / 2
    senkou_b = (high.rolling(senkou_b_period).max() + low.rolling(senkou_b_period).min()) / 2
    return tenkan.values, kijun.values, senkou_a.values, senkou_b.values


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
    tenkan_period   = int(params["tenkan_period"])
    kijun_period    = int(params["kijun_period"])
    senkou_b_period = int(params["senkou_b_period"])
    use_cloud       = bool(params.get("use_cloud",  True))
    use_chikou      = bool(params.get("use_chikou", True))
    sl_pct          = float(params.get("sl_pct", 0.0))

    displacement = kijun_period   # standard Ichimoku: displacement equals kijun period

    tenkan, kijun, senkou_a, senkou_b = _calc_ichimoku(
        data, tenkan_period, kijun_period, senkou_b_period
    )

    close  = data["Close"].values
    open_  = data["Open"].values
    low_   = data["Low"].values
    n      = len(close)

    # Visible cloud at bar i = span values calculated displacement bars ago
    cloud_a = np.full(n, np.nan)
    cloud_b = np.full(n, np.nan)
    if displacement < n:
        cloud_a[displacement:] = senkou_a[:n - displacement]
        cloud_b[displacement:] = senkou_b[:n - displacement]

    live_start = max(trade_start_idx, senkou_b_period + displacement)
    port       = _Portfolio(initial_capital, commission, risk_pct)
    pending    = 0      # 1 = enter long, -1 = exit long
    sl_level   = 0.0

    equity_curve = np.full(n, float(initial_capital))
    trades: list[dict] = []

    for i in range(1, n):
        if np.isnan(tenkan[i]) or np.isnan(kijun[i]):
            equity_curve[i] = port.mtm(close[i])
            continue

        # ── 1. Execute pending at open ─────────────────────────────────────
        if i >= live_start:
            if pending == 1 and port.position == 0:
                port.enter_long(open_[i])
                sl_level = port.entry_price * (1.0 - sl_pct) if sl_pct > 0 else 0.0
                trades.append({"entry": port.entry_price,
                               "notional": port.notional, "bar_in": i})
            elif pending == -1 and port.position == 1:
                pnl = port.exit_long(open_[i])
                if trades and "pnl" not in trades[-1]:
                    trades[-1].update({"exit": open_[i], "pnl": pnl,
                                       "bar_out": i, "reason": "signal"})
                sl_level = 0.0
            pending = 0

        # ── 2. Intraday SL ─────────────────────────────────────────────────
        if port.position == 1 and sl_pct > 0 and sl_level > 0 and i >= live_start:
            if low_[i] <= sl_level:
                fill = open_[i] if open_[i] <= sl_level else sl_level
                pnl  = port.exit_long(fill)
                if trades and "pnl" not in trades[-1]:
                    trades[-1].update({"exit": fill, "pnl": pnl,
                                       "bar_out": i, "reason": "stop"})
                sl_level = 0.0

        # ── 3. Generate signals at bar close ───────────────────────────────
        if i >= live_start:
            if port.position == 0 and pending == 0:
                tk_cross_up = tenkan[i - 1] < kijun[i - 1] and tenkan[i] >= kijun[i]
                if tk_cross_up:
                    cloud_ok  = (not use_cloud) or (senkou_a[i] > senkou_b[i])
                    chikou_ok = True
                    if use_chikou and i >= displacement:
                        chikou_ok = close[i] > close[i - displacement]
                    if cloud_ok and chikou_ok:
                        pending = 1

            if port.position == 1 and pending == 0:
                tk_cross_dn = tenkan[i - 1] > kijun[i - 1] and tenkan[i] <= kijun[i]
                below_cloud = False
                if not np.isnan(cloud_a[i]) and not np.isnan(cloud_b[i]):
                    below_cloud = close[i] < min(cloud_a[i], cloud_b[i])
                if tk_cross_dn or below_cloud:
                    pending = -1

        equity_curve[i] = port.mtm(close[i])

    # Force-close at last bar
    if port.position == 1:
        pnl = port.exit_long(close[-1])
        if trades and "pnl" not in trades[-1]:
            trades[-1].update({"exit": close[-1], "pnl": pnl,
                               "bar_out": n - 1, "reason": "EOD"})

    equity_curve[-1] = port.cash if port.position == 0 else port.mtm(close[-1])
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
        hold_bars     = [t["bar_out"] - t["bar_in"]
                         for t in closed if "bar_out" in t and "bar_in" in t]
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
