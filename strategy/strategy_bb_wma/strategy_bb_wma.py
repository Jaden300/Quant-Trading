"""
strategy_bb_wma.py — Bollinger Band WMA Daily backtest engine.

Adapted from "Bollinger WMA Daily Strat v29" (community Pine Script).

Entry signal (at bar close): hlc3 > upperBand
  AND (close > close[2] OR close > open)
  AND NOT (shadow > 10 × body)
  AND 1-bar cooldown since last exit
→ execute at next bar open

Exit signal (at bar close): close < lowerBand
→ execute at next bar open

Bands:
  src   = hlc3 = (high + low + close) / 3
  basis = WMA(src, bb_length)
  dev   = stdev(src, bb_length)   — population std (ddof=0), matches Pine ta.stdev
  upper = basis + std_buy  × dev
  lower = basis - std_sell × dev

No SL, no TP, no trail — hold until close < lower band.
Seasonal matrix disabled.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _wma(series: pd.Series, length: int) -> np.ndarray:
    weights = np.arange(1, length + 1, dtype=float)
    return series.rolling(length).apply(
        lambda x: np.dot(x, weights) / weights.sum(), raw=True
    ).values


def _std_pop(series: pd.Series, length: int) -> np.ndarray:
    return series.rolling(length).std(ddof=0).values


class _Portfolio:
    def __init__(self, initial_capital: float, commission: float, risk_pct: float):
        self.cash  = initial_capital
        self._comm = commission
        self._risk = risk_pct
        self.qty   = 0.0
        self._cost = 0.0

    @property
    def in_position(self) -> bool:
        return self.qty > 0

    def enter(self, price: float) -> None:
        vol  = self.cash * self._risk
        cost = vol * (1.0 + self._comm)
        if cost > self.cash:
            vol  = self.cash / (1.0 + self._comm)
            cost = self.cash
        self.qty   += vol / price
        self._cost += cost
        self.cash  -= cost

    def close(self, price: float) -> float:
        proceeds   = self.qty * price * (1.0 - self._comm)
        pnl        = proceeds - self._cost
        self.cash += proceeds
        self.qty   = 0.0
        self._cost = 0.0
        return pnl

    def mtm(self, price: float) -> float:
        return self.cash + self.qty * price


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
    bb_length = int(params["bb_length"])
    std_buy   = float(params["std_buy"])
    std_sell  = float(params["std_sell"])

    close_a = data["Close"].values
    open_a  = data["Open"].values
    high_a  = data["High"].values
    low_a   = data["Low"].values
    n       = len(close_a)

    hlc3    = (high_a + low_a + close_a) / 3.0
    src     = pd.Series(hlc3)
    basis_a = _wma(src, bb_length)
    dev_a   = _std_pop(src, bb_length)
    upper_a = basis_a + std_buy  * dev_a
    lower_a = basis_a - std_sell * dev_a

    warmup      = bb_length + 3
    live_start  = max(trade_start_idx, warmup, 1)
    live_len    = n - live_start
    equity_curve = np.empty(max(live_len, 1))
    equity_curve[0] = initial_capital

    port          = _Portfolio(initial_capital, commission, risk_pct)
    trades: list[dict] = []
    pending_entry = False
    pending_exit  = False
    last_exit_bar = -999
    bar_in        = 0

    for i in range(1, n):
        if i < live_start:
            continue

        ei = i - live_start

        # ── 1. Execute pending orders at this bar's open ───────────────
        if pending_exit and port.in_position:
            pnl = port.close(open_a[i])
            trades.append({"pnl": pnl, "bar_in": bar_in, "bar_out": i})
            pending_exit  = False
            last_exit_bar = i

        if pending_entry and not port.in_position:
            port.enter(open_a[i])
            bar_in        = i
            pending_entry = False

        # ── 2. Exit signal at bar close ───────────────────────────────
        if port.in_position and not pending_exit:
            if not np.isnan(lower_a[i]) and close_a[i] < lower_a[i]:
                pending_exit = True

        # ── 3. Entry signal at bar close ──────────────────────────────
        if not port.in_position and not pending_entry and not pending_exit:
            if i > last_exit_bar:   # 1-bar cooldown
                if not np.isnan(upper_a[i]) and hlc3[i] > upper_a[i]:
                    close_gt_close2 = (i >= 2) and (close_a[i] > close_a[i - 2])
                    close_gt_open   = close_a[i] > open_a[i]
                    candle_ok       = close_gt_close2 or close_gt_open

                    body      = abs(close_a[i] - open_a[i])
                    candle_rng = high_a[i] - low_a[i]
                    shadow    = candle_rng - body
                    shadow_ok = not (body > 0 and shadow > 10.0 * body)

                    if candle_ok and shadow_ok:
                        pending_entry = True

        if ei < live_len:
            equity_curve[ei] = port.mtm(close_a[i])

    # Force-close at last bar
    if port.in_position and port.qty > 0:
        pnl = port.close(close_a[-1])
        trades.append({"pnl": pnl, "bar_in": bar_in, "bar_out": n - 1, "reason": "EOD"})

    equity_curve[-1] = port.mtm(close_a[-1])

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

    rets   = np.diff(equity) / np.where(equity[:-1] != 0, equity[:-1], 1.0)
    std    = rets.std()
    sharpe = float(rets.mean() / std * np.sqrt(252)) if std > 0 else 0.0

    peak   = np.maximum.accumulate(equity)
    dd     = (equity - peak) / np.where(peak != 0, peak, 1.0)
    max_dd = float(dd.min() * 100.0)

    n_trades = len(trades)
    if n_trades:
        wins          = [t["pnl"] for t in trades if t["pnl"] > 0]
        losses        = [t["pnl"] for t in trades if t["pnl"] <= 0]
        win_rate      = len(wins) / n_trades * 100.0
        gross_profit  = sum(wins)
        gross_loss    = abs(sum(losses))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")
        hold_bars     = [t["bar_out"] - t["bar_in"] for t in trades
                         if "bar_out" in t and "bar_in" in t]
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
