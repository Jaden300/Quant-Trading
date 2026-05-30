"""
strategy_dca.py — DCA Long backtest engine (long-only, daily bars).

Adapted from "[3Commas] Gold DCA Long" by demeth5D.

Nonstop: when no deal is open, immediately queues a new base order.
Safety orders are limit fills checked intrabar via low ≤ trigger.
Take profit checked intrabar via high ≥ avg_entry × (1 + tp_pct).
Multiple SOs can fill in a single bar (while loop, same as Pine original).

Sizing: bo_vol = cash × risk_pct / total_factor
where total_factor = 1 + sum(SO_RATIO × VOL_COEF^(i−1)) for i=1..max_so
This guarantees all SOs fit within the risk_pct budget from deal open.

Fixed config (from original 3Commas bot):
  SO_RATIO = 9/12 = 0.75   (SO volume relative to BO)
  VOL_COEF = 1.05           (each SO 5% larger than previous)
  STEP_COEF = 1.0           (constant deviation step)
"""

from __future__ import annotations

import numpy as np
import pandas as pd


_SO_RATIO  = 9.0 / 12.0
_VOL_COEF  = 1.05
_STEP_COEF = 1.0


class _Portfolio:
    def __init__(self, initial_capital: float, commission: float):
        self.cash      = initial_capital
        self._comm     = commission
        self.qty       = 0.0
        self._spent    = 0.0   # raw volume deployed (sum of vol_i, for avg price)
        self._cash_out = 0.0   # cash committed incl. commissions

    def enter(self, price: float, vol: float) -> None:
        self.qty       += vol / price
        self._spent    += vol
        self._cash_out += vol * (1.0 + self._comm)
        self.cash      -= vol * (1.0 + self._comm)

    def close(self, price: float) -> float:
        proceeds       = self.qty * price * (1.0 - self._comm)
        pnl            = proceeds - self._cash_out
        self.cash     += proceeds
        self.qty       = 0.0
        self._spent    = 0.0
        self._cash_out = 0.0
        return pnl

    @property
    def avg_price(self) -> float:
        return self._spent / self.qty if self.qty > 0 else 0.0

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
    tp_pct  = float(params["tp_pct"]) / 100.0
    so_step = float(params["so_step"])
    max_so  = int(params["max_so"])

    # Total scaling factor: ensures full SO ladder fits in risk_pct budget
    total_factor = 1.0
    for k in range(1, max_so + 1):
        total_factor += _SO_RATIO * (_VOL_COEF ** (k - 1))

    close_a = data["Close"].values
    open_a  = data["Open"].values
    high_a  = data["High"].values
    low_a   = data["Low"].values
    n       = len(close_a)

    port     = _Portfolio(initial_capital, commission)
    in_deal  = False
    pending  = True    # queue first BO immediately at live_start
    base_p   = 0.0
    so_count = 0
    bo_vol   = 0.0
    bar_in   = 0

    live_start   = max(trade_start_idx, 1)
    live_len     = n - live_start
    equity_curve = np.empty(max(live_len, 1))
    equity_curve[0] = initial_capital
    trades: list[dict] = []

    for i in range(1, n):
        if i < live_start:
            continue

        # ── 1. Execute pending base order at this bar's open ──────────
        if pending and not in_deal:
            bo_vol = port.cash * risk_pct / total_factor
            if bo_vol * (1.0 + commission) <= port.cash and bo_vol > 0:
                port.enter(open_a[i], bo_vol)
                base_p   = open_a[i]
                in_deal  = True
                so_count = 0
                bar_in   = i
            pending = False

        # ── 2. Safety orders + TP (intrabar) ─────────────────────────
        if in_deal:
            # Safety orders — while loop fires multiple in one bar
            while so_count < max_so:
                next_n = so_count + 1
                dev    = so_step * next_n   # constant step_coef = 1.0
                trig   = base_p * (1.0 - dev / 100.0)
                vol    = bo_vol * _SO_RATIO * (_VOL_COEF ** (next_n - 1))
                if low_a[i] <= trig and vol * (1.0 + commission) <= port.cash:
                    port.enter(trig, vol)
                    so_count += 1
                else:
                    break

            # Take profit
            tp = port.avg_price * (1.0 + tp_pct)
            if high_a[i] >= tp:
                pnl = port.close(tp)
                trades.append({"pnl": pnl, "bar_in": bar_in, "bar_out": i})
                in_deal = False
                pending = True   # nonstop reload

        equity_curve[i - live_start] = port.mtm(close_a[i])

    # Force-close at last bar
    if in_deal and port.qty > 0:
        pnl = port.close(close_a[-1])
        trades.append({"pnl": pnl, "bar_in": bar_in, "bar_out": n - 1,
                        "reason": "EOD"})

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
