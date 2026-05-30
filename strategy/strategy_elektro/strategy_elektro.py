"""
strategy_elektro.py — Tetrad Elektro Community backtest engine (long-only).

Logic:
  basis = SMA(close, bb_length)
  dev   = bb_mult * stdev(close, bb_length)
  upper = basis + dev
  lower = basis - dev
  rsi   = RSI(close, rsi_length)

  Long entry signal:  close <= lower  AND  rsi <= rsi_oversold
  Long exit  signal:  close >= upper  AND  rsi >= rsi_exit

  Signal fires at bar close → executes at NEXT bar's open (pending pattern).
  No TP/SL — hold until exit signal fires.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ── Indicators ────────────────────────────────────────────────────────────────

def _rsi(close: pd.Series, length: int) -> pd.Series:
    delta = close.diff()
    gain  = delta.clip(lower=0)
    loss  = (-delta).clip(lower=0)
    avg_gain = gain.ewm(alpha=1.0 / length, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / length, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100.0 - (100.0 / (1.0 + rs))


def calculate_bands(
    data: pd.DataFrame,
    bb_length: int,
    bb_mult: float,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Return (basis, upper_band, lower_band)."""
    src   = data["Close"]
    basis = src.rolling(bb_length).mean()
    dev   = src.rolling(bb_length).std(ddof=0) * bb_mult
    return basis, basis + dev, basis - dev


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
    return_trades: bool = False,
    trade_start_idx: int = 0,
) -> dict:
    bb_length    = int(params["bb_length"])
    bb_mult      = float(params["bb_mult"])
    rsi_length   = int(params["rsi_length"])
    rsi_oversold = float(params["rsi_oversold"])
    rsi_exit     = float(params["rsi_exit"])

    _, upper, lower = calculate_bands(data, bb_length, bb_mult)
    rsi = _rsi(data["Close"], rsi_length)

    close  = data["Close"].values
    open_  = data["Open"].values
    up_a   = upper.values
    lo_a   = lower.values
    rsi_a  = rsi.values
    n      = len(close)

    port     = _Portfolio(initial_capital, commission, risk_pct)
    pending  = 0   # 1 = enter long, -1 = exit long

    live_start = max(trade_start_idx, 1)
    live_len   = n - live_start
    equity_curve = np.empty(max(live_len, 1))
    equity_curve[0] = initial_capital
    trades: list[dict] = []

    for i in range(1, n):
        if np.isnan(up_a[i]) or np.isnan(lo_a[i]) or np.isnan(rsi_a[i]):
            if i >= live_start:
                equity_curve[i - live_start] = port.mtm(close[i])
            continue

        # ── Execute pending signal at this bar's open ──────────────────────
        if pending != 0 and i >= live_start:
            exec_price = open_[i]
            if pending == 1 and port.position == 0:
                port.enter_long(exec_price)
                trades.append({"entry": port.entry_price,
                                "notional": port.notional, "bar_in": i})
            elif pending == -1 and port.position == 1:
                pnl = port.exit_long(exec_price)
                if trades and "pnl" not in trades[-1]:
                    trades[-1].update({"exit": exec_price, "pnl": pnl, "bar_out": i})
            pending = 0

        # ── Evaluate signals from this bar's close ─────────────────────────
        if i >= live_start:
            if port.position == 0:
                if close[i] <= lo_a[i] and rsi_a[i] <= rsi_oversold:
                    pending = 1
            elif port.position == 1:
                if close[i] >= up_a[i] and rsi_a[i] >= rsi_exit:
                    pending = -1

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
