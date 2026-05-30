"""
strategy_ott.py — Optimized Trend Tracker (OTT) backtest engine.

HOW IT WORKS:
  Computes an adaptive moving average (VAR by default) then builds an
  asymmetric trailing channel around it. The OTT line smooths the channel
  boundary with a 2-bar lag to reduce whipsaws.

  When the MA crosses above OTT → trend flipped bullish → go long.
  When the MA crosses below OTT → trend flipped bearish → exit long.

Entry:
  MAvg crosses above OTT (2-bar lagged)  — bullish trend confirmed
  → go long at next bar's open

Exit (whichever comes first):
  MAvg crosses below OTT                 — signal → exit at next bar's open
  high >= entry * (1 + tp_pct)           — optional TP hit intraday (0 = off)

MA types supported: VAR (default, adaptive), EMA, WWMA
VAR (Variable-Adjusted Return): adapts smoothing to directional momentum —
  fast in trending conditions, slow and stable in choppy markets.

Adapted from "Optimized Trend Tracker" by KivancOzbilgic / Anil_Ozeksi.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _var_ma(close: pd.Series, length: int) -> np.ndarray:
    """Variable-Adjusted Return MA — adaptive EMA based on directional momentum."""
    alpha  = 2.0 / (length + 1)
    src    = close.values
    n      = len(src)
    result = np.zeros(n)
    result[0] = src[0]

    for i in range(1, n):
        up   = max(src[i] - src[i - 1], 0.0)
        dn   = max(src[i - 1] - src[i], 0.0)
        # 9-bar sums of up/dn moves
        lo   = max(0, i - 9)
        vUD  = sum(max(src[j] - src[j - 1], 0.0) for j in range(lo + 1, i + 1))
        vDD  = sum(max(src[j - 1] - src[j], 0.0) for j in range(lo + 1, i + 1))
        cmo  = (vUD - vDD) / (vUD + vDD) if (vUD + vDD) > 0 else 0.0
        result[i] = alpha * abs(cmo) * src[i] + (1 - alpha * abs(cmo)) * result[i - 1]

    return result


def _ema(close: pd.Series, length: int) -> np.ndarray:
    return close.ewm(span=length, adjust=False).mean().values


def _wwma(close: pd.Series, length: int) -> np.ndarray:
    alpha  = 1.0 / length
    src    = close.values
    n      = len(src)
    result = np.zeros(n)
    result[0] = src[0]
    for i in range(1, n):
        result[i] = alpha * src[i] + (1 - alpha) * result[i - 1]
    return result


def _get_ma(close: pd.Series, length: int, mav: str) -> np.ndarray:
    if mav == "EMA":
        return _ema(close, length)
    if mav == "WWMA":
        return _wwma(close, length)
    return _var_ma(close, length)   # default: VAR


def _calc_ott(mavg: np.ndarray, percent: float) -> tuple[np.ndarray, np.ndarray]:
    """Returns (OTT, MT) arrays matching TradingView's implementation."""
    n         = len(mavg)
    fark      = mavg * percent * 0.01
    long_stop = mavg - fark
    shrt_stop = mavg + fark
    mt        = np.zeros(n)
    ott       = np.zeros(n)
    direction = np.ones(n, dtype=int)

    mt[0]        = long_stop[0]
    direction[0] = 1

    for i in range(1, n):
        prev_ls = mt[i - 1] if direction[i - 1] == 1 else long_stop[i]
        prev_ss = mt[i - 1] if direction[i - 1] == -1 else shrt_stop[i]

        ls = long_stop[i]
        ss = shrt_stop[i]
        ls = max(ls, prev_ls) if mavg[i] > prev_ls else ls
        ss = min(ss, prev_ss) if mavg[i] < prev_ss else ss

        prev_dir = direction[i - 1]
        if prev_dir == -1 and mavg[i] > prev_ss:
            direction[i] = 1
        elif prev_dir == 1 and mavg[i] < prev_ls:
            direction[i] = -1
        else:
            direction[i] = prev_dir

        mt[i] = ls if direction[i] == 1 else ss

        if mavg[i] > mt[i]:
            ott[i] = mt[i] * (200 + percent) / 200
        else:
            ott[i] = mt[i] * (200 - percent) / 200

    return ott, mt


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
    length  = int(params["length"])
    percent = float(params["percent"])
    mav     = str(params["mav"])
    tp_pct  = float(params["tp_pct"])

    close_a = data["Close"].values
    high_a  = data["High"].values
    open_a  = data["Open"].values
    n       = len(close_a)

    mavg_a      = _get_ma(data["Close"], length, mav)
    ott_a, _    = _calc_ott(mavg_a, percent)
    # OTT signal uses OTT[2] (2-bar lag) to match TradingView
    ott_lag     = np.full(n, np.nan)
    ott_lag[2:] = ott_a[:-2]

    live_start = max(trade_start_idx, length + 15)
    port       = _Portfolio(initial_capital, commission, risk_pct)
    tp_level   = 0.0
    pending    = 0

    equity_curve = np.full(n, float(initial_capital))
    trades: list[dict] = []

    for i in range(1, n):
        if np.isnan(ott_lag[i]) or np.isnan(ott_lag[i - 1]):
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
                                       "bar_out": i, "reason": "signal"})
                tp_level = 0.0
            pending = 0

        # ── 2. Check intraday TP ──────────────────────────────────────────
        if port.position == 1 and tp_pct > 0 and high_a[i] >= tp_level and i >= live_start:
            pnl = port.exit_long(tp_level)
            if trades and "pnl" not in trades[-1]:
                trades[-1].update({"exit": tp_level, "pnl": pnl,
                                   "bar_out": i, "reason": "tp"})
            tp_level = 0.0

        # ── 3. Generate signal from MA vs OTT at bar close ────────────────
        if i >= live_start:
            cross_up   = mavg_a[i - 1] < ott_lag[i - 1] and mavg_a[i] >= ott_lag[i]
            cross_down = mavg_a[i - 1] > ott_lag[i - 1] and mavg_a[i] <= ott_lag[i]

            if port.position == 0 and cross_up:
                pending = 1
            elif port.position == 1 and cross_down:
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
