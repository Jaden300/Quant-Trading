"""
strategy_utbot.py — UT Bot + EMA/RSI/ADX backtest engine (long-only, daily bars).

Adapted from "UT BOT - SHAH 200/50 EMA". Session/news filters dropped;
fixed-dollar TP/SL replaced with the UT Bot trail itself as exit.

UT Bot trailing stop:
  n_loss = ATR(atr_period) * sensitivity
  if close > prev_trail and close[1] > prev_trail:
      trail = max(prev_trail, close - n_loss)
  elif close < prev_trail and close[1] < prev_trail:
      trail = min(prev_trail, close + n_loss)
  elif close > prev_trail:
      trail = close - n_loss
  else:
      trail = close + n_loss

Signals (fire at bar close → execute at next open):
  utBuy  = close crosses above trail  AND  close > ema_slow
           AND ema50 > ema_slow  AND  RSI > rsi_buy  AND  ADX > adx_min
  utSell = trail crosses above close  →  exit long
"""

from __future__ import annotations

import numpy as np
import pandas as pd


EMA_FAST = 50   # fixed trend momentum EMA


# ── Indicators ────────────────────────────────────────────────────────────────

def _atr(data: pd.DataFrame, length: int) -> np.ndarray:
    prev  = data["Close"].shift(1)
    tr    = pd.concat([
        data["High"] - data["Low"],
        (data["High"] - prev).abs(),
        (data["Low"]  - prev).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(length).mean().values


def _ema(series: pd.Series, length: int) -> np.ndarray:
    return series.ewm(span=length, adjust=False).mean().values


def _rsi(close: pd.Series, length: int) -> np.ndarray:
    delta    = close.diff()
    gain     = delta.clip(lower=0)
    loss     = (-delta).clip(lower=0)
    avg_gain = gain.ewm(alpha=1.0 / length, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / length, adjust=False).mean()
    rs       = avg_gain / avg_loss.replace(0, np.nan)
    return (100.0 - 100.0 / (1.0 + rs)).values


def _adx(data: pd.DataFrame, length: int) -> np.ndarray:
    high  = data["High"]
    low   = data["Low"]
    close = data["Close"]

    prev_high  = high.shift(1)
    prev_low   = low.shift(1)
    prev_close = close.shift(1)

    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low  - prev_close).abs(),
    ], axis=1).max(axis=1)

    up_move   = high - prev_high
    down_move = prev_low - low

    plus_dm  = np.where((up_move > down_move) & (up_move > 0),   up_move,   0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    alpha = 1.0 / length
    tr_s       = tr.ewm(alpha=alpha, adjust=False).mean()
    plus_dm_s  = pd.Series(plus_dm, index=data.index).ewm(alpha=alpha, adjust=False).mean()
    minus_dm_s = pd.Series(minus_dm, index=data.index).ewm(alpha=alpha, adjust=False).mean()

    plus_di  = 100.0 * plus_dm_s  / tr_s.replace(0, np.nan)
    minus_di = 100.0 * minus_dm_s / tr_s.replace(0, np.nan)

    di_sum  = (plus_di + minus_di).replace(0, np.nan)
    dx      = 100.0 * (plus_di - minus_di).abs() / di_sum
    adx_out = dx.ewm(alpha=alpha, adjust=False).mean()
    return adx_out.values


def _ut_bot_trail(close: np.ndarray, atr: np.ndarray, sensitivity: float) -> np.ndarray:
    n      = len(close)
    trail  = np.zeros(n)
    n_loss = atr * sensitivity

    for i in range(1, n):
        if np.isnan(n_loss[i]):
            trail[i] = trail[i - 1]
            continue
        prev = trail[i - 1]
        c    = close[i]
        cp   = close[i - 1]
        nl   = n_loss[i]

        if c > prev and cp > prev:
            trail[i] = max(prev, c - nl)
        elif c < prev and cp < prev:
            trail[i] = min(prev, c + nl)
        elif c > prev:
            trail[i] = c - nl
        else:
            trail[i] = c + nl

    return trail


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
        return self.cash + self.notional * (1.0 + (price - self.entry_price) / self.entry_price)


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
    sensitivity = float(params["sensitivity"])
    atr_period  = int(params["atr_period"])
    ema_slow    = int(params["ema_slow"])
    rsi_buy     = float(params["rsi_buy"])
    adx_min     = float(params["adx_min"])

    atr_a   = _atr(data, atr_period)
    ema50_a = _ema(data["Close"], EMA_FAST)
    ema200_a = _ema(data["Close"], ema_slow)
    rsi_a   = _rsi(data["Close"], 14)
    adx_a   = _adx(data, 14)
    close_a = data["Close"].values
    open_a  = data["Open"].values
    n       = len(close_a)

    trail_a = _ut_bot_trail(close_a, atr_a, sensitivity)

    port      = _Portfolio(initial_capital, commission, risk_pct)
    pending   = 0   # 1 = enter long,  -1 = exit long

    live_start   = max(trade_start_idx, 1)
    live_len     = n - live_start
    equity_curve = np.empty(max(live_len, 1))
    equity_curve[0] = initial_capital
    trades: list[dict] = []

    for i in range(1, n):
        if np.isnan(trail_a[i]) or np.isnan(ema200_a[i]) or np.isnan(adx_a[i]):
            if i >= live_start:
                equity_curve[i - live_start] = port.mtm(close_a[i])
            continue

        # ── 1. Execute pending signal at this bar's open ──────────────────
        if pending != 0 and i >= live_start:
            exec_price = open_a[i]
            if pending == 1 and port.position == 0:
                port.enter_long(exec_price)
                trades.append({"entry": port.entry_price,
                                "notional": port.notional, "bar_in": i})
            elif pending == -1 and port.position == 1:
                pnl = port.exit_long(exec_price)
                if trades and "pnl" not in trades[-1]:
                    trades[-1].update({"exit": exec_price, "pnl": pnl, "bar_out": i})
            pending = 0

        # ── 2. Evaluate signals from this bar's close ─────────────────────
        if i >= live_start:
            ut_buy  = close_a[i] > trail_a[i] and close_a[i - 1] <= trail_a[i - 1]
            ut_sell = trail_a[i] > close_a[i] and trail_a[i - 1] <= close_a[i - 1]

            bull_trend = close_a[i] > ema200_a[i] and ema50_a[i] > ema200_a[i]

            if port.position == 0 and pending == 0:
                if (ut_buy and bull_trend
                        and rsi_a[i] > rsi_buy
                        and adx_a[i] > adx_min):
                    pending = 1

            elif port.position == 1:
                if ut_sell:
                    pending = -1

            equity_curve[i - live_start] = port.mtm(close_a[i])

    # Force-close at last bar
    if port.position == 1:
        pnl = port.exit_long(close_a[-1])
        if trades and "pnl" not in trades[-1]:
            trades[-1].update({"exit": close_a[-1], "pnl": pnl,
                               "bar_out": n - 1, "reason": "EOD"})

    equity_curve[-1] = port.cash if port.position == 0 else port.mtm(close_a[-1])

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
