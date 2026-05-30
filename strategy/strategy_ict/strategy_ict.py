"""
strategy_ict.py — ICT + Price Action backtest engine (long-only, daily bars).

Adapted from "ICT + Price Action | XAUUSD & EURUSD [v6]" by demeth5D.
Kill-zone filter (London/NY session hours) and VWAP dropped — both are
intraday-only constructs that don't fire on daily bars. All other logic
is native daily:

  Trend:     close > EMA_slow AND EMA_fast > EMA_slow
  RSI:       RSI(14) < rsi_ob  (avoid overbought entries)
  Setup (any one triggers entry):
    FVG        — high[i-2] < low[i]  (bullish gap / 3-bar inefficiency)
    OrderBlock — close[i-1] < open[i-1] AND close[i] > high[i-1]
                 (bearish candle fully engulfed upward)
    PinBar     — lower wick ≥ 66.7 % of range AND body ≤ 35 % of range
                 (checked on bar i-1, signal fires on bar i)

  Exit: ATR-based SL (entry − sl_mult × ATR) and TP (entry + tp_mult × ATR),
        fixed at entry open price.  Intrabar GTC check via high/low.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ── Fixed parameters ──────────────────────────────────────────────────────────

_EMA_FAST_LEN = 50
_ATR_LEN      = 14
_RSI_LEN      = 14


# ── Indicators ────────────────────────────────────────────────────────────────

def _ema(series: pd.Series, length: int) -> np.ndarray:
    return series.ewm(span=length, adjust=False).mean().values


def _atr_wilder(data: pd.DataFrame, length: int) -> np.ndarray:
    prev = data["Close"].shift(1)
    tr = pd.concat([
        data["High"] - data["Low"],
        (data["High"] - prev).abs(),
        (data["Low"]  - prev).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1.0 / length, adjust=False).mean().values


def _rsi(close: pd.Series, length: int) -> np.ndarray:
    delta    = close.diff()
    gain     = delta.clip(lower=0)
    loss     = (-delta).clip(lower=0)
    avg_gain = gain.ewm(alpha=1.0 / length, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / length, adjust=False).mean()
    rs       = avg_gain / avg_loss.replace(0, np.nan)
    return (100.0 - 100.0 / (1.0 + rs)).values


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


# ── Backtest Engine ───────────────────────────────────────────────────────────

def run_backtest(
    data: pd.DataFrame,
    params: dict,
    initial_capital: float = 1500.0,
    commission: float = 0.015,
    risk_pct: float = 0.33,
    return_equity_curve: bool = False,
    trade_start_idx: int = 0,
) -> dict:
    ema_slow_len = int(params["ema_slow"])
    sl_mult      = float(params["atr_sl"])
    tp_mult      = float(params["atr_tp"])
    rsi_ob       = float(params["rsi_ob"])

    close_a = data["Close"].values
    open_a  = data["Open"].values
    high_a  = data["High"].values
    low_a   = data["Low"].values
    n       = len(close_a)

    ema_fast_a = _ema(data["Close"], _EMA_FAST_LEN)
    ema_slow_a = _ema(data["Close"], ema_slow_len)
    atr_a      = _atr_wilder(data, _ATR_LEN)
    rsi_a      = _rsi(data["Close"], _RSI_LEN)

    port        = _Portfolio(initial_capital, commission, risk_pct)
    pending     = 0       # 1 = enter long
    pending_atr = 0.0     # ATR captured at signal bar for SL/TP levels
    stop_px     = 0.0
    target_px   = 0.0

    live_start   = max(trade_start_idx, 2)   # need i-2 for FVG
    live_len     = n - live_start
    equity_curve = np.empty(max(live_len, 1))
    equity_curve[0] = initial_capital
    trades: list[dict] = []

    for i in range(2, n):
        if (np.isnan(ema_slow_a[i]) or np.isnan(atr_a[i]) or np.isnan(rsi_a[i])):
            if i >= live_start:
                equity_curve[i - live_start] = port.mtm(close_a[i])
            continue

        # ── 1. Intrabar TP/SL (GTC orders) ───────────────────────────────────
        if port.position == 1 and i >= live_start:
            exit_price  = None
            exit_reason = ""
            if low_a[i] <= stop_px:
                exit_price  = min(open_a[i], stop_px) if open_a[i] <= stop_px else stop_px
                exit_reason = "stop"
            elif high_a[i] >= target_px:
                exit_price  = max(open_a[i], target_px) if open_a[i] >= target_px else target_px
                exit_reason = "target"

            if exit_price is not None:
                pnl = port.exit_long(exit_price)
                if trades and "pnl" not in trades[-1]:
                    trades[-1].update({"exit": exit_price, "pnl": pnl,
                                       "bar_out": i, "reason": exit_reason})
                equity_curve[i - live_start] = port.mtm(close_a[i])
                continue

        # ── 2. Execute pending entry at this bar's open ───────────────────────
        if pending == 1 and i >= live_start and port.position == 0:
            entry_raw = open_a[i]
            port.enter_long(entry_raw)
            stop_px   = entry_raw - sl_mult * pending_atr
            target_px = entry_raw + tp_mult * pending_atr
            trades.append({"entry": port.entry_price,
                            "notional": port.notional, "bar_in": i})
            pending = 0

        # ── 3. Evaluate signals from this bar's close ─────────────────────────
        if i >= live_start:
            # Trend filter
            bullish_trend = close_a[i] > ema_slow_a[i] and ema_fast_a[i] > ema_slow_a[i]
            rsi_ok        = rsi_a[i] < rsi_ob

            # ICT setups
            bull_fvg = high_a[i - 2] < low_a[i]
            bull_ob  = close_a[i - 1] < open_a[i - 1] and close_a[i] > high_a[i - 1]

            tr_prev       = high_a[i - 1] - low_a[i - 1]
            if tr_prev > 0:
                lower_wick  = min(open_a[i - 1], close_a[i - 1]) - low_a[i - 1]
                body_prev   = abs(close_a[i - 1] - open_a[i - 1])
                bull_pinbar = (lower_wick >= tr_prev * 0.667
                               and body_prev <= tr_prev * 0.35)
            else:
                bull_pinbar = False

            ict_setup   = bull_fvg or bull_ob or bull_pinbar
            long_signal = bullish_trend and rsi_ok and ict_setup

            if port.position == 0 and pending == 0 and long_signal:
                pending     = 1
                pending_atr = atr_a[i]

            equity_curve[i - live_start] = port.mtm(close_a[i])

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
