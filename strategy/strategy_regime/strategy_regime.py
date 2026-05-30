"""
strategy_regime.py — Regime Execution Strategy [JOAT] (long-only, daily bars).

Adapted from "Regime Execution Strategy [JOAT]" by officialjackofalltrades.

Three engines:
  1. RLS Forecast — recursive least-squares trend line + adaptive error bands
     (forgetting factor λ downweights old data; rebase every 200 bars)
  2. Channel — highest-high/lowest-low tracker; decays toward price after
     reset_len bars of no new extreme (prevents stale channel bounds)
  3. Regime — composite score weighted across forecast/channel/pressure/structure

Entry:  longBreakout (close x-over forecastMean) OR longReclaim (close x-over lowerBand)
        with bull regime (score > 0.12, RVOL ≥ 0.80) and ≥ 2 bull votes
        → queue entry, execute at next bar open

Exit:   1. Intrabar TP/SL (GTC style): dynamic stop = max(entry − ATR*stop_atr, lowerCore)
           dynamic target = entry + ATR*target_atr (uses current-bar ATR)
        2. Regime-flip close: bear_votes ≥ 2 AND score < −0.10 → exit at bar close
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ── Fixed parameters (not grid-searched) ─────────────────────────────────────

_REG_LEN         = 80
_ATR_MIX         = 0.35
_REBASE_BARS     = 200
_CHANNEL_RESET   = 20
_CHANNEL_ALPHA   = 0.50
_FAST_EMA_LEN    = 21
_SLOW_EMA_LEN    = 55
_PRESSURE_LEN    = 20
_PRESSURE_THRESH = 0.10
_RVOL_MIN        = 0.80
_VOTE_THRESH     = 2
_REGIME_THRESH   = 0.12


# ── Indicator helpers ─────────────────────────────────────────────────────────

def _ema(series: pd.Series, length: int) -> np.ndarray:
    return series.ewm(span=length, adjust=False).mean().values


def _atr14(data: pd.DataFrame) -> np.ndarray:
    prev = data["Close"].shift(1)
    tr = pd.concat([
        data["High"] - data["Low"],
        (data["High"] - prev).abs(),
        (data["Low"]  - prev).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(span=14, adjust=False).mean().values


def _rolling_std(arr: np.ndarray, length: int) -> np.ndarray:
    return pd.Series(arr).rolling(length).std(ddof=1).values


def _rolling_max(arr: np.ndarray, length: int) -> np.ndarray:
    return pd.Series(arr).rolling(length).max().values


def _rolling_min(arr: np.ndarray, length: int) -> np.ndarray:
    return pd.Series(arr).rolling(length).min().values


def _rolling_mean(arr: np.ndarray, length: int) -> np.ndarray:
    return pd.Series(arr).rolling(length).mean().values


# ── RLS Forecast Engine ───────────────────────────────────────────────────────

def _compute_forecast(
    src: np.ndarray,
    atr: np.ndarray,
    lambda_f: float,
    band_mult: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Recursive least-squares linear trend + error bands."""
    n  = len(src)
    fm = np.full(n, np.nan)
    ub = np.full(n, np.nan)
    lb = np.full(n, np.nan)

    beta0    = src[0]
    beta1    = 0.0
    p00      = 1000.0
    p01      = 0.0
    p11      = 1000.0
    err_ewma = 0.0
    base_bar = 0

    for i in range(n):
        if i > 0 and _REBASE_BARS > 0 and i % _REBASE_BARS == 0:
            base_bar = i
            beta0    = src[i]
            beta1    = 0.0
            p00      = 1000.0
            p01      = 0.0
            p11      = 1000.0
            err_ewma = 0.0

        x_norm  = max(0.0, (i - base_bar) / max(1.0, float(_REG_LEN)))
        f_mean  = beta0 + beta1 * x_norm        # pre-update prediction
        err     = src[i] - f_mean
        err_ewma = lambda_f * err_ewma + (1.0 - lambda_f) * err * err

        px0   = p00 + p01 * x_norm
        px1   = p01 + p11 * x_norm
        denom = lambda_f + px0 + x_norm * px1

        if denom != 0.0:
            k0     = px0 / denom
            k1     = px1 / denom
            beta0 += k0 * err
            beta1 += k1 * err
            p00    = (p00 - k0 * px0) / lambda_f
            p01    = (p01 - k0 * px1) / lambda_f
            p11    = (p11 - k1 * px1) / lambda_f

        atr_v  = atr[i] if not np.isnan(atr[i]) else 0.0
        spread = max(0.0, np.sqrt(max(err_ewma, 0.0)) * band_mult + atr_v * _ATR_MIX)

        fm[i] = f_mean
        ub[i] = f_mean + spread
        lb[i] = f_mean - spread

    return fm, ub, lb


# ── Channel Engine ────────────────────────────────────────────────────────────

def _compute_channel(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    open_: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Adaptive channel: track extremes, decay toward price after reset_len bars."""
    n      = len(high)
    uc_arr = np.full(n, np.nan)
    lc_arr = np.full(n, np.nan)
    cs_arr = np.zeros(n, dtype=int)

    u_core = high[0]
    l_core = low[0]
    u_age  = 0
    l_age  = 0
    state  = 0

    for i in range(n):
        new_upper = high[i] >= u_core
        new_lower = low[i]  <= l_core

        if new_upper:
            u_core = high[i]
            u_age  = 0
            state  = 1
        else:
            u_age += 1
            if u_age >= _CHANNEL_RESET:
                u_core = u_core * (1.0 - _CHANNEL_ALPHA) + high[i] * _CHANNEL_ALPHA
                u_age  = 0

        if new_lower:
            l_core = low[i]
            l_age  = 0
            state  = -1
        else:
            l_age += 1
            if l_age >= _CHANNEL_RESET:
                l_core = l_core * (1.0 - _CHANNEL_ALPHA) + low[i] * _CHANNEL_ALPHA
                l_age  = 0

        ch_mid = (u_core + l_core) / 2.0
        # Bullish key reversal: new low but closed above mid on up-candle
        if new_lower and close[i] > ch_mid and close[i] > open_[i]:
            state = 1
        # Bearish key reversal: new high but closed below mid on down-candle
        if new_upper and close[i] < ch_mid and close[i] < open_[i]:
            state = -1

        uc_arr[i] = u_core
        lc_arr[i] = l_core
        cs_arr[i] = state

    return uc_arr, lc_arr, cs_arr


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
    lambda_f   = float(params["lambda_f"])
    band_mult  = float(params["band_mult"])
    stop_atr   = float(params["stop_atr"])
    target_atr = float(params["target_atr"])

    close_a = data["Close"].values
    open_a  = data["Open"].values
    high_a  = data["High"].values
    low_a   = data["Low"].values
    vol_a   = data["Volume"].values.astype(float)
    n       = len(close_a)

    atr_a    = _atr14(data)
    fast_ema = _ema(data["Close"], _FAST_EMA_LEN)
    slow_ema = _ema(data["Close"], _SLOW_EMA_LEN)
    vol_sma  = _rolling_mean(vol_a, 20)

    roc_a = np.full(n, np.nan)
    for i in range(_PRESSURE_LEN, n):
        if close_a[i - _PRESSURE_LEN] != 0:
            roc_a[i] = (close_a[i] - close_a[i - _PRESSURE_LEN]) / close_a[i - _PRESSURE_LEN] * 100.0
    roc_std   = _rolling_std(roc_a, _PRESSURE_LEN)
    roll_high = _rolling_max(high_a, _PRESSURE_LEN)
    roll_low  = _rolling_min(low_a,  _PRESSURE_LEN)

    fm_a, ub_a, lb_a = _compute_forecast(close_a, atr_a, lambda_f, band_mult)
    uc_a, lc_a, cs_a = _compute_channel(high_a, low_a, close_a, open_a)

    port    = _Portfolio(initial_capital, commission, risk_pct)
    pending = 0

    live_start   = max(trade_start_idx, 1)
    live_len     = n - live_start
    equity_curve = np.empty(max(live_len, 1))
    equity_curve[0] = initial_capital
    trades: list[dict] = []

    for i in range(1, n):
        if (np.isnan(fm_a[i]) or np.isnan(atr_a[i]) or
                np.isnan(roc_a[i]) or np.isnan(roc_std[i]) or
                np.isnan(roll_high[i]) or np.isnan(roll_low[i])):
            if i >= live_start:
                equity_curve[i - live_start] = port.mtm(close_a[i])
            continue

        # ── 1. Intrabar TP/SL (GTC orders) ───────────────────────────────────
        if port.position == 1 and i >= live_start:
            stop_cand = max(port.entry_price - atr_a[i] * stop_atr, lc_a[i])
            stop_px   = min(stop_cand, port.entry_price * (1.0 - 1e-6))
            target_px = port.entry_price + atr_a[i] * target_atr

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
            port.enter_long(open_a[i])
            trades.append({"entry": port.entry_price,
                            "notional": port.notional, "bar_in": i})
            pending = 0

        # ── 3. Evaluate signals from this bar's close ─────────────────────────
        if i >= live_start:
            ch_mid   = (uc_a[i] + lc_a[i]) / 2.0
            ch_range = max(1e-10, uc_a[i] - lc_a[i])
            band_sp  = max(1e-10, ub_a[i] - fm_a[i])

            rvol = vol_a[i] / max(vol_sma[i], 1.0) if not np.isnan(vol_sma[i]) else 0.0

            mom_std    = max(abs(roc_std[i]), 1e-10)
            mom_norm   = np.tanh(roc_a[i] / mom_std)
            pb_spread  = max(1e-10, roll_high[i] - roll_low[i])
            pb_bias    = (close_a[i] - roll_low[i]) / pb_spread * 2.0 - 1.0

            struct_bias   = 1.0 if fast_ema[i] >= slow_ema[i] else -1.0
            forecast_bias = np.tanh((close_a[i] - fm_a[i]) / band_sp)
            channel_bias  = (1.0  if cs_a[i] == 1 else
                             -1.0 if cs_a[i] == -1 else
                             np.tanh((close_a[i] - ch_mid) / ch_range))
            pressure_bias = np.tanh((mom_norm + pb_bias) / 2.0)

            regime_score = (forecast_bias * 0.40 + channel_bias * 0.30 +
                            pressure_bias * 0.20 + struct_bias * 0.10)

            bull_votes = ((1 if forecast_bias > 0 else 0) +
                          (1 if channel_bias  > 0 else 0) +
                          (1 if pressure_bias > _PRESSURE_THRESH else 0))
            bear_votes = ((1 if forecast_bias < 0 else 0) +
                          (1 if channel_bias  < 0 else 0) +
                          (1 if pressure_bias < -_PRESSURE_THRESH else 0))

            bull_regime = regime_score > _REGIME_THRESH and rvol >= _RVOL_MIN

            long_breakout = (close_a[i] > fm_a[i] and close_a[i - 1] <= fm_a[i - 1]
                             and close_a[i] > ch_mid
                             and pressure_bias > _PRESSURE_THRESH)
            long_reclaim  = (close_a[i] > lb_a[i] and close_a[i - 1] <= lb_a[i - 1]
                             and channel_bias >= 0
                             and pressure_bias > 0)

            long_signal = (rvol >= _RVOL_MIN and bull_votes >= _VOTE_THRESH
                           and bull_regime and (long_breakout or long_reclaim))

            if port.position == 0 and pending == 0 and long_signal:
                pending = 1
            elif port.position == 1:
                if bear_votes >= _VOTE_THRESH and regime_score < -0.10:
                    pnl = port.exit_long(close_a[i])
                    if trades and "pnl" not in trades[-1]:
                        trades[-1].update({"exit": close_a[i], "pnl": pnl,
                                           "bar_out": i, "reason": "regime_flip"})

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
