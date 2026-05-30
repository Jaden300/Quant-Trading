"""
strategy_whale.py — Whale Force backtest engine (long-only, daily bars).

Adapted from "Whale Force" by dropping the 4H MTF wrapper; all indicators
computed natively on daily bars.

Entry (next-bar open):
  thrustBreak: ATR quiet-zone + Donchian breakout + vol Z ≥ threshold
               + strong close (CLV ≥ 0.6) + VWMA uptrend
  absorbBreak: absorption candle (high vol, small body, strong close)
               + closes above yesterday's high + VWMA uptrend

Exit (next-bar open) — voting system, fires when ≥ min_exit_votes of:
  1. Close drops below VWMA × (1 − 0.15 %)
  2. Weak close: CLV ≤ 0.3 AND close < previous low
  3. Volume reversal: yesterday Z ≥ threshold, today Z < 0
  4. Bearish engulfing candle
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ── Fixed parameters ──────────────────────────────────────────────────────────

_VWMA_LEN      = 24
_QUIET_ATR_LEN = 14
_QUIET_REF_LEN = 50
_VOL_LEN       = 50
_EXIT_VWMA_EPS = 0.15   # %


# ── Indicator helpers ─────────────────────────────────────────────────────────

def _vwma(close: pd.Series, volume: pd.Series, length: int) -> np.ndarray:
    cv = close * volume
    vol_roll = volume.rolling(length).sum().replace(0, np.nan)
    return (cv.rolling(length).sum() / vol_roll).values


def _atr_wilder(data: pd.DataFrame, length: int) -> np.ndarray:
    prev = data["Close"].shift(1)
    tr = pd.concat([
        data["High"] - data["Low"],
        (data["High"] - prev).abs(),
        (data["Low"]  - prev).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1.0 / length, adjust=False).mean().values


def _vol_zscore(volume: np.ndarray, length: int) -> np.ndarray:
    vol_s   = pd.Series(volume)
    vol_sma = vol_s.rolling(length).mean()
    vol_std = vol_s.rolling(length).std(ddof=1).replace(0, np.nan)
    return ((vol_s - vol_sma) / vol_std).values


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
    z_thresh       = float(params["z_thresh"])
    don_len        = int(params["don_len"])
    quiet_ratio    = float(params["quiet_ratio"])
    min_exit_votes = int(params["min_exit_votes"])

    close_a = data["Close"].values
    open_a  = data["Open"].values
    high_a  = data["High"].values
    low_a   = data["Low"].values
    vol_a   = data["Volume"].values.astype(float)
    n       = len(close_a)

    vol_z_a       = _vol_zscore(vol_a, _VOL_LEN)
    vwma_a        = _vwma(data["Close"], data["Volume"], _VWMA_LEN)
    vwma2_a       = _vwma(data["Close"], data["Volume"], _VWMA_LEN * 2)
    atr_now_a     = _atr_wilder(data, _QUIET_ATR_LEN)
    atr_ref_a     = pd.Series(atr_now_a).rolling(_QUIET_REF_LEN).mean().values
    # don_high_prev: highest high over don_len bars, using value from previous bar
    don_high_prev = pd.Series(high_a).rolling(don_len).max().shift(1).values

    port    = _Portfolio(initial_capital, commission, risk_pct)
    pending = 0   # 1 = enter long,  -1 = exit long

    live_start   = max(trade_start_idx, 1)
    live_len     = n - live_start
    equity_curve = np.empty(max(live_len, 1))
    equity_curve[0] = initial_capital
    trades: list[dict] = []

    for i in range(1, n):
        if (np.isnan(vol_z_a[i]) or np.isnan(vwma_a[i]) or np.isnan(vwma2_a[i]) or
                np.isnan(atr_now_a[i]) or np.isnan(atr_ref_a[i]) or
                np.isnan(don_high_prev[i])):
            if i >= live_start:
                equity_curve[i - live_start] = port.mtm(close_a[i])
            continue

        # ── 1. Execute pending signal at this bar's open ──────────────────────
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

        # ── 2. Evaluate signals from this bar's close ─────────────────────────
        if i >= live_start:
            bar_range = max(high_a[i] - low_a[i], 1e-10)
            clv       = (close_a[i] - low_a[i]) / bar_range
            body_pct  = abs(close_a[i] - open_a[i]) / bar_range

            trend_ok = close_a[i] > vwma_a[i] and vwma_a[i] > vwma2_a[i]
            is_quiet = atr_ref_a[i] > 0 and (atr_now_a[i] / atr_ref_a[i]) <= quiet_ratio
            is_break = close_a[i] > don_high_prev[i]

            absorp       = vol_z_a[i] >= z_thresh and body_pct <= 0.4 and clv >= 0.6
            thrust_break = (is_quiet and is_break and vol_z_a[i] >= z_thresh
                            and clv >= 0.6 and trend_ok)
            absorb_break = absorp and close_a[i] > high_a[i - 1] and trend_ok
            whale_buy    = thrust_break or absorb_break

            prev_vol_z  = vol_z_a[i - 1] if not np.isnan(vol_z_a[i - 1]) else 0.0
            exit_trend  = close_a[i] < vwma_a[i] * (1.0 - _EXIT_VWMA_EPS / 100.0)
            exit_momo   = clv <= 0.3 and close_a[i] < low_a[i - 1]
            exit_vol    = prev_vol_z >= z_thresh and vol_z_a[i] < 0.0
            bear_engulf = (open_a[i] > close_a[i]
                           and open_a[i] >= close_a[i - 1]
                           and close_a[i] <= open_a[i - 1])
            exit_votes  = int(exit_trend) + int(exit_momo) + int(exit_vol) + int(bear_engulf)
            whale_sell  = exit_votes >= min_exit_votes

            if port.position == 0 and pending == 0 and whale_buy:
                pending = 1
            elif port.position == 1 and pending == 0 and whale_sell:
                pending = -1

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
