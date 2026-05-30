"""
strategy_roji.py — Roji Pattern backtest engine (long-only, daily bars).

Adapted from "Roji Scalping Pattern" Pine Script. Session/MTF filters dropped;
SL replaced with ATR-based stops. Three long setups implemented:

  RBR  (Rally-Base-Rally):  impulse up → tight base → break above base
  DBR  (Drop-Base-Rally):   impulse down → tight base → break above base
  Double Bottom:            two similar pivot lows → break above neckline

Signal fires at bar close → entry executes at NEXT bar's open.
SL = base_low − atr * 0.5 buffer.  TP = entry + risk × rr.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


IMPULSE_LOOKBACK = 4   # bars to measure impulse (fixed)
PIVOT_LEN        = 5   # bars each side for pivot low confirmation
DT_MAX_BARS      = 35  # max bars between two double-bottom lows
SL_BUFFER_ATR    = 0.5 # ATR buffer below base_low for SL


# ── Indicators ────────────────────────────────────────────────────────────────

def _atr(data: pd.DataFrame, length: int) -> pd.Series:
    prev = data["Close"].shift(1)
    tr   = pd.concat([
        data["High"] - data["Low"],
        (data["High"] - prev).abs(),
        (data["Low"]  - prev).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(length).mean()


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
    base_bars    = int(params["base_bars"])
    impulse_mult = float(params["impulse_mult"])
    base_mult    = float(params["base_mult"])
    atr_length   = int(params["atr_length"])
    rr           = float(params["rr"])

    atr_s = _atr(data, atr_length)

    # Base high/low: rolling over the previous base_bars bars of high/low
    base_high_s = data["High"].shift(1).rolling(base_bars).max()
    base_low_s  = data["Low"].shift(1).rolling(base_bars).min()

    # Impulse: move from (base_bars + lookback) ago to base_bars ago
    pre_move_s = (
        data["Close"].shift(base_bars)
        - data["Close"].shift(base_bars + IMPULSE_LOOKBACK)
    )

    close    = data["Close"].values
    high     = data["High"].values
    low      = data["Low"].values
    open_    = data["Open"].values
    atr_a    = atr_s.values
    bh_a     = base_high_s.values
    bl_a     = base_low_s.values
    pm_a     = pre_move_s.values
    n        = len(close)

    port       = _Portfolio(initial_capital, commission, risk_pct)
    pending    = 0       # 1 = enter long at next open
    pending_sl = np.nan

    active_sl  = np.nan
    active_tp  = np.nan

    # Double-bottom state
    prev_low_price = np.nan
    prev_low_bar   = -999
    wait_db        = False
    db_neck        = np.nan

    live_start   = max(trade_start_idx, 1)
    live_len     = n - live_start
    equity_curve = np.empty(max(live_len, 1))
    equity_curve[0] = initial_capital
    trades: list[dict] = []

    for i in range(1, n):
        if np.isnan(atr_a[i]):
            if i >= live_start:
                equity_curve[i - live_start] = port.mtm(close[i])
            continue

        # ── 1. Execute pending entry ──────────────────────────────────────
        if pending == 1 and i >= live_start and port.position == 0:
            exec_price = open_[i]
            port.enter_long(exec_price)
            risk = exec_price - pending_sl
            if risk <= 0:
                risk = atr_a[i]
            active_sl = pending_sl
            active_tp = exec_price + risk * rr
            trades.append({"entry": port.entry_price,
                            "notional": port.notional, "bar_in": i})
            pending    = 0
            pending_sl = np.nan

        # ── 2. Manage open position ───────────────────────────────────────
        if port.position == 1 and i >= live_start:
            hit_sl = low[i] <= active_sl
            hit_tp = high[i] >= active_tp

            if hit_sl or hit_tp:
                if hit_sl:
                    fill = min(open_[i], active_sl)
                    reason = "SL"
                else:
                    fill = open_[i] if open_[i] >= active_tp else active_tp
                    reason = "TP"
                pnl = port.exit_long(fill)
                if trades and "pnl" not in trades[-1]:
                    trades[-1].update({"exit": fill, "pnl": pnl,
                                       "bar_out": i, "reason": reason})
                active_sl = np.nan
                active_tp = np.nan

        # ── 3. Pivot low detection (double bottom) ────────────────────────
        pl_idx = i - PIVOT_LEN
        if pl_idx >= PIVOT_LEN:
            window_start = max(0, pl_idx - PIVOT_LEN)
            window_end   = min(n, pl_idx + PIVOT_LEN + 1)
            if low[pl_idx] == np.min(low[window_start:window_end]):
                curr_low     = low[pl_idx]
                curr_low_bar = pl_idx
                if (not np.isnan(prev_low_price)
                        and curr_low_bar - prev_low_bar <= DT_MAX_BARS
                        and abs(curr_low - prev_low_price) <= atr_a[i] * 1.5):
                    wait_db = True
                    neck_start = max(0, prev_low_bar)
                    neck_end   = min(n, curr_low_bar + 1)
                    db_neck = float(np.max(high[neck_start:neck_end]))
                prev_low_price = curr_low
                prev_low_bar   = curr_low_bar

        # ── 4. Evaluate entry signals ─────────────────────────────────────
        if i >= live_start and port.position == 0 and pending == 0:
            if not (np.isnan(bh_a[i]) or np.isnan(bl_a[i])
                    or np.isnan(pm_a[i]) or np.isnan(atr_a[i])):

                base_range  = bh_a[i] - bl_a[i]
                base_ok     = base_range <= atr_a[i] * base_mult
                break_above = close[i] > bh_a[i] and close[i - 1] <= bh_a[i]

                rbr = (pm_a[i] >  atr_a[i] * impulse_mult) and base_ok and break_above
                dbr = (pm_a[i] < -atr_a[i] * impulse_mult) and base_ok and break_above

                db  = wait_db and not np.isnan(db_neck) and close[i] > db_neck
                if db:
                    wait_db = False
                    db_neck = np.nan

                if rbr or dbr or db:
                    sig_sl  = bl_a[i] - atr_a[i] * SL_BUFFER_ATR
                    pending    = 1
                    pending_sl = sig_sl

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
