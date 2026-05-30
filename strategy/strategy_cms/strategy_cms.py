"""
strategy_cms.py — Caldera Meridian Strategy backtest engine (long-only).

Adapted from "Caldera Meridian Strategy [JOAT]" by officialjackofalltrades.

Modules (all self-contained, no external indicators):
  Regime:    Triple EMA (fast/mid/slow) + MACD + RSI + DMI + rolling VWAP
             (rolling VWAP replaces daily-resetting ta.vwap, which is flat on daily bars)
  Pressure:  Volume-weighted body pressure oscillator
  Auction:   Rolling VWAP ± stdev value area (90-bar window)
  Structure: Break of Structure, Fair Value Gaps, pivot sweeps
  HTF:       Weekly EMA filter (replaces 4h — meaningless on daily timeframe)
  Score:     0-100 weighted execution score; long only when score >= min_score

Entry:  Signal confirmed at bar close → execute at next bar open.
Exit:   SL intraday (low <= stop_level) | TP intraday (high >= target)
        Risk-off at bar close → exit at next bar open.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ── Indicator helpers ─────────────────────────────────────────────────────────

def _ema(s: pd.Series, n: int) -> np.ndarray:
    return s.ewm(span=n, adjust=False).mean().values


def _atr(high: pd.Series, low: pd.Series, close: pd.Series, n: int) -> np.ndarray:
    prev = close.shift(1)
    tr = pd.concat([high - low, (high - prev).abs(), (low - prev).abs()], axis=1).max(axis=1)
    return tr.ewm(span=n, adjust=False).mean().values


def _rsi(close: pd.Series, n: int = 14) -> np.ndarray:
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1.0 / n, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1.0 / n, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    return (100.0 - 100.0 / (1.0 + rs)).fillna(50.0).values


def _macd_hist(close: pd.Series) -> np.ndarray:
    fast = close.ewm(span=12, adjust=False).mean()
    slow_m = close.ewm(span=26, adjust=False).mean()
    macd = fast - slow_m
    return (macd - macd.ewm(span=9, adjust=False).mean()).values


def _dmi(high: pd.Series, low: pd.Series, close: pd.Series,
         n: int = 14) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    up  = high.diff()
    dn  = -low.diff()
    pdm = np.where((up > dn) & (up > 0), up.values, 0.0)
    ndm = np.where((dn > up) & (dn > 0), dn.values, 0.0)
    prev = close.shift(1)
    tr  = pd.concat([high - low, (high - prev).abs(), (low - prev).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1.0 / n, adjust=False).mean()
    pdi = (pd.Series(pdm, index=close.index).ewm(alpha=1.0 / n, adjust=False).mean()
           / atr.replace(0, np.nan) * 100).fillna(0).values
    ndi = (pd.Series(ndm, index=close.index).ewm(alpha=1.0 / n, adjust=False).mean()
           / atr.replace(0, np.nan) * 100).fillna(0).values
    dx  = np.where((pdi + ndi) > 0, np.abs(pdi - ndi) / (pdi + ndi) * 100, 0.0)
    adx = pd.Series(dx, index=close.index).ewm(alpha=1.0 / n, adjust=False).mean().values
    return pdi, ndi, adx


def _rolling_vwap(hlc3: pd.Series, volume: pd.Series, n: int) -> np.ndarray:
    num = (hlc3 * volume).rolling(n).sum()
    den = volume.rolling(n).sum()
    return (num / den.replace(0, np.nan)).fillna(hlc3).values


def _weekly_ema(data: pd.DataFrame, n: int) -> np.ndarray:
    """EMA(n) on weekly closes, forward-filled to daily bars."""
    weekly = data["Close"].resample("W").last().dropna()
    wema   = weekly.ewm(span=n, adjust=False).mean()
    return wema.reindex(data.index, method="ffill").bfill().values


def _percentrank(arr: np.ndarray, n: int) -> np.ndarray:
    result = np.full(len(arr), 50.0)
    for i in range(n, len(arr)):
        w = arr[i - n:i]
        result[i] = float(np.sum(w < arr[i])) / n * 100.0
    return result


def _pivot_highs(high: np.ndarray, n: int) -> np.ndarray:
    """Strictly greater pivot high, confirmed n bars after the pivot bar."""
    result = np.full(len(high), np.nan)
    for i in range(2 * n, len(high)):
        p = i - n
        lo, hi = max(0, p - n), p + n + 1
        window = high[lo:hi]
        if high[p] > 0 and np.all(high[p] > np.concatenate([window[:p - lo], window[p - lo + 1:]])):
            result[i] = high[p]
    return result


def _pivot_lows(low: np.ndarray, n: int) -> np.ndarray:
    result = np.full(len(low), np.nan)
    for i in range(2 * n, len(low)):
        p = i - n
        lo, hi = max(0, p - n), p + n + 1
        window = low[lo:hi]
        if np.all(low[p] < np.concatenate([window[:p - lo], window[p - lo + 1:]])):
            result[i] = low[p]
    return result


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
    fast_len    = int(params["fast_len"])
    mid_len     = int(params["mid_len"])
    slow_len    = int(params["slow_len"])
    min_score   = float(params["min_score"])
    risk_atr    = float(params["risk_atr"])

    # Fixed params (not optimized)
    ATR_LEN       = 21
    PIVOT_LEN     = 5
    PRESSURE_LEN  = 24
    VOL_LEN       = 34
    AUCTION_LEN   = 90
    VALUE_DEV     = 1.15
    DISPLACE_MULT = 1.10
    REGIME_LEN    = 120
    TRANS_LEN     = 80
    MIN_CONT      = 58.0
    HTF_LEN       = 55   # weekly bars
    VWAP_LEN      = 20   # rolling daily bars
    RR_EFF        = 2.01  # weighted TP: 35%@1R + 35%@2R + 30%@3.2R

    close  = data["Close"]
    open_s = data["Open"]
    high_s = data["High"]
    low_s  = data["Low"]
    vol_s  = data["Volume"]
    hlc3   = (high_s + low_s + close) / 3.0

    # ── Indicators ────────────────────────────────────────────────────────────
    fast_a = _ema(close, fast_len)
    mid_a  = _ema(close, mid_len)
    slow_a = _ema(close, slow_len)
    atr_a  = _atr(high_s, low_s, close, ATR_LEN)
    rsi_a  = _rsi(close)
    mhist  = _macd_hist(close)
    pdi, ndi, adx = _dmi(high_s, low_s, close)
    rvwap  = _rolling_vwap(hlc3, vol_s, VWAP_LEN)
    htf_a  = _weekly_ema(data, HTF_LEN)

    spread_s = (high_s - low_s).clip(lower=1e-8)
    body_s   = close - open_s
    sb       = (body_s / spread_s * vol_s).values
    pr_raw   = pd.Series(sb).ewm(span=PRESSURE_LEN, adjust=False).mean().values
    pr_base  = pd.Series(np.abs(sb)).ewm(span=PRESSURE_LEN, adjust=False).mean().values
    pr_osc   = np.clip(np.where(pr_base > 1e-10, pr_raw / pr_base * 100.0, 0.0), -100, 100)
    vol_base = vol_s.rolling(VOL_LEN).mean().values
    vol_imp  = np.where(vol_base > 0, vol_s.values / vol_base, 1.0)

    wt_price  = ((hlc3 * vol_s).rolling(AUCTION_LEN).sum() /
                 vol_s.rolling(AUCTION_LEN).sum()).values
    auc_dev   = hlc3.rolling(AUCTION_LEN).std().values
    val_high  = wt_price + auc_dev * VALUE_DEV
    val_low   = wt_price - auc_dev * VALUE_DEV

    atr_pct  = np.where(close.values > 0, atr_a / close.values * 100.0, 0.0)
    atr_rank = _percentrank(atr_pct, REGIME_LEN)

    ph_a = _pivot_highs(high_s.values, PIVOT_LEN)
    pl_a = _pivot_lows(low_s.values, PIVOT_LEN)

    close_a = close.values
    open_a  = open_s.values
    high_a  = high_s.values
    low_a   = low_s.values
    n       = len(close_a)

    # ── State / Markov setup ──────────────────────────────────────────────────
    state = np.zeros(n, dtype=int)
    for i in range(n):
        if fast_a[i] > mid_a[i] and mid_a[i] > slow_a[i] and close_a[i] > mid_a[i]:
            state[i] = 1
        elif fast_a[i] < mid_a[i] and mid_a[i] < slow_a[i] and close_a[i] < mid_a[i]:
            state[i] = -1

    from_bull = np.where(np.roll(state, 1) == 1, 1.0, 0.0)
    from_bear = np.where(np.roll(state, 1) == -1, 1.0, 0.0)
    to_bull   = np.where(state == 1, 1.0, 0.0)
    to_bear   = np.where(state == -1, 1.0, 0.0)
    from_bull[0] = from_bear[0] = 0.0

    fb_sma   = pd.Series(from_bull).rolling(TRANS_LEN).mean().values
    fbtb_sma = pd.Series(from_bull * to_bull).rolling(TRANS_LEN).mean().values
    fd_sma   = pd.Series(from_bear).rolling(TRANS_LEN).mean().values
    fdtd_sma = pd.Series(from_bear * to_bear).rolling(TRANS_LEN).mean().values

    # ── Backtest loop ─────────────────────────────────────────────────────────
    live_start = max(trade_start_idx, slow_len + REGIME_LEN)
    port       = _Portfolio(initial_capital, commission, risk_pct)
    pending    = 0      # 1=enter long, -1=exit long (risk-off)
    sl_level   = 0.0
    tp_level   = 0.0

    equity_curve = np.full(n, float(initial_capital))
    trades: list[dict] = []

    last_ph = np.nan
    last_pl = np.nan
    structure_bias = 0

    for i in range(1, n):
        # ── 1. Execute pending at open ─────────────────────────────────────
        if i >= live_start:
            if pending == 1 and port.position == 0:
                port.enter_long(open_a[i])
                # SL: min(atr-based stop, last pivot low) clamped to atr*0.25 risk
                atr_stop = port.entry_price * (1.0 - risk_atr * atr_a[i] / close_a[i - 1])
                struct_stop = last_pl if not np.isnan(last_pl) else atr_stop
                stop = min(atr_stop, struct_stop)
                risk = max(abs(port.entry_price - stop), atr_a[i] * 0.25)
                sl_level = port.entry_price - risk
                tp_level = port.entry_price + risk * RR_EFF
                trades.append({"entry": port.entry_price, "notional": port.notional,
                               "bar_in": i})
            elif pending == -1 and port.position == 1:
                pnl = port.exit_long(open_a[i])
                if trades and "pnl" not in trades[-1]:
                    trades[-1].update({"exit": open_a[i], "pnl": pnl,
                                       "bar_out": i, "reason": "risk_off"})
                sl_level = tp_level = 0.0
            pending = 0

        # ── 2. Intraday SL / TP ────────────────────────────────────────────
        if port.position == 1 and i >= live_start:
            if low_a[i] <= sl_level:
                fill = open_a[i] if open_a[i] <= sl_level else sl_level
                pnl  = port.exit_long(fill)
                if trades and "pnl" not in trades[-1]:
                    trades[-1].update({"exit": fill, "pnl": pnl,
                                       "bar_out": i, "reason": "stop"})
                sl_level = tp_level = 0.0
            elif high_a[i] >= tp_level:
                pnl = port.exit_long(tp_level)
                if trades and "pnl" not in trades[-1]:
                    trades[-1].update({"exit": tp_level, "pnl": pnl,
                                       "bar_out": i, "reason": "tp"})
                sl_level = tp_level = 0.0

        # ── 3. Persistent structure state ─────────────────────────────────
        if not np.isnan(ph_a[i]):
            last_ph = ph_a[i]
        if not np.isnan(pl_a[i]):
            last_pl = pl_a[i]

        disp_i = (abs(close_a[i] - open_a[i]) > atr_a[i] * DISPLACE_MULT
                  and vol_imp[i] > 1.0)
        bos_up = (not np.isnan(last_ph) and close_a[i] > last_ph
                  and close_a[i - 1] <= last_ph and disp_i)
        bos_dn = (not np.isnan(last_pl) and close_a[i] < last_pl
                  and close_a[i - 1] >= last_pl and disp_i)
        if bos_up:
            structure_bias = 1
        elif bos_dn:
            structure_bias = -1

        # ── 4. Compute scores at bar close ────────────────────────────────
        if i >= live_start:
            # Regime components
            trend_bull  = (fast_a[i] > mid_a[i] and mid_a[i] > slow_a[i]
                           and close_a[i] > mid_a[i])
            trend_bear  = (fast_a[i] < mid_a[i] and mid_a[i] < slow_a[i]
                           and close_a[i] < mid_a[i])
            mom_bull    = mhist[i] > 0 and rsi_a[i] > 52 and pdi[i] > ndi[i]
            mom_bear    = mhist[i] < 0 and rsi_a[i] < 48 and ndi[i] > pdi[i]
            rvwap_bull  = close_a[i] > rvwap[i] and rvwap[i] > rvwap[i - 1]
            rvwap_bear  = close_a[i] < rvwap[i] and rvwap[i] < rvwap[i - 1]

            expansion   = atr_rank[i] > 62.0
            htf_bull    = close_a[i] > htf_a[i]
            htf_bear    = close_a[i] < htf_a[i]

            # Markov continuation
            cont_bull = (fbtb_sma[i] / fb_sma[i] if fb_sma[i] > 1e-8 else 0.5)
            cont_bear = (fdtd_sma[i] / fd_sma[i] if fd_sma[i] > 1e-8 else 0.5)
            cont_pct  = (cont_bull if state[i] == 1 else
                         cont_bear if state[i] == -1 else 0.33) * 100.0
            cont_pct  = float(np.clip(cont_pct, 0, 100))
            prob_pass = cont_pct >= MIN_CONT or expansion

            # Pressure components
            bid_pres  = pr_osc[i] > 12.0 and vol_imp[i] > 1.05
            ask_pres  = pr_osc[i] < -12.0 and vol_imp[i] > 1.05
            abs_bull  = (low_a[i] < low_a[i - 1] and close_a[i] > open_a[i]
                         and pr_osc[i] > pr_osc[i - 1] and vol_imp[i] > 1.1)
            abs_bear  = (high_a[i] > high_a[i - 1] and close_a[i] < open_a[i]
                         and pr_osc[i] < pr_osc[i - 1] and vol_imp[i] > 1.1)

            # Auction components
            spread_i   = max(high_a[i] - low_a[i], 1e-8)
            auc_disc   = (close_a[i] < val_low[i] and
                          close_a[i] > low_a[i] + spread_i * 0.45)
            auc_prem   = (close_a[i] > val_high[i] and
                          close_a[i] < high_a[i] - spread_i * 0.45)
            auc_recl   = (not np.isnan(val_low[i]) and not np.isnan(val_low[i - 1])
                          and close_a[i] > val_low[i]
                          and close_a[i - 1] <= val_low[i - 1])

            # Structure components
            fvg_bull = i >= 2 and low_a[i] > high_a[i - 2] and close_a[i] > open_a[i]
            fvg_bear = i >= 2 and high_a[i] < low_a[i - 2] and close_a[i] < open_a[i]
            swp_low  = (not np.isnan(last_pl) and low_a[i] < last_pl
                        and close_a[i] > last_pl)
            swp_high = (not np.isnan(last_ph) and high_a[i] > last_ph
                        and close_a[i] < last_ph)
            struct_bull = structure_bias == 1 or swp_low or fvg_bull
            struct_bear = structure_bias == -1 or swp_high or fvg_bear

            # Execution scores
            adx_bonus = float(np.clip(adx[i] / 8.0, 0.0, 4.0))
            long_score = (16.0 * trend_bull + 12.0 * mom_bull + 9.0 * rvwap_bull
                          + 12.0 * bid_pres + 8.0 * abs_bull + 14.0 * struct_bull
                          + 10.0 * (auc_disc or auc_recl)
                          + 8.0 * htf_bull + 7.0 * prob_pass + adx_bonus)
            short_score = (16.0 * trend_bear + 12.0 * mom_bear + 9.0 * rvwap_bear
                           + 12.0 * ask_pres + 8.0 * abs_bear + 14.0 * struct_bear
                           + 10.0 * auc_prem
                           + 8.0 * htf_bear + 7.0 * prob_pass + adx_bonus)
            long_score  = float(np.clip(long_score,  0, 100))
            short_score = float(np.clip(short_score, 0, 100))

            # Entry signal
            if port.position == 0 and pending == 0:
                if (long_score >= min_score and long_score > short_score + 8.0
                        and htf_bull and not auc_prem):
                    pending = 1

            # Risk-off exit
            if port.position == 1 and pending == 0:
                regime_flip = state[i] == -1
                press_off   = pr_osc[i] < -20.0
                score_flip  = short_score > long_score + 6.0
                if regime_flip or press_off or score_flip:
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
