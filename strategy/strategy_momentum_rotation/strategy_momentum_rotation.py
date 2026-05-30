"""
strategy_momentum_rotation.py — Cross-asset momentum rotation backtest engine.

HOW IT WORKS:
  Every rebal_every bars, rank all assets by their lookback-period price return.
  Hold the top top_n assets, equally weighted from available cash.
  Only sell a holding when its rank falls to top_n + hold_buffer or worse
  (the buffer avoids rotating on every tiny rank shuffle, reducing fee drag).

Execution:
  Signal fires at bar close → trades execute at NEXT bar's open (one-bar delay).
  Sells always execute before buys so the cash is available immediately.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


class _Portfolio:
    def __init__(self, initial_capital: float, commission: float):
        self.cash     = initial_capital
        self._comm    = commission
        self.holdings: dict[int, dict] = {}   # idx → {notional, entry_price, bar_in}

    def enter(self, idx: int, price: float, notional: float, bar_in: int) -> None:
        fill = price * (1.0 + self._comm)
        self.cash -= notional
        self.holdings[idx] = {"notional": notional, "entry_price": fill, "bar_in": bar_in}

    def exit(self, idx: int, price: float, bar_out: int) -> dict:
        pos  = self.holdings.pop(idx)
        fill = price * (1.0 - self._comm)
        pnl_pct  = (fill - pos["entry_price"]) / pos["entry_price"]
        proceeds = pos["notional"] * (1.0 + pnl_pct)
        self.cash += proceeds
        return {"pnl": proceeds - pos["notional"],
                "bar_in": pos["bar_in"], "bar_out": bar_out}

    def mtm(self, closes: np.ndarray) -> float:
        total = self.cash
        for idx, pos in self.holdings.items():
            p = closes[idx]
            if not np.isnan(p):
                pnl_pct = (p - pos["entry_price"]) / pos["entry_price"]
                total  += pos["notional"] * (1.0 + pnl_pct)
        return total


def run_backtest(
    all_data: dict[str, pd.DataFrame],
    params: dict,
    initial_capital: float = 1500.0,
    commission: float = 0.015,
    trade_start_idx: int = 0,
    return_equity_curve: bool = False,
) -> dict:
    lookback    = int(params["lookback"])
    top_n       = int(params["top_n"])
    hold_buffer = int(params["hold_buffer"])
    rebal_every = int(params["rebal_every"])

    tickers   = list(all_data.keys())
    n_tickers = len(tickers)
    ref_index = all_data[tickers[0]].index
    n         = len(ref_index)

    closes = np.column_stack([all_data[t]["Close"].reindex(ref_index).values for t in tickers])
    opens  = np.column_stack([all_data[t]["Open"].reindex(ref_index).values  for t in tickers])

    live_start = max(trade_start_idx, lookback + 1)

    port          = _Portfolio(initial_capital, commission)
    equity_curve  = np.full(n, float(initial_capital))
    trades:       list[dict] = []
    pending_sells: list[int] = []
    pending_buys:  list[int] = []

    for i in range(1, n):

        # ── 1. Execute pending trades at this bar's open ──────────────────
        if pending_sells or pending_buys:

            for idx in list(pending_sells):
                if idx in port.holdings:
                    trades.append(port.exit(idx, opens[i, idx], bar_out=i))
            pending_sells.clear()

            if pending_buys and port.cash > 0:
                per_slot = port.cash / len(pending_buys)
                for idx in pending_buys:
                    if idx not in port.holdings and per_slot > 0.01:
                        port.enter(idx, opens[i, idx], per_slot, bar_in=i)
            pending_buys.clear()

        # ── 2. Generate rebalance signal at close ─────────────────────────
        if i >= live_start and (i - live_start) % rebal_every == 0:
            cur   = closes[i]
            old   = closes[i - lookback]
            valid = ~(np.isnan(cur) | np.isnan(old) | (old == 0))
            mom   = np.where(valid, cur / old - 1.0, -np.inf)

            ranked  = np.argsort(-mom)           # index 0 = best stock
            rank_of = np.empty(n_tickers, dtype=int)
            rank_of[ranked] = np.arange(n_tickers)

            # Sell anything that fell out of the tolerance zone
            to_sell = [idx for idx in port.holdings if rank_of[idx] >= top_n + hold_buffer]
            pending_sells.extend(to_sell)

            # Buy enough top stocks to fill up to top_n slots
            staying    = {idx for idx in port.holdings if idx not in to_sell}
            slots_free = top_n - len(staying)
            for idx in ranked:
                if slots_free <= 0:
                    break
                if idx not in staying:
                    pending_buys.append(idx)
                    slots_free -= 1

        equity_curve[i] = port.mtm(closes[i])

    # Force-close all remaining positions at last close
    for idx in list(port.holdings.keys()):
        trades.append(port.exit(idx, closes[-1, idx], bar_out=n - 1))
    equity_curve[-1] = port.cash

    live_eq = equity_curve[live_start:]
    metrics = _calc_metrics(live_eq, trades, initial_capital)
    if return_equity_curve:
        metrics["equity_curve"] = live_eq
    return metrics


def _calc_metrics(equity: np.ndarray, trades: list[dict], initial_capital: float) -> dict:
    if len(equity) < 2:
        return {"total_return": 0.0, "sharpe_ratio": 0.0, "max_drawdown": 0.0,
                "win_rate": 0.0, "num_trades": 0, "profit_factor": 0.0,
                "mean_hold_bars": 0.0, "final_equity": float(equity[-1]) if len(equity) else initial_capital}

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
