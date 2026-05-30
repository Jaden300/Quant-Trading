"""
scanner.py — EMA Trail signal scanner across the S&P 500.

Downloads ~1 year of data for all S&P 500 stocks, runs the EMA Trail
logic with the optimised parameters, and reports which stocks just had
a bullish EMA crossover — ranked by confidence (trail iterations).

Usage:
    python strategy_ema_trail/scanner.py
    python strategy_ema_trail/scanner.py --tickers NVDA AAPL MSFT AVGO
    python strategy_ema_trail/scanner.py --top 30
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

# ── Optimised params ──────────────────────────────────────────────────────────
PARAMS = {
    "fast_len":          10,
    "slow_len":          50,
    "tp_pct":            0.30,
    "sl_pct":            0.10,
    "trail_trigger_pct": 0.06,
    "trail_step_pct":    0.04,
}
FRESH_BARS = 5   # crossover within this many bars = "fresh"


def get_sp500_tickers() -> list[str]:
    return [
        "MMM","AOS","ABT","ABBV","ACN","ADBE","AMD","AES","AFL","A","APD","ABNB",
        "AKAM","ALB","ARE","ALGN","ALLE","LNT","ALL","GOOGL","GOOG","MO","AMZN",
        "AMCR","AEE","AAL","AEP","AXP","AIG","AMT","AWK","AMP","AME","AMGN",
        "APH","ADI","ANSS","AON","APA","APO","AAPL","AMAT","APTV","ACGL","ADM",
        "ANET","AJG","AIZ","T","ATO","ADSK","ADP","AZO","AVB","AVY","AXON","BKR",
        "BALL","BAC","BAX","BDX","BRK-B","BBY","TECH","BIIB","BLK","BX","BA",
        "BCH","BMY","AVGO","BR","BRO","BF-B","BLDR","BG","CDNS","CZR","CPT",
        "CPB","COF","CAH","KMX","CCL","CARR","CTLT","CAT","CBOE","CBRE","CDW",
        "CE","COR","CNC","CNX","CDAY","CF","CRL","SCHW","CHTR","CVX","CMG","CB",
        "CHD","CI","CINF","CTAS","CSCO","C","CFG","CLX","CME","CMS","KO","CTSH",
        "CL","CMCSA","CMA","CAG","COP","ED","STZ","CEG","COO","CPRT","GLW","CPAY",
        "CTVA","CSGP","COST","CTRA","CCI","CSX","CMI","CVS","DHR","DRI","DVA",
        "DAY","DECK","DE","DAL","DVN","DXCM","FANG","DLR","DFS","DG","DLTR","D",
        "DPZ","DOV","DOW","DHI","DTE","DUK","DD","EMN","ETN","EBAY","ECL","EIX",
        "EW","EA","ELV","EMR","ENPH","ETR","EOG","EPAM","EQT","EFX","EQIX","EQR",
        "ESS","EL","ETSY","EG","EVRG","ES","EXC","EXPE","EXPD","EXR","XOM","FFIV",
        "FDS","FICO","FAST","FRT","FDX","FIS","FITB","FSLR","FE","FI","FMC","F",
        "FTNT","FTV","FOXA","FOX","BEN","FCX","GRMN","IT","GE","GEHC","GEV",
        "GEN","GNRC","GD","GIS","GM","GPC","GILD","GS","HAL","HIG","HAS","HCA",
        "DOC","HSIC","HSY","HES","HPE","HLT","HOLX","HD","HON","HRL","HST","HWM",
        "HPQ","HUBB","HUM","HBAN","HII","IBM","IEX","IDXX","ITW","INCY","IR",
        "PODD","INTC","ICE","IFF","IP","IPG","INTU","ISRG","IVZ","INVH","IQV",
        "IRM","JBHT","JBL","JKHY","J","JNJ","JCI","JPM","JNPR","K","KVUE","KDP",
        "KEY","KEYS","KMB","KIM","KMI","KLAC","KHC","KR","LHX","LH","LRCX","LW",
        "LVS","LDOS","LEN","LLY","LIN","LYV","LKQ","LMT","L","LOW","LULU","LYB",
        "MTB","MRO","MPC","MKTX","MAR","MMC","MLM","MAS","MA","MTCH","MKC","MCD",
        "MCK","MDT","MRK","META","MET","MTD","MGM","MCHP","MU","MSFT","MAA","MRNA",
        "MHK","MOH","TAP","MDLZ","MPWR","MNST","MCO","MS","MOS","MSI","MSCI","NDAQ",
        "NTAP","NOC","NFLX","NEM","NWSA","NWS","NEE","NKE","NI","NDSN","NSC","NTRS",
        "NOC","NRG","NUE","NVDA","NVR","NXPI","ORLY","OXY","ODFL","OMC","ON","OKE",
        "ORCL","OTIS","OC","PCAR","PKG","PANW","PH","PAYX","PAYC","PYPL","PNR","PEP",
        "PFE","PCG","PM","PSX","PNW","PXD","PNC","POOL","PPG","PPL","PFG","PG","PGR",
        "PRU","PEG","PTC","PSA","PHM","QRVO","PWR","QCOM","RL","RJF","RTX","O","REG",
        "REGN","RF","RSG","RMD","RVTY","ROK","ROL","ROP","ROST","RCL","SPGI","CRM",
        "SBAC","SLB","STX","SRE","NOW","SHW","SPG","SWKS","SJM","SNA","SOLV","SO",
        "LUV","SWK","SBUX","STT","STLD","STE","SYK","SMCI","SYF","SNPS","SYY","TMUS",
        "TROW","TTWO","TPR","TRGP","TGT","TEL","TDY","TFX","TER","TSLA","TXN","TXT",
        "TMO","TJX","TSCO","TT","TDG","TRV","TRMB","TFC","TYL","TSN","USB","UDR",
        "ULTA","UNP","UAL","UPS","URI","UNH","UHS","VLO","VTR","VLTO","VRSN","VRSK",
        "VZ","VRTX","VTRS","VICI","V","VMC","WRB","GWW","WAB","WBA","WMT","WBD",
        "WM","WAT","WEC","WFC","WELL","WST","WDC","WRK","WY","WHR","WMB","WTW",
        "WYNN","XEL","XYL","YUM","ZBRA","ZBH","ZTS",
    ]


def download_batch(tickers: list[str], start: str, end: str) -> dict[str, pd.DataFrame]:
    print(f"  Downloading {len(tickers)} tickers [{start} → {end}] …")
    raw = yf.download(
        tickers, start=start, end=end,
        interval="1d", auto_adjust=True,
        progress=False, group_by="ticker",
    )
    result: dict[str, pd.DataFrame] = {}
    for ticker in tickers:
        try:
            df = raw if len(tickers) == 1 else raw[ticker]
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
            if len(df) >= PARAMS["slow_len"] + 10:
                result[ticker] = df
        except Exception:
            pass
    return result


def scan_ticker(ticker: str, df: pd.DataFrame) -> dict | None:
    """
    Returns signal info if currently in an active long (fast EMA > slow EMA
    and position not yet stopped/targeted out), else None.
    """
    try:
        fast_len          = PARAMS["fast_len"]
        slow_len          = PARAMS["slow_len"]
        tp_pct            = PARAMS["tp_pct"]
        sl_pct            = PARAMS["sl_pct"]
        trail_trigger_pct = PARAMS["trail_trigger_pct"]
        trail_step_pct    = PARAMS["trail_step_pct"]

        close_a  = df["Close"].values
        fast_ema = df["Close"].ewm(span=fast_len, adjust=False).mean().values
        slow_ema = df["Close"].ewm(span=slow_len, adjust=False).mean().values
        n        = len(close_a)

        in_pos     = False
        tp_level   = 0.0
        sl_level   = 0.0
        ref_price  = 0.0
        trail_iter = 0
        entry_bar  = -1
        entry_price = 0.0

        for i in range(1, n):
            if np.isnan(fast_ema[i]) or np.isnan(slow_ema[i]):
                continue

            if in_pos:
                # update trailing
                if ref_price > 0 and close_a[i] >= ref_price * (1.0 + trail_trigger_pct):
                    tp_level  *= (1.0 + trail_step_pct)
                    sl_level  *= (1.0 + trail_step_pct)
                    ref_price *= (1.0 + trail_trigger_pct)
                    trail_iter += 1

                # check exit
                if close_a[i] <= sl_level or close_a[i] >= tp_level:
                    in_pos = False

            if not in_pos:
                crossover = fast_ema[i - 1] < slow_ema[i - 1] and fast_ema[i] >= slow_ema[i]
                if crossover:
                    in_pos      = True
                    entry_bar   = i
                    entry_price = close_a[i]
                    tp_level    = entry_price * (1.0 + tp_pct)
                    sl_level    = entry_price * (1.0 - sl_pct)
                    ref_price   = entry_price
                    trail_iter  = 0

        if not in_pos or entry_bar < 0:
            return None

        bars_ago   = (n - 1) - entry_bar
        current    = close_a[-1]
        ret_pct    = (current / entry_price - 1.0) * 100.0
        confidence = min(trail_iter * 25, 100)

        return {
            "ticker":     ticker,
            "confidence": confidence,
            "trail_iter": trail_iter,
            "bars_ago":   bars_ago,
            "ret_pct":    round(ret_pct, 2),
            "fresh":      bars_ago <= FRESH_BARS,
        }
    except Exception:
        return None


def print_results(signals: list[dict], top: int) -> None:
    fresh  = [s for s in signals if s["fresh"]]
    active = [s for s in signals if not s["fresh"]]

    def _row(s: dict) -> str:
        tag   = "★ FRESH" if s["fresh"] else "  active"
        ret   = f"+{s['ret_pct']:.1f}%" if s["ret_pct"] >= 0 else f"{s['ret_pct']:.1f}%"
        trail = f"{s['trail_iter']} trail{'s' if s['trail_iter'] != 1 else ''}"
        return (f"  {tag}   {s['ticker']:<6}  conf {s['confidence']:>3}/100  "
                f"  {s['bars_ago']:>3}d in trade   {ret:>8} since entry   ({trail})")

    print(f"\n{'─' * 80}")
    print(f"  EMA TRAIL — SIGNAL SCAN   {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"  Params: EMA {PARAMS['fast_len']}/{PARAMS['slow_len']}  "
          f"TP {int(PARAMS['tp_pct']*100)}%  SL {int(PARAMS['sl_pct']*100)}%  "
          f"Trail {int(PARAMS['trail_trigger_pct']*100)}%/{int(PARAMS['trail_step_pct']*100)}%")
    print(f"{'─' * 80}")

    if fresh:
        print(f"\n  FRESH SIGNALS  (EMA crossover within last {FRESH_BARS} days)\n")
        for s in fresh[:top]:
            print(_row(s))
    else:
        print(f"\n  No fresh signals right now.\n")

    if active:
        print(f"\n  ACTIVE TRADES  (crossover already happened, still in position)\n")
        for s in active[:top]:
            print(_row(s))

    total = len(fresh) + len(active)
    print(f"\n{'─' * 80}")
    print(f"  {total} active  |  {len(fresh)} fresh  |  {len(active)} ongoing\n")


def main() -> None:
    p = argparse.ArgumentParser(description="EMA Trail — S&P 500 signal scanner")
    p.add_argument("--tickers", nargs="+", default=None,
                   help="Specific tickers to scan (default: full S&P 500)")
    p.add_argument("--top", default=20, type=int,
                   help="Max results to show per section")
    args = p.parse_args()

    if args.tickers:
        tickers = [t.upper() for t in args.tickers]
        print(f"Scanning {len(tickers)} specified tickers …")
    else:
        print("Fetching S&P 500 ticker list …")
        tickers = get_sp500_tickers()
        print(f"  {len(tickers)} tickers found.")

    end   = datetime.today().strftime("%Y-%m-%d")
    start = (datetime.today() - timedelta(days=365)).strftime("%Y-%m-%d")

    all_data = download_batch(tickers, start, end)
    print(f"  {len(all_data)} tickers downloaded successfully.\n")
    print("Scanning …")

    signals = []
    for ticker, df in all_data.items():
        result = scan_ticker(ticker, df)
        if result:
            signals.append(result)

    signals.sort(key=lambda s: (-s["fresh"], -s["confidence"]))
    print_results(signals, top=args.top)


if __name__ == "__main__":
    main()
