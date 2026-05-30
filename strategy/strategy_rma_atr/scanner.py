"""
scanner.py — RMA ATR Bands signal scanner across the S&P 500.

Downloads ~6 months of data for all S&P 500 stocks, runs the RMA ATR
bands logic with the optimised parameters, and reports which stocks are
currently in a bullish trend — ranked by confidence.

Usage:
    python strategy_rma_atr/scanner.py
    python strategy_rma_atr/scanner.py --tickers NVDA AAPL MSFT AVGO
    python strategy_rma_atr/scanner.py --top 30
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

sys.path.insert(0, str(Path(__file__).parent))
from strategy_rma_atr import calculate_bands

# ── Optimised params (from test_largecap result) ──────────────────────────────
PARAMS = {
    "ma_src":     "close",
    "ma_length":  8,
    "atr_length": 20,
    "upper_mult": 1.50,
    "lower_mult": 2.50,
}
FRESH_BARS = 3   # signal fired within this many bars = "fresh"


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
            if len(tickers) == 1:
                df = raw
            else:
                df = raw[ticker]
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
            if len(df) >= 60:   # need enough bars for indicators
                result[ticker] = df
        except Exception:
            pass
    return result


def scan_ticker(ticker: str, df: pd.DataFrame) -> dict | None:
    """Returns signal info if bullish trend active, else None."""
    try:
        ma, upper, lower = calculate_bands(
            df,
            ma_length  = PARAMS["ma_length"],
            atr_length = PARAMS["atr_length"],
            upper_mult = PARAMS["upper_mult"],
            lower_mult = PARAMS["lower_mult"],
            ma_src     = PARAMS["ma_src"],
        )
        close  = df["Close"].values
        up_a   = upper.values
        lo_a   = lower.values
        ma_a   = ma.values
        atr_a  = (upper - ma).values / PARAMS["upper_mult"]   # recover ATR from bands

        n = len(close)
        trend = 0
        signal_bar = -1   # bar index when trend last flipped to 1

        for i in range(1, n):
            if np.isnan(up_a[i]):
                continue
            prev = trend
            if close[i] > up_a[i]:
                trend = 1
            elif close[i] < lo_a[i]:
                trend = -1
            if trend == 1 and prev != 1:
                signal_bar = i

        if trend != 1 or signal_bar < 0:
            return None   # not currently bullish

        bars_ago    = (n - 1) - signal_bar
        entry_price = close[signal_bar]
        current     = close[-1]
        ret_pct     = (current / entry_price - 1.0) * 100.0

        # Confidence: breakout strength + ATR percentile rank
        breakout_str = min((close[-1] - up_a[-1]) / max(atr_a[-1], 1e-9), 3.0) / 3.0 * 100
        atr_window   = atr_a[max(0, n - 63):n]
        atr_rank     = float(np.sum(atr_window < atr_a[-1]) / len(atr_window) * 100)
        confidence   = round((breakout_str + atr_rank) / 2)

        return {
            "ticker":     ticker,
            "confidence": confidence,
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
        tag = "★ FRESH" if s["fresh"] else "  active"
        ret = f"+{s['ret_pct']:.1f}%" if s['ret_pct'] >= 0 else f"{s['ret_pct']:.1f}%"
        return (f"  {tag}   {s['ticker']:<6}  conf {s['confidence']:>3}/100  "
                f"  {s['bars_ago']:>3}d in trend   {ret:>8} since entry")

    print(f"\n{'─' * 72}")
    print(f"  RMA ATR BANDS — SIGNAL SCAN   {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"  Params: MA={PARAMS['ma_length']} ATR={PARAMS['atr_length']} "
          f"upper×{PARAMS['upper_mult']} lower×{PARAMS['lower_mult']}")
    print(f"{'─' * 72}")

    if fresh:
        print(f"\n  FRESH SIGNALS  (flipped bullish within last {FRESH_BARS} days)\n")
        for s in fresh[:top]:
            print(_row(s))
    else:
        print(f"\n  No fresh signals right now.\n")

    if active:
        print(f"\n  ACTIVE TRENDS  (already in bullish trend, missed entry window)\n")
        for s in active[:top]:
            print(_row(s))

    total = len(fresh) + len(active)
    print(f"\n{'─' * 72}")
    print(f"  {total} bullish  |  {len(fresh)} fresh  |  {len(active)} active\n")


def main() -> None:
    p = argparse.ArgumentParser(description="RMA ATR Bands — S&P 500 signal scanner")
    p.add_argument("--tickers", nargs="+", default=None,
                   help="Specific tickers to scan (default: full S&P 500)")
    p.add_argument("--top",     default=20, type=int,
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
