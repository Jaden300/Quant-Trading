"""
scanner.py — Elektro BB signal scanner across the S&P 500.

Downloads ~2 years of data (needs 350-bar BB warmup), runs the Bollinger
Band + RSI mean-reversion logic, and reports which stocks are currently
in an active long position — ranked by confidence (RSI recovery %).

Usage:
    python strategy_elektro/scanner.py
    python strategy_elektro/scanner.py --tickers NVDA AAPL MSFT AVGO
    python strategy_elektro/scanner.py --top 30
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
from strategy_elektro import calculate_bands

# ── Optimised params ──────────────────────────────────────────────────────────
PARAMS = {
    "bb_length":    350,
    "bb_mult":      1.5,
    "rsi_length":   35,
    "rsi_oversold": 40.0,
    "rsi_exit":     80.0,
}
FRESH_BARS = 5


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


def _rsi(close: np.ndarray, length: int) -> np.ndarray:
    delta = np.diff(close.astype(float))
    gain  = np.where(delta > 0, delta, 0.0)
    loss  = np.where(delta < 0, -delta, 0.0)
    avg_g = pd.Series(gain).ewm(alpha=1.0 / length, adjust=False).mean().values
    avg_l = pd.Series(loss).ewm(alpha=1.0 / length, adjust=False).mean().values
    rs    = avg_g / np.where(avg_l > 0, avg_l, 1e-9)
    return np.concatenate([[np.nan], 100.0 - 100.0 / (1.0 + rs)])


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
            if len(df) >= PARAMS["bb_length"] + 50:
                result[ticker] = df
        except Exception:
            pass
    return result


def scan_ticker(ticker: str, df: pd.DataFrame) -> dict | None:
    try:
        bb_length    = PARAMS["bb_length"]
        rsi_oversold = PARAMS["rsi_oversold"]
        rsi_exit     = PARAMS["rsi_exit"]

        close_s = df["Close"]
        _, upper, lower = calculate_bands(df, bb_length, PARAMS["bb_mult"])
        rsi_a   = _rsi(close_s.values, PARAMS["rsi_length"])
        close_a = close_s.values
        up_a    = upper.values
        lo_a    = lower.values
        n       = len(close_a)

        in_pos      = False
        entry_bar   = -1
        entry_price = 0.0

        for i in range(1, n):
            if np.isnan(up_a[i]) or np.isnan(lo_a[i]) or np.isnan(rsi_a[i]):
                continue
            if not in_pos:
                if close_a[i] <= lo_a[i] and rsi_a[i] <= rsi_oversold:
                    in_pos      = True
                    entry_bar   = i
                    entry_price = close_a[i]
            else:
                if close_a[i] >= up_a[i] and rsi_a[i] >= rsi_exit:
                    in_pos = False

        if not in_pos or entry_bar < 0:
            return None

        bars_ago   = (n - 1) - entry_bar
        current    = close_a[-1]
        ret_pct    = (current / entry_price - 1.0) * 100.0
        rsi_now    = rsi_a[-1] if not np.isnan(rsi_a[-1]) else 0.0
        confidence = round(max(0.0, min((rsi_now - rsi_oversold) / max(rsi_exit - rsi_oversold, 1.0) * 100.0, 100.0)))

        return {
            "ticker":     ticker,
            "confidence": confidence,
            "bars_ago":   bars_ago,
            "ret_pct":    round(ret_pct, 2),
            "rsi":        round(rsi_now, 1),
            "fresh":      bars_ago <= FRESH_BARS,
        }
    except Exception:
        return None


def print_results(signals: list[dict], top: int) -> None:
    fresh  = [s for s in signals if s["fresh"]]
    active = [s for s in signals if not s["fresh"]]

    def _row(s: dict) -> str:
        tag = "★ FRESH" if s["fresh"] else "  active"
        ret = f"+{s['ret_pct']:.1f}%" if s["ret_pct"] >= 0 else f"{s['ret_pct']:.1f}%"
        return (f"  {tag}   {s['ticker']:<6}  conf {s['confidence']:>3}/100  "
                f"  {s['bars_ago']:>4}d in trade   {ret:>8} since entry   RSI {s['rsi']:.1f}")

    print(f"\n{'─' * 80}")
    print(f"  ELEKTRO BB — SIGNAL SCAN   {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"  Params: BB({PARAMS['bb_length']}, {PARAMS['bb_mult']}×)  "
          f"RSI({PARAMS['rsi_length']})  entry≤{PARAMS['rsi_oversold']}  exit≥{PARAMS['rsi_exit']}")
    print(f"{'─' * 80}")

    if fresh:
        print(f"\n  FRESH SIGNALS  (entered within last {FRESH_BARS} days)\n")
        for s in fresh[:top]:
            print(_row(s))
    else:
        print(f"\n  No fresh signals right now.\n")

    if active:
        print(f"\n  ACTIVE POSITIONS  (already in trade, holding for exit)\n")
        for s in active[:top]:
            print(_row(s))

    total = len(fresh) + len(active)
    print(f"\n{'─' * 80}")
    print(f"  {total} active  |  {len(fresh)} fresh  |  {len(active)} ongoing\n")


def main() -> None:
    p = argparse.ArgumentParser(description="Elektro BB — S&P 500 signal scanner")
    p.add_argument("--tickers", nargs="+", default=None)
    p.add_argument("--top",     default=20, type=int)
    args = p.parse_args()

    if args.tickers:
        tickers = [t.upper() for t in args.tickers]
        print(f"Scanning {len(tickers)} specified tickers …")
    else:
        print("Fetching S&P 500 ticker list …")
        tickers = get_sp500_tickers()
        print(f"  {len(tickers)} tickers found.")

    end   = datetime.today().strftime("%Y-%m-%d")
    start = (datetime.today() - timedelta(days=730)).strftime("%Y-%m-%d")  # 2yr for 350-bar warmup

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
