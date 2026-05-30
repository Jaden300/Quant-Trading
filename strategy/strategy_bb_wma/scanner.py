"""
scanner.py — BB WMA signal scanner across the S&P 500.

Downloads ~1 year of data, runs the Bollinger WMA logic with optimised
parameters, and reports which stocks are currently in an active long
position — ranked by confidence.

Usage:
    python strategy_bb_wma/scanner.py
    python strategy_bb_wma/scanner.py --tickers NVDA AAPL MSFT AVGO
    python strategy_bb_wma/scanner.py --top 30
"""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import yfinance as yf

# ── Optimised params ──────────────────────────────────────────────────────────
PARAMS = {
    "bb_length": 150,
    "std_buy":   1.0,
    "std_sell":  3.0,
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


def _wma(series: pd.Series, length: int) -> np.ndarray:
    weights = np.arange(1, length + 1, dtype=float)
    return series.rolling(length).apply(
        lambda x: np.dot(x, weights) / weights.sum(), raw=True
    ).values


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
            if len(df) >= PARAMS["bb_length"] + 10:
                result[ticker] = df
        except Exception:
            pass
    return result


def scan_ticker(ticker: str, df: pd.DataFrame) -> dict | None:
    try:
        bb_length = PARAMS["bb_length"]
        std_buy   = PARAMS["std_buy"]
        std_sell  = PARAMS["std_sell"]

        high_a  = df["High"].values
        low_a   = df["Low"].values
        close_a = df["Close"].values
        open_a  = df["Open"].values
        hlc3    = (high_a + low_a + close_a) / 3.0
        src     = pd.Series(hlc3)

        basis_a = _wma(src, bb_length)
        dev_a   = src.rolling(bb_length).std(ddof=0).values
        upper_a = basis_a + std_buy  * dev_a
        lower_a = basis_a - std_sell * dev_a

        n             = len(close_a)
        in_pos        = False
        entry_bar     = -1
        entry_price   = 0.0
        last_exit_bar = -999
        pending_entry = False
        pending_exit  = False

        for i in range(1, n):
            if np.isnan(upper_a[i]) or np.isnan(lower_a[i]):
                continue

            if pending_exit and in_pos:
                in_pos        = False
                last_exit_bar = i
                pending_exit  = False

            if pending_entry and not in_pos:
                in_pos      = True
                entry_bar   = i
                entry_price = open_a[i]
                pending_entry = False

            if in_pos and not pending_exit:
                if close_a[i] < lower_a[i]:
                    pending_exit = True

            if not in_pos and not pending_entry and not pending_exit:
                if i > last_exit_bar and hlc3[i] > upper_a[i]:
                    close_gt_close2 = (i >= 2) and (close_a[i] > close_a[i - 2])
                    close_gt_open   = close_a[i] > open_a[i]
                    body   = abs(close_a[i] - open_a[i])
                    shadow = (high_a[i] - low_a[i]) - body
                    if (close_gt_close2 or close_gt_open) and not (body > 0 and shadow > 10.0 * body):
                        pending_entry = True

        if not in_pos or entry_bar < 0:
            return None

        bars_ago = (n - 1) - entry_bar
        current  = close_a[-1]
        ret_pct  = (current / entry_price - 1.0) * 100.0

        lower_now = lower_a[-1] if not np.isnan(lower_a[-1]) else 0.0
        dev_now   = dev_a[-1]   if not np.isnan(dev_a[-1])   else 1.0
        dist_to_exit = max(0.0, min((current - lower_now) / max(dev_now, 1e-9), 10.0)) / 10.0 * 100.0
        confidence   = round(dist_to_exit)

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
        ret = f"+{s['ret_pct']:.1f}%" if s["ret_pct"] >= 0 else f"{s['ret_pct']:.1f}%"
        return (f"  {tag}   {s['ticker']:<6}  conf {s['confidence']:>3}/100  "
                f"  {s['bars_ago']:>4}d in trade   {ret:>8} since entry")

    print(f"\n{'─' * 80}")
    print(f"  BB WMA — SIGNAL SCAN   {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"  Params: WMA({PARAMS['bb_length']})  "
          f"upper×{PARAMS['std_buy']}  lower×{PARAMS['std_sell']}")
    print(f"  Confidence = distance above lower exit band (100 = far from exit)")
    print(f"{'─' * 80}")

    if fresh:
        print(f"\n  FRESH SIGNALS  (entered within last {FRESH_BARS} days)\n")
        for s in fresh[:top]:
            print(_row(s))
    else:
        print(f"\n  No fresh signals right now.\n")

    if active:
        print(f"\n  ACTIVE POSITIONS  (already in trade)\n")
        for s in active[:top]:
            print(_row(s))

    total = len(fresh) + len(active)
    print(f"\n{'─' * 80}")
    print(f"  {total} active  |  {len(fresh)} fresh  |  {len(active)} ongoing\n")


def main() -> None:
    p = argparse.ArgumentParser(description="BB WMA — S&P 500 signal scanner")
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
