"""
scanner.py — DCA Long signal scanner across the S&P 500.

Since DCA is nonstop (always in the market), this scanner reports the
current deal state for every ticker: SOs filled, % progress to TP,
and whether the base order was opened recently (fresh entry).

Usage:
    python strategy_dca/scanner.py
    python strategy_dca/scanner.py --tickers NVDA AAPL MSFT AVGO
    python strategy_dca/scanner.py --top 30
"""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import yfinance as yf

# ── Optimised params ──────────────────────────────────────────────────────────
PARAMS = {
    "tp_pct":  0.30,
    "so_step": 0.18,
    "max_so":  5,
}
_SO_RATIO  = 0.75
_VOL_COEF  = 1.05
FRESH_BARS = 5   # BO opened within this many bars = "fresh"


def _total_factor(max_so: int) -> float:
    f = 1.0
    for k in range(1, max_so + 1):
        f += _SO_RATIO * (_VOL_COEF ** (k - 1))
    return f


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
            if len(df) >= 30:
                result[ticker] = df
        except Exception:
            pass
    return result


def scan_ticker(ticker: str, df: pd.DataFrame) -> dict:
    """Always returns a result (DCA is always in the market)."""
    tp_pct   = PARAMS["tp_pct"]
    so_step  = PARAMS["so_step"]
    max_so   = PARAMS["max_so"]
    tf       = _total_factor(max_so)

    open_a  = df["Open"].values
    high_a  = df["High"].values
    low_a   = df["Low"].values
    close_a = df["Close"].values
    n       = len(close_a)

    cash     = 1000.0   # normalised — only used for sizing ratios
    in_deal  = False
    pending  = True
    base_p   = 0.0
    so_count = 0
    bo_vol   = 0.0
    bar_in   = 0
    spent    = 0.0
    qty      = 0.0

    for i in range(1, n):
        if pending and not in_deal:
            bo_vol   = cash * 0.33 / tf
            qty      = bo_vol / open_a[i]
            spent    = bo_vol
            base_p   = open_a[i]
            in_deal  = True
            so_count = 0
            bar_in   = i
            pending  = False

        if in_deal:
            while so_count < max_so:
                next_n = so_count + 1
                trig   = base_p * (1.0 - so_step * next_n)
                vol    = bo_vol * _SO_RATIO * (_VOL_COEF ** (next_n - 1))
                if low_a[i] <= trig:
                    qty      += vol / trig
                    spent    += vol
                    so_count += 1
                else:
                    break

            avg_entry = spent / qty if qty > 0 else base_p
            tp        = avg_entry * (1.0 + tp_pct)
            if high_a[i] >= tp:
                in_deal  = False
                pending  = True
                qty      = 0.0
                spent    = 0.0

    avg_entry = spent / qty if (in_deal and qty > 0) else close_a[-1]
    tp_level  = avg_entry * (1.0 + tp_pct)
    current   = close_a[-1]
    ret_pct   = (current / avg_entry - 1.0) * 100.0
    bars_ago  = (n - 1) - bar_in
    confidence = round(max(0.0, min((current - avg_entry) / max(tp_level - avg_entry, 1e-9) * 100.0, 100.0)))

    so_triggers = [f"-{so_step * k * 100:.0f}%" for k in range(1, max_so + 1)]

    return {
        "ticker":     ticker,
        "confidence": confidence,
        "so_filled":  so_count,
        "bars_ago":   bars_ago,
        "ret_pct":    round(ret_pct, 2),
        "avg_entry":  round(avg_entry, 2),
        "tp_level":   round(tp_level, 2),
        "fresh":      bars_ago <= FRESH_BARS,
    }


def print_results(signals: list[dict], top: int) -> None:
    fresh  = [s for s in signals if s["fresh"]]
    active = [s for s in signals if not s["fresh"]]

    def _row(s: dict) -> str:
        tag    = "★ FRESH" if s["fresh"] else "  active"
        ret    = f"+{s['ret_pct']:.1f}%" if s["ret_pct"] >= 0 else f"{s['ret_pct']:.1f}%"
        so_str = f"SO{s['so_filled']}" if s["so_filled"] > 0 else "BO only"
        return (f"  {tag}   {s['ticker']:<6}  conf {s['confidence']:>3}/100  "
                f"  {s['bars_ago']:>4}d in deal   {ret:>8} to TP   [{so_str}]")

    print(f"\n{'─' * 80}")
    print(f"  DCA LONG — SIGNAL SCAN   {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"  Params: TP {int(PARAMS['tp_pct']*100)}%  "
          f"SO step {int(PARAMS['so_step']*100)}%  Max SOs {PARAMS['max_so']}")
    print(f"  Confidence = % progress from avg entry to TP target")
    print(f"{'─' * 80}")

    if fresh:
        print(f"\n  FRESH BASE ORDERS  (opened within last {FRESH_BARS} days)\n")
        for s in fresh[:top]:
            print(_row(s))
    else:
        print(f"\n  No fresh base orders right now.\n")

    if active:
        print(f"\n  ACTIVE DEALS  (all tickers, sorted by confidence)\n")
        for s in active[:top]:
            print(_row(s))

    total = len(fresh) + len(active)
    high_conf = sum(1 for s in signals if s["confidence"] >= 70)
    print(f"\n{'─' * 80}")
    print(f"  {total} active deals  |  {len(fresh)} fresh  |  {high_conf} at ≥70% to TP\n")


def main() -> None:
    p = argparse.ArgumentParser(description="DCA Long — S&P 500 signal scanner")
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
    start = (datetime.today() - timedelta(days=548)).strftime("%Y-%m-%d")  # 18 months

    all_data = download_batch(tickers, start, end)
    print(f"  {len(all_data)} tickers downloaded successfully.\n")
    print("Scanning …")

    signals = [scan_ticker(ticker, df) for ticker, df in all_data.items()]
    signals.sort(key=lambda s: (-s["fresh"], -s["confidence"]))
    print_results(signals, top=args.top)


if __name__ == "__main__":
    main()
