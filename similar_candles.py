import time
import random
import yfinance as yf
import pandas as pd
import numpy as np

TARGET_DATE = "2026-08-17"
BTC_TICKER = "BTC-USD"
TOP_COINS = [
    "ETH-USD", "XRP-USD", "BNB-USD", "SOL-USD", "ADA-USD",
    "DOGE-USD", "TRX-USD", "DOT-USD", "AVAX-USD", "LINK-USD",
]
START_DATE, END_DATE = "2020-01-01", "2025-12-31"
LOOKBACK, DIST_TH = 3, 0.08
DROP_TH, TARGET_TH, FUTURE = 4.5, 8.4, 30
MAX_RETRIES, BASE_BACKOFF = 3, 5.0


def download_with_retry(ticker, start, end):
    for attempt in range(MAX_RETRIES + 1):
        try:
            df = yf.download(ticker, start=start, end=end, interval="1d",
                             progress=False, auto_adjust=False, threads=False)
            if df is None or df.empty:
                raise RuntimeError(f"empty: {ticker}")
            return df
        except Exception as exc:
            if "RateLimit" in type(exc).__name__ and attempt < MAX_RETRIES:
                s = BASE_BACKOFF * (2 ** attempt) + random.uniform(0, 0.75)
                print(f"   rate limit {ticker}, sleep {s:.1f}s")
                time.sleep(s)
                continue
            raise


def fetch_ohlc(ticker, start, end):
    df = download_with_retry(ticker, start, end)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.reset_index().rename(columns={
        "Date": "date", "Open": "open", "High": "high",
        "Low": "low", "Close": "close",
    })
    df = df[["date", "open", "high", "low", "close"]].dropna()
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
    return df.reset_index(drop=True)


def normalize(w):
    b = float(w["close"].iloc[0])
    n = w.copy()
    for c in ["open", "high", "low", "close"]:
        n[c] = n[c] / b
    return n


def euclid(a, b):
    v1 = np.concatenate([a[c].values for c in ["open", "high", "low", "close"]])
    v2 = np.concatenate([b[c].values for c in ["open", "high", "low", "close"]])
    return float(np.linalg.norm(v1 - v2))


def outcome(entry, future_df):
    dl = entry * (1 - DROP_TH / 100)
    pl = entry * (1 + TARGET_TH / 100)
    dh = ph = False
    md = mp = 0.0
    for _, c in future_df.iterrows():
        md = max(md, (entry - float(c["low"])) / entry * 100)
        mp = max(mp, (float(c["high"]) - entry) / entry * 100)
        if float(c["low"])  <= dl: dh = True
        if float(c["high"]) >= pl: ph = True
    return dh, ph, md, mp


def main():
    print("=" * 60)
    print(f"BTC target candle: {TARGET_DATE}  (source: Yahoo Finance)")
    print(f"DIST_TH = {DIST_TH}  |  DROP_TH = {DROP_TH}%  |  TARGET_TH = {TARGET_TH}%")
    print("=" * 60)

    # ---- ۱. دریافت کندل هدف واقعی + ۳ کندل قبلش از Yahoo ----
    btc_recent = fetch_ohlc(BTC_TICKER, "2026-08-01", "2026-08-20").reset_index(drop=True)

    target_rows = btc_recent[btc_recent["date"] == pd.Timestamp(TARGET_DATE)]
    if target_rows.empty:
        print(f"❌ کندل {TARGET_DATE} در Yahoo پیدا نشد.")
        print("داده‌های موجود در بازه:")
        print(btc_recent[["date", "open", "high", "low", "close"]].to_string())
        return

    idx = target_rows.index[0]
    if idx < LOOKBACK:
        print(f"❌ کندل کافی قبل از {TARGET_DATE} موجود نیست.")
        return

    target_window = btc_recent.iloc[idx - LOOKBACK:idx].copy()
    target_candle = btc_recent.iloc[idx]
    entry = float(target_candle["close"])

    print(f"\n🎯 کندل هدف واقعی:")
    print(f"   تاریخ: {target_candle['date'].date()}")
    print(f"   O={float(target_candle['open']):.2f}  "
          f"H={float(target_candle['high']):.2f}  "
          f"L={float(target_candle['low']):.2f}  "
          f"C={entry:.2f}")
    print(f"   سطح ریزش {DROP_TH}%: {entry*(1-DROP_TH/100):.2f}")
    print(f"   سطح هدف  {TARGET_TH}%: {entry*(1+TARGET_TH/100):.2f}")

    tn = normalize(target_window)

    # ---- ۲. دانلود داده‌های تاریخی ۲۰۲۰-۲۰۲۵ ----
    data = {}
    for ticker in [BTC_TICKER] + TOP_COINS:
        print(f"fetch {ticker} ...")
        try:
            df = fetch_ohlc(ticker, START_DATE, END_DATE)
            data[ticker] = df
            print(f"   {len(df)} candles")
        except Exception as e:
            print(f"   ⚠️ skip {ticker}: {e}")
        time.sleep(1)

    # ---- ۳. جستجوی مشابه‌ها ----
    matches = []
    for ticker, df in data.items():
        if len(df) < LOOKBACK + FUTURE:
            continue
        for i in range(LOOKBACK, len(df) - FUTURE):
            w = df.iloc[i - LOOKBACK:i]
            t = df.iloc[i]
            d = euclid(tn, normalize(w))
            if d < DIST_TH:
                dh, ph, md, mp = outcome(float(t["close"]), df.iloc[i+1:i+1+FUTURE])
                matches.append({
                    "ticker": ticker,
                    "date": str(t["date"].date()),
                    "distance": round(d, 4),
                    "drop_hit": dh,
                    "profit_hit": ph,
                    "max_drop": round(md, 2),
                    "max_profit": round(mp, 2),
                })

    # ---- ۴. نتایج ----
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)

    if not matches:
        print(f"No matches (DIST_TH={DIST_TH}). Increase DIST_TH and rerun.")
        return

    m = pd.DataFrame(matches).sort_values("distance").reset_index(drop=True)
    total = len(m)
    drop_n   = int(m["drop_hit"].sum())
    profit_n = int(m["profit_hit"].sum())

    print(f"total matches : {total}")
    print(f"drop <{DROP_TH}%   : {drop_n} ({drop_n/total*100:.1f}%)")
    print(f"hit  {TARGET_TH}%   : {profit_n} ({profit_n/total*100:.1f}%)")
    print()
    print("max_drop / max_profit stats:")
    print(m[["max_drop", "max_profit"]].describe().round(2))
    print("\ntop 10 closest:")
    for _, r in m.head(10).iterrows():
        print(f"  {r['ticker']:10s} | {r['date']} | d={r['distance']:.4f} | "
              f"drop={r['max_drop']:>5.1f}% | profit={r['max_profit']:>5.1f}%")

    m.to_csv("btc_similar_patterns.csv", index=False)
    print("\nsaved: btc_similar_patterns.csv")


if __name__ == "__main__":
    main()
