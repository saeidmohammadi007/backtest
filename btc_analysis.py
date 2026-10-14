import time
import random
import yfinance as yf
import pandas as pd
import numpy as np

# ============ تنظیمات ============
TARGET_DATE = "2026-08-17"
T = {"o": 62892.0, "h": 64526.0, "l": 62778.0, "c": 64526.0}

# تیکرهای Yahoo Finance (به جای CoinGecko ID)
TOP_COINS = [
    "ETH-USD", "XRP-USD", "BNB-USD", "SOL-USD", "ADA-USD",
    "DOGE-USD", "TRX-USD", "DOT-USD", "AVAX-USD", "LINK-USD",
]
BTC_TICKER = "BTC-USD"

START_DATE, END_DATE = "2020-01-01", "2025-12-31"
LOOKBACK, DIST_TH = 3, 0.5
DROP_TH, TARGET_TH, FUTURE = 4.5, 8.4, 30

MAX_RETRIES = 3
BASE_BACKOFF = 5.0


# ============ دانلود با retry روی rate limit ============
def download_with_retry(ticker, start=None, end=None, period=None):
    last_exc = None
    for attempt in range(MAX_RETRIES + 1):
        try:
            df = yf.download(
                ticker,
                start=start, end=end, period=period,
                interval="1d",
                progress=False,
                auto_adjust=False,
                threads=False,
            )
            if df is None or df.empty:
                raise RuntimeError(f"empty data: {ticker}")
            return df
        except Exception as exc:
            last_exc = exc
            if "RateLimit" in type(exc).__name__ and attempt < MAX_RETRIES:
                sleep_for = BASE_BACKOFF * (2 ** attempt) + random.uniform(0, 0.75)
                print(f"   rate limit on {ticker}, sleeping {sleep_for:.1f}s ...")
                time.sleep(sleep_for)
                continue
            raise
    raise last_exc


def fetch_ohlc(ticker, start, end):
    """دانلود OHLC روزانه از Yahoo Finance و یکسان‌سازی فرمت خروجی"""
    df = download_with_retry(ticker, start=start, end=end)
    # اگر MultiIndex برگردد، سطح تیکر را حذف کن
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.reset_index()
    df = df.rename(columns={
        "Date": "date",
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
    })
    df = df[["date", "open", "high", "low", "close"]].dropna()
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
    return df.reset_index(drop=True)


# ============ توابع تحلیل ============
def normalize(w):
    base = float(w["close"].iloc[0])
    n = w.copy()
    for c in ["open", "high", "low", "close"]:
        n[c] = n[c] / base
    return n


def euclid(a, b):
    v1 = np.concatenate([a[c].values for c in ["open", "high", "low", "close"]])
    v2 = np.concatenate([b[c].values for c in ["open", "high", "low", "close"]])
    return float(np.linalg.norm(v1 - v2))


def outcome(entry, future_df):
    drop_lvl   = entry * (1 - DROP_TH / 100)
    target_lvl = entry * (1 + TARGET_TH / 100)
    dh = ph = False
    md = mp = 0.0
    for _, c in future_df.iterrows():
        md = max(md, (entry - float(c["low"]))  / entry * 100)
        mp = max(mp, (float(c["high"]) - entry) / entry * 100)
        if float(c["low"])  <= drop_lvl:   dh = True
        if float(c["high"]) >= target_lvl: ph = True
    return dh, ph, md, mp


# ============ اجرای اصلی ============
def main():
    print("=" * 60)
    print(f"BTC target candle: {TARGET_DATE}  (source: Yahoo Finance)")
    print("=" * 60)

    # پنجره‌ی مصنوعی قبل از کندل هدف
    target_window = pd.DataFrame([
        {"open": T["o"]*0.98,  "high": T["o"]*0.99,  "low": T["o"]*0.97, "close": T["o"]*0.985},
        {"open": T["o"]*0.985, "high": T["o"]*0.995, "low": T["o"]*0.98, "close": T["o"]*0.99},
        {"open": T["o"]*0.99,  "high": T["h"]*0.99,  "low": T["l"],       "close": T["c"]*0.99},
    ])
    tn = normalize(target_window)
    entry = T["c"]
    print(f"entry={entry:.0f}  "
          f"drop<{DROP_TH}%→{entry*(1-DROP_TH/100):.0f}  "
          f"target={TARGET_TH}%→{entry*(1+TARGET_TH/100):.0f}")

    # ---- دانلود داده ----
    data = {}
    for ticker in [BTC_TICKER] + TOP_COINS:
        print(f"fetch {ticker} ...")
        try:
            df = fetch_ohlc(ticker, START_DATE, END_DATE)
            data[ticker] = df
            print(f"   {len(df)} candles  ({df['date'].min().date()} → {df['date'].max().date()})")
        except Exception as e:
            print(f"   ⚠️  skip {ticker}: {e}")
        time.sleep(1)  # محافظت از rate limit

    # ---- جستجوی مشابه‌ها ----
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

    # ---- نتایج ----
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
    print()
    print("top 10 closest:")
    for _, r in m.head(10).iterrows():
        print(f"  {r['ticker']:10s} | {r['date']} | d={r['distance']:.4f} | "
              f"drop={r['max_drop']:>5.1f}% | profit={r['max_profit']:>5.1f}%")

    m.to_csv("btc_similar_patterns.csv", index=False)
    print("\nsaved: btc_similar_patterns.csv")


if __name__ == "__main__":
    main()
