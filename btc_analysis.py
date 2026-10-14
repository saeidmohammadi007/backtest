import time
import requests
import pandas as pd
import numpy as np

TARGET_DATE = "2026-08-17"
T = {"o": 62892.0, "h": 64526.0, "l": 62778.0, "c": 64526.0}
TOP_COINS = ["ethereum","ripple","binancecoin","solana","cardano",
             "dogecoin","tron","polkadot","avalanche-2","chainlink"]
START_DATE, END_DATE = "2020-01-01", "2025-12-31"
LOOKBACK, DIST_TH, DROP_TH, TARGET_TH, FUTURE = 3, 0.5, 4.5, 8.4, 30


def fetch_ohlc(cid):
    url = f"https://api.coingecko.com/api/v3/coins/{cid}/ohlc"
    for a in range(4):
        try:
            r = requests.get(url, params={"vs_currency": "usd", "days": "max"}, timeout=30)
            if r.status_code == 429:
                print("  rate limit, sleeping 30s...")
                time.sleep(30)
                continue
            r.raise_for_status()
            df = pd.DataFrame(r.json(), columns=["timestamp", "open", "high", "low", "close"])
            df["date"] = pd.to_datetime(df["timestamp"], unit="ms")
            return df[["date", "open", "high", "low", "close"]]
        except Exception as e:
            print(f"  retry {a+1}: {e}")
            time.sleep(5)
    raise RuntimeError(f"failed: {cid}")


def normalize(w):
    base = w["close"].iloc[0]
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
        md = max(md, (entry - c["low"]) / entry * 100)
        mp = max(mp, (c["high"] - entry) / entry * 100)
        if c["low"]  <= drop_lvl:   dh = True
        if c["high"] >= target_lvl: ph = True
    return dh, ph, md, mp


def main():
    print("=" * 60)
    print(f"BTC target candle: {TARGET_DATE}")
    print("=" * 60)

    # reconstruct a pseudo 3-candle window ending before target
    target_window = pd.DataFrame([
        {"open": T["o"]*0.98,  "high": T["o"]*0.99,  "low": T["o"]*0.97, "close": T["o"]*0.985},
        {"open": T["o"]*0.985, "high": T["o"]*0.995, "low": T["o"]*0.98, "close": T["o"]*0.99},
        {"open": T["o"]*0.99,  "high": T["h"]*0.99,  "low": T["l"],       "close": T["c"]*0.99},
    ])
    tn = normalize(target_window)
    entry = T["c"]
    print(f"entry={entry:.0f}  drop<{DROP_TH}%→{entry*(1-DROP_TH/100):.0f}  target={TARGET_TH}%→{entry*(1+TARGET_TH/100):.0f}")

    data = {}
    for cid in ["bitcoin"] + TOP_COINS:
        print(f"fetch {cid} ...")
        df = fetch_ohlc(cid)
        df = df[(df["date"] >= START_DATE) & (df["date"] <= END_DATE)].reset_index(drop=True)
        data[cid] = df
        print(f"   {len(df)} candles")
        time.sleep(2)

    matches = []
    for coin, df in data.items():
        if len(df) < LOOKBACK + FUTURE:
            continue
        for i in range(LOOKBACK, len(df) - FUTURE):
            w = df.iloc[i-LOOKBACK:i]
            t = df.iloc[i]
            d = euclid(tn, normalize(w))
            if d < DIST_TH:
                dh, ph, md, mp = outcome(t["close"], df.iloc[i+1:i+1+FUTURE])
                matches.append({
                    "coin": coin,
                    "date": str(t["date"].date()),
                    "distance": round(d, 4),
                    "drop_hit": dh,
                    "profit_hit": ph,
                    "max_drop": round(md, 2),
                    "max_profit": round(mp, 2),
                })

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
        print(f"  {r['coin']:14s} | {r['date']} | d={r['distance']:.4f} | "
              f"drop={r['max_drop']:>5.1f}% | profit={r['max_profit']:>5.1f}%")

    m.to_csv("btc_similar_patterns.csv", index=False)
    print("\nsaved: btc_similar_patterns.csv")


if __name__ == "__main__":
    main()
