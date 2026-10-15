import os
import numpy as np
import pandas as pd
import yfinance as yf
import requests
from concurrent.futures import ThreadPoolExecutor

# ---------------- تنظیمات ----------------
COINS = [
    "BTC-USD", "ETH-USD", "XRP-USD", "BNB-USD", "SOL-USD",
    "DOGE-USD", "TRX-USD", "ADA-USD", "AVAX-USD", "LINK-USD",
]

HORIZON     = int(os.getenv("HORIZON", "60"))
TARGETS     = [5, 10, 20, 30]
DDS         = [2, 3, 4]
N_BINS      = int(os.getenv("N_BINS", "10"))
MIN_SAMPLES = int(os.getenv("MIN_SAMPLES", "50"))
MIN_RANGE   = float(os.getenv("MIN_RANGE", "0.002"))
T_RANK      = float(os.getenv("TARGET_RANK", "5"))
DD_RANK     = float(os.getenv("DD_RANK", "3"))
SHOW_FAILS  = os.getenv("SHOW_FAILS", "0") == "1"

TG_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TG_CHAT  = os.getenv("TELEGRAM_CHAT_ID")
CACHE_DIR = "cache"
os.makedirs(CACHE_DIR, exist_ok=True)


def flatten(df):
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df


def get_4h(symbol):
    path = os.path.join(CACHE_DIR, f"{symbol}.parquet")
    if os.path.exists(path):
        try:
            df = pd.read_parquet(path)
            if len(df) > 0:
                return df
        except Exception:
            pass
    df = yf.download(symbol, period="729d", interval="1h",
                     progress=False, auto_adjust=False)
    df = flatten(df)
    if df.empty:
        return None
    df = df.resample("4h").agg(
        {"Open": "first", "High": "max", "Low": "min", "Close": "last"}
    ).dropna()
    df.to_parquet(path)
    return df


def scan_coin(df, symbol):
    o, h, l, c = (df[k].values.astype(float) for k in ["Open", "High", "Low", "Close"])
    idx = df.index
    n = len(df)
    if n < HORIZON + 10:
        return []

    rng = h - l
    valid = (rng > 0) & (rng / np.where(o == 0, np.nan, o) >= MIN_RANGE)
    valid = np.nan_to_num(valid, nan=False)

    safe = np.where(rng > 0, rng, 1.0)
    o_pos = (o - l) / safe
    c_pos = (c - l) / safe
    direction = (c >= o).astype(int)

    out = []
    for i in range(0, n - HORIZON):
        if not valid[i]:
            continue
        entry = c[i]
        fh = h[i + 1:i + 1 + HORIZON]
        fl = l[i + 1:i + 1 + HORIZON]
        runmin = np.minimum.accumulate(fl)

        ob = min(int(o_pos[i] * N_BINS), N_BINS - 1)
        cb = min(int(c_pos[i] * N_BINS), N_BINS - 1)
        bucket = (ob, cb, int(direction[i]))

        result = {}
        for t in TARGETS:
            hits = np.where(fh >= entry * (1 + t / 100))[0]
            if len(hits) == 0:
                for d in DDS:
                    result[(t, d)] = 0
                continue
            worst = runmin[hits[0]]
            for d in DDS:
                result[(t, d)] = 1 if worst > entry * (1 - d / 100) else 0

        out.append((bucket, result, idx[i], symbol))
    return out


def bucket_label(bucket):
    ob, cb, dirn = bucket
    o_lo, o_hi = ob / N_BINS, (ob + 1) / N_BINS
    c_lo, c_hi = cb / N_BINS, (cb + 1) / N_BINS
    dir_s = "صعودی" if dirn == 1 else "نزولی"
    return f"open∈[{o_lo:.2f},{o_hi:.2f}) close∈[{c_lo:.2f},{c_hi:.2f}) {dir_s}"


def send_telegram(text):
    if not TG_TOKEN or not TG_CHAT:
        print(text)
        return
    url = f"https://api.telegram.org/bot{TG_TOKEN.strip()}/sendMessage"
    chunks, cur = [], ""
    for line in text.split("\n"):
        if len(cur) + len(line) + 1 > 3800:
            chunks.append(cur)
            cur = ""
        cur += line + "\n"
    if cur:
        chunks.append(cur)
    for ch in chunks:
        r = requests.post(url, data={"chat_id": TG_CHAT.strip(), "text": ch}, timeout=30)
        r.raise_for_status()


def main():
    print("در حال دانلود داده‌ها ...")
    with ThreadPoolExecutor(max_workers=5) as ex:
        dfs = list(ex.map(get_4h, COINS))

    all_rows = []
    for sym, df in zip(COINS, dfs):
        if df is None:
            continue
        all_rows.extend(scan_coin(df, sym))

    if not all_rows:
        send_telegram("داده‌ای نیامد.")
        return

    buckets = {}
    for b, r, dt, sym in all_rows:
        buckets.setdefault(b, {
            "n": 0,
            "succ": {(t, d): 0 for t in TARGETS for d in DDS},
            "events": [],
        })
        buckets[b]["n"] += 1
        for k, v in r.items():
            buckets[b]["succ"][k] += v
        buckets[b]["events"].append((dt, sym, r[(T_RANK, DD_RANK)]))

    total_n = len(all_rows)
    baseline_cnt = sum(r[(T_RANK, DD_RANK)] for _, r, _, _ in all_rows)
    base_wr = 100.0 * baseline_cnt / total_n

    key = (T_RANK, DD_RANK)
    ranked = []
    for b, info in buckets.items():
        if info["n"] < MIN_SAMPLES:
            continue
        wr = 100.0 * info["succ"][key] / info["n"]
        ranked.append((wr, b, info))
    ranked.sort(reverse=True)

    if not ranked:
        send_telegram("هیچ سطلی معتبر نبود.")
        return

    best_wr, best_b, best_info = ranked[0]
    evs = sorted(best_info["events"], key=lambda x: x[0])
    succ = [(d, s) for d, s, ok in evs if ok == 1]
    fail = [(d, s) for d, s, ok in evs if ok == 0]

    lines = [
        "🔍 بهترین الگو + لیست رخدادها",
        f"الگو: {bucket_label(best_b)}",
        f"n={best_info['n']} | موفقیت={best_wr:.1f}% | پایه={base_wr:.1f}% | لبه={best_wr-base_wr:+.1f}%",
        f"معیار: هدف {T_RANK:.0f}% / ریزش <{DD_RANK:.0f}%",
        "",
        f"✅ رخدادهای موفق ({len(succ)} تا):",
        "─────────────",
    ]
    for d, s in succ:
        lines.append(f"{d.strftime('%Y-%m-%d %H:%M')}  |  {s}")

    if SHOW_FAILS:
        lines.append("")
        lines.append(f"❌ رخدادهای ناموفق ({len(fail)} تا):")
        lines.append("─────────────")
        for d, s in fail:
            lines.append(f"{d.strftime('%Y-%m-%d %H:%M')}  |  {s}")

    lines.append("")
    lines.append("📊 سهم هر ارز در رخدادهای موفق:")
    coin_count = {}
    for _, s in succ:
        coin_count[s] = coin_count.get(s, 0) + 1
    for s, cnt in sorted(coin_count.items(), key=lambda x: -x[1]):
        lines.append(f"  {s:<10} {cnt}")

    lines.append("")
    lines.append("📅 توزیع سالانه رخدادهای موفق:")
    year_count = {}
    for d, _ in succ:
        year_count[d.year] = year_count.get(d.year, 0) + 1
    for y in sorted(year_count):
        bar = "█" * min(year_count[y], 40)
        lines.append(f"  {y}  {year_count[y]:>3}  {bar}")

    send_telegram("\n".join(lines))


if __name__ == "__main__":
    main()
