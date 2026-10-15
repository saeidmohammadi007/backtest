import os
import numpy as np
import pandas as pd
import yfinance as yf
import requests
from concurrent.futures import ThreadPoolExecutor

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
MIN_COIN_N  = int(os.getenv("MIN_COIN_N", "3"))

TG_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TG_CHAT  = os.getenv("TELEGRAM_CHAT_ID")
CACHE_DIR = "cache"
os.makedirs(CACHE_DIR, exist_ok=True)


def flatten(df):
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df


def get_4h(symbol):
    path = os.path.join(CACHE_DIR, f"{symbol}.pkl")
    if os.path.exists(path):
        try:
            df = pd.read_pickle(path)
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
    try:
        df.to_pickle(path)
    except Exception:
        pass
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
        # max_favorable: بیشترین سود بدون تجاوز از ریزش
        # max_runup: بیشترین سود تا قبل از ریزش مجاز
        runup_before_dd = {}
        for t in TARGETS:
            hits = np.where(fh >= entry * (1 + t / 100))[0]
            if len(hits) == 0:
                for d in DDS:
                    result[(t, d)] = 0
                continue
            worst = runmin[hits[0]]
            for d in DDS:
                result[(t, d)] = 1 if worst > entry * (1 - d / 100) else 0

        # محاسبه سود/ضرر نهایی (بدون هدف، فقط برای رتبه‌بندی رخداد)
        future_max_pct = 100 * (np.max(fh) - entry) / entry
        future_min_pct = 100 * (np.min(fl) - entry) / entry

        out.append((bucket, result, idx[i], symbol,
                    future_max_pct, future_min_pct, entry))
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

    # تجمیع سطل‌ها
    buckets = {}
    for b, r, dt, sym, fmax, fmin, entry in all_rows:
        buckets.setdefault(b, {
            "n": 0,
            "succ": {(t, d): 0 for t in TARGETS for d in DDS},
            "events": [],
        })
        buckets[b]["n"] += 1
        for k, v in r.items():
            buckets[b]["succ"][k] += v
        buckets[b]["events"].append({
            "date": dt, "symbol": sym,
            "success": r[(T_RANK, DD_RANK)],
            "fmax": fmax, "fmin": fmin, "entry": entry,
        })

    total_n = len(all_rows)
    baseline_cnt = sum(r[(T_RANK, DD_RANK)] for _, r, _, _, _, _, _ in all_rows)
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
    evs = best_info["events"]

    # ----------------- تحلیل per-coin -----------------
    coin_stats = {}
    for e in evs:
        s = e["symbol"]
        coin_stats.setdefault(s, {"n": 0, "succ": 0, "events": []})
        coin_stats[s]["n"] += 1
        coin_stats[s]["succ"] += e["success"]
        coin_stats[s]["events"].append(e)

    coin_rank = []
    for s, st in coin_stats.items():
        if st["n"] < MIN_COIN_N:
            continue
        wr = 100.0 * st["succ"] / st["n"]
        coin_rank.append((wr, s, st))
    coin_rank.sort(reverse=True)

    # ----------------- بهترین رخداد تکی -----------------
    # معیار: موفق + بیشترین سود محقق‌شده (fmax) + کمترین ضرر (fmin نزدیک به صفر)
    succ_evs = [e for e in evs if e["success"] == 1]
    succ_evs_sorted = sorted(succ_evs, key=lambda x: -x["fmax"])

    # ----------------- گزارش -----------------
    lines = [
        "🎯 بهترین الگو + بهترین ارز + بهترین تاریخ",
        f"الگو: {bucket_label(best_b)}",
        f"n={best_info['n']} | موفقیت={best_wr:.1f}% | پایه={base_wr:.1f}% | لبه={best_wr-base_wr:+.1f}%",
        f"معیار: هدف {T_RANK:.0f}% / ریزش <{DD_RANK:.0f}%",
        "",
    ]

    # 🏆 بهترین ارز
    if coin_rank:
        lines.append("🏆 بهترین ارز برای این الگو:")
        lines.append(f"{'ارز':<10} {'n':<5} {'موفق':<5} {'وین‌ریت'}")
        lines.append("─" * 35)
        for wr, s, st in coin_rank:
            lines.append(f"{s:<10} {st['n']:<5} {st['succ']:<5} {wr:5.1f}%")
        lines.append("")
        lines.append(f"🥇 بهترین ارز: {coin_rank[0][1]} (وین‌ریت {coin_rank[0][0]:.1f}% با n={coin_rank[0][2]['n']})")
        lines.append("")

    # 🌟 بهترین رخداد تکی
    if succ_evs_sorted:
        lines.append("🌟 بهترین رخداد تکی (بیشترین سود محقق‌شده):")
        lines.append(f"{'تاریخ':<18} {'ارز':<10} {'سود max':>8} {'ضرر min':>9}")
        lines.append("─" * 50)
        for e in succ_evs_sorted[:10]:
            lines.append(
                f"{e['date'].strftime('%Y-%m-%d %H:%M'):<18} "
                f"{e['symbol']:<10} "
                f"{e['fmax']:>7.2f}% "
                f"{e['fmin']:>8.2f}%"
            )
        best = succ_evs_sorted[0]
        lines.append("")
        lines.append(f"🥇 بهترین رخداد: {best['date'].strftime('%Y-%m-%d %H:%M')} روی {best['symbol']}")
        lines.append(f"   سود حداکثری: {best['fmax']:.2f}%  |  ضرر حداکثری: {best['fmin']:.2f}%")
        lines.append("")

    # 📋 لیست همه رخدادهای موفق (مثل قبل)
    lines.append(f"📋 همه رخدادهای موفق ({len(succ_evs)} تا):")
    lines.append("─────────────")
    for e in sorted(succ_evs, key=lambda x: x["date"]):
        lines.append(f"{e['date'].strftime('%Y-%m-%d %H:%M')}  |  {e['symbol']}")

    if SHOW_FAILS:
        fails = [e for e in evs if e["success"] == 0]
        lines.append("")
        lines.append(f"❌ رخدادهای ناموفق ({len(fails)} تا):")
        lines.append("─────────────")
        for e in sorted(fails, key=lambda x: x["date"]):
            lines.append(f"{e['date'].strftime('%Y-%m-%d %H:%M')}  |  {e['symbol']}")

    send_telegram("\n".join(lines))


if __name__ == "__main__":
    main()
