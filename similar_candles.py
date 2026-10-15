import os
import numpy as np
import pandas as pd
import yfinance as yf
import requests

# ---------------- تنظیمات ----------------
COINS = [
    "BTC-USD", "ETH-USD", "XRP-USD", "BNB-USD", "SOL-USD",
    "DOGE-USD", "TRX-USD", "ADA-USD", "AVAX-USD", "LINK-USD",
]

HORIZON     = int(os.getenv("HORIZON", "60"))      # افق بررسی (60 کندل 4h = 10 روز)
TARGETS     = [5, 10, 20, 30]                       # اهداف سود (درصد)
DDS         = [2, 3, 4]                             # حداکثر ریزش مجاز (درصد)
N_BINS      = int(os.getenv("N_BINS", "10"))        # تعداد سطل برای o_pos و c_pos
MIN_SAMPLES = int(os.getenv("MIN_SAMPLES", "50"))   # حداقل نمونه برای معتبر بودن یک سطل
MIN_RANGE   = float(os.getenv("MIN_RANGE", "0.002"))# حداقل رنج نسبت به open (فیلتر نویز)

TG_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TG_CHAT  = os.getenv("TELEGRAM_CHAT_ID")


def flatten(df):
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df


def get_4h(symbol):
    """از داده 1h، 4h می‌سازه (یاهو 4h نداره)"""
    df = yf.download(symbol, period="729d", interval="1h",
                     progress=False, auto_adjust=False)
    df = flatten(df)
    if df.empty:
        return None
    return df.resample("4h").agg(
        {"Open": "first", "High": "max", "Low": "min", "Close": "last"}
    ).dropna()


def scan_coin(df):
    """
    برای هر کندل: سطل آن + نتیجه آینده‌اش را برمی‌گرداند.
    خروجی: لیستی از (bucket_key, { (target,dd): 0/1 })
    """
    o, h, l, c = (df[k].values.astype(float) for k in ["Open", "High", "Low", "Close"])
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

        out.append((bucket, result))
    return out


def bucket_label(bucket):
    ob, cb, dirn = bucket
    o_lo, o_hi = ob / N_BINS, (ob + 1) / N_BINS
    c_lo, c_hi = cb / N_BINS, (cb + 1) / N_BINS
    dir_s = "صعودی" if dirn == 1 else "نزولی"
    return f"open∈[{o_lo:.1f},{o_hi:.1f}) close∈[{c_lo:.1f},{c_hi:.1f}) {dir_s}"


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
    # 1) اسکن همه ارزها
    all_rows = []
    for sym in COINS:
        try:
            df = get_4h(sym)
            if df is None:
                print(f"{sym}: داده نیامد")
                continue
            rows = scan_coin(df)
            print(f"{sym}: {len(rows)} کندل اسکن شد")
            for b, r in rows:
                all_rows.append((b, r, sym))
        except Exception as e:
            print(f"{sym}: خطا {e}")

    if not all_rows:
        send_telegram("هیچ داده‌ای اسکن نشد.")
        return

    # 2) تجمیع در سطل‌ها
    buckets = {}
    for b, r, sym in all_rows:
        if b not in buckets:
            buckets[b] = {"n": 0, "succ": {(t, d): 0 for t in TARGETS for d in DDS},
                          "coins": {}}
        buckets[b]["n"] += 1
        for k, v in r.items():
            buckets[b]["succ"][k] += v
        buckets[b]["coins"][sym] = buckets[b]["coins"].get(sym, 0) + 1

    # 3) بازه پایه (baseline) روی کل داده
    total_n = len(all_rows)
    baseline = {(t, d): 0 for t in TARGETS for d in DDS}
    for _, r, _ in all_rows:
        for k, v in r.items():
            baseline[k] += v

    # 4) رتبه‌بندی سطل‌ها بر اساس هدف/ریزش مشخص (TARGET_RANK/DD_RANK)
    T_RANK = float(os.getenv("TARGET_RANK", "5"))
    DD_RANK = float(os.getenv("DD_RANK", "3"))
    key = (T_RANK, DD_RANK)

    ranked = []
    for b, info in buckets.items():
        if info["n"] < MIN_SAMPLES:
            continue
        wr = 100.0 * info["succ"][key] / info["n"]
        base_wr = 100.0 * baseline[key] / total_n
        ranked.append((wr, base_wr, b, info))

    ranked.sort(reverse=True)

    # 5) ساخت گزارش
    lines = [
        "🔍 اسکنر خودکار بهترین الگوی کندلی",
        f"تعداد کل کندل اسکن‌شده: {total_n}",
        f"افق: {HORIZON} کندل 4h | حداقل رنج: {MIN_RANGE*100:.2f}%",
        f"سطل‌بندی: {N_BINS}×{N_BINS}×2 (open × close × جهت)",
        f"معیار رتبه‌بندی: هدف {T_RANK:.0f}% / ریزش <{DD_RANK:.0f}%",
        f"حداقل نمونه هر سطل: {MIN_SAMPLES}",
        "",
        f"📊 بازه پایه (کل داده): {100.0*baseline[key]/total_n:.1f}%",
        "",
    ]

    if not ranked:
        lines.append("⚠️ هیچ سطلی حداقل نمونه را نداشت. MIN_SAMPLES را کم کن.")
        send_telegram("\n".join(lines))
        return

    lines.append("🏆 ۱۵ سطل برتر (فراتر از بازه پایه):")
    lines.append("-" * 40)
    for wr, base_wr, b, info in ranked[:15]:
        edge = wr - base_wr
        lines.append(f"{bucket_label(b)}")
        lines.append(f"  n={info['n']:<5} موفقیت={wr:5.1f}%  (پایه={base_wr:4.1f}%، لبه={edge:+.1f}%)")
        # ارزهایی که بیشترین سهم را داشتند
        top_coins = sorted(info["coins"].items(), key=lambda x: -x[1])[:3]
        lines.append("  بیشتر در: " + ", ".join(f"{s}({c})" for s, c in top_coins))
        lines.append("")

    lines.append("📋 عملکرد کامل بهترین سطل:")
    best_b = ranked[0][2]
    best_info = ranked[0][3]
    lines.append(f"الگو: {bucket_label(best_b)}")
    lines.append("هدف |  ریزش<2%  ریزش<3%  ریزش<4%")
    for t in TARGETS:
        row = "  ".join(
            f"{100.0*best_info['succ'][(t,d)]/best_info['n']:6.1f}%" for d in DDS
        )
        lines.append(f"{t:>3}% | {row}")
    lines.append("")

    lines.append("📉 بدترین سطل (برای مقایسه):")
    worst = ranked[-1]
    lines.append(f"الگو: {bucket_label(worst[2])}")
    lines.append(f"  n={worst[3]['n']:<5} موفقیت={worst[0]:5.1f}%")

    send_telegram("\n".join(lines))


if __name__ == "__main__":
    main()
