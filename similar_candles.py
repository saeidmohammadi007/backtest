import os
import math
import time
import requests
import ccxt
import pandas as pd
from datetime import datetime, timezone, timedelta

# ---------------- تنظیمات (از Environment قابل تغییر) ----------------
SYMBOL = os.getenv("SYMBOL", "ETH/USDT")
TIMEFRAME = "4h"
HORIZON = int(os.getenv("HORIZON", "100"))        # تعداد کندل بعد از هر مورد مشابه
THRESHOLD = float(os.getenv("THRESHOLD", "0.006"))  # فاصله اقلیدسی مجاز (0.006 = 0.6%)
REF_TIME = os.getenv("REF_TIME", "")              # مثلا 2026-10-18T08:00:00 (UTC)؛ خالی = آخرین کندل بسته
START_DATE = os.getenv("START_DATE", "2020-01-01")
TOP_N = int(os.getenv("TOP_N", "10"))             # تعداد برترین‌ها در گزارش
MIN_MAE = 0.001                                   # حداقل ریزش برای جلوگیری از تقسیم بر صفر

TG_TOKEN = os.getenv("TELEGRAM_TOKEN", "")
TG_CHAT = os.getenv("TELEGRAM_CHAT_ID", "")


def fetch_all(exchange, symbol, tf, start_iso):
    since = exchange.parse8601(start_iso + "T00:00:00Z")
    tf_ms = exchange.parse_timeframe(tf) * 1000
    rows = []
    while True:
        batch = exchange.fetch_ohlcv(symbol, tf, since=since, limit=1000)
        if not batch:
            break
        rows += batch
        last = batch[-1][0]
        if last + tf_ms >= exchange.milliseconds() or len(batch) < 2:
            break
        since = last + tf_ms
        time.sleep(exchange.rateLimit / 1000)
    df = pd.DataFrame(rows, columns=["ts", "o", "h", "l", "c", "v"])
    df = df.drop_duplicates("ts").sort_values("ts").reset_index(drop=True)
    df["time"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    # آخرین کندل ممکن است هنوز بسته نشده باشد
    df = df[df["ts"] + tf_ms <= exchange.milliseconds()].reset_index(drop=True)
    return df


def add_features(df):
    top = df[["o", "c"]].max(axis=1)
    bot = df[["o", "c"]].min(axis=1)
    df["body"] = (df["c"] - df["o"]) / df["o"]
    df["up"] = (df["h"] - top) / df["o"]
    df["lo"] = (bot - df["l"]) / df["o"]
    return df


def fmt_time(t):
    tehran = t + timedelta(hours=3, minutes=30)
    return f"{t:%Y-%m-%d %H:%M} UTC | {tehran:%Y-%m-%d %H:%M} تهران"


def main():
    ex = ccxt.lbank({"enableRateLimit": True})
    df = add_features(fetch_all(ex, SYMBOL, TIMEFRAME, START_DATE))
    print(f"{len(df)} candles loaded for {SYMBOL}")

    if REF_TIME:
        ref_ts = pd.Timestamp(REF_TIME, tz="UTC")
        idx = df.index[df["time"] == ref_ts]
        if len(idx) == 0:
            raise SystemExit("REF_TIME پیدا نشد (باید دقیقاً روی شروع یک کندل ۴ساعته باشد)")
        ref_i = idx[0]
    else:
        ref_i = len(df) - 1
    ref = df.iloc[ref_i]

    results = []
    for i in range(0, len(df) - HORIZON - 1):
        if i == ref_i:
            continue
        r = df.iloc[i]
        d = math.sqrt((r.body - ref.body) ** 2 + (r.up - ref.up) ** 2 + (r.lo - ref.lo) ** 2)
        if d > THRESHOLD:
            continue
        entry = r.c
        fut = df.iloc[i + 1: i + 1 + HORIZON]
        mae = max((entry - fut["l"].min()) / entry, 0)   # بیشترین ریزش نسبت به ورود
        mfe = max((fut["h"].max() - entry) / entry, 0)   # بیشترین سود ممکن
        rr = mfe / max(mae, MIN_MAE)
        win = fut["c"].iloc[-1] > entry                   # برد = بسته شدن کندل صدم بالاتر از ورود
        results.append({"time": r.time, "dist": d, "mae": mae, "mfe": mfe,
                        "rr": rr, "win": win})

    if not results:
        msg = f"{SYMBOL} {TIMEFRAME}: موردی با THRESHOLD={THRESHOLD} پیدا نشد."
        print(msg)
        send(msg)
        return

    res = pd.DataFrame(results)
    winrate = res["win"].mean() * 100

    lines = [
        f"📊 {SYMBOL} | {TIMEFRAME} | افق {HORIZON} کندل",
        f"کندل مرجع: {fmt_time(ref.time)}",
        f"تعداد موارد مشابه: {len(res)} | وین‌ریت کل: {winrate:.1f}%",
        f"میانگین ریزش: {res.mae.mean()*100:.2f}% | میانگین سود حداکثر: {res.mfe.mean()*100:.2f}% | میانگین RR: {res.rr.mean():.2f}",
        "",
        f"🔻 کمترین ریزش اولیه (Top {min(TOP_N, 5)}):",
    ]
    for _, x in res.sort_values("mae").head(min(TOP_N, 5)).iterrows():
        lines.append(f"{fmt_time(x.time)} | DD {x.mae*100:.2f}% | RR {x.rr:.2f} | {'✅' if x.win else '❌'}")

    lines += ["", f"🏆 بالاترین ریسک به ریوارد (Top {min(TOP_N, 5)}):"]
    for _, x in res.sort_values("rr", ascending=False).head(min(TOP_N, 5)).iterrows():
        lines.append(f"{fmt_time(x.time)} | RR {x.rr:.2f} | DD {x.mae*100:.2f}% | MFE {x.mfe*100:.1f}%")

    best = res[res["win"]].sort_values(["rr", "mae"], ascending=[False, True]).head(1)
    if len(best):
        x = best.iloc[0]
        lines += ["", f"⭐ بهترین کندل (برنده + بالاترین RR): {fmt_time(x.time)}",
                  f"DD {x.mae*100:.2f}% | RR {x.rr:.2f} | MFE {x.mfe*100:.1f}%"]

    msg = "\n".join(lines)
    print(msg)
    send(msg)


def send(text):
    if not (TG_TOKEN and TG_CHAT):
        return
    requests.post(f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage",
                  data={"chat_id": TG_CHAT, "text": text}, timeout=30)


if __name__ == "__main__":
    main()
