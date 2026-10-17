"""
candle_finder.py
پیدا کردن خودکار «شکل کندل ۴ ساعته»ای که ۱۰۰ کندل بعدش
کمترین ریزش اولیه، بالاترین ریسک‌به‌ریوارد و بالاترین وین‌ریت را داشته.

- داده: LBank با ccxt
- شکل کندل: بدنه + سایه بالا + سایه پایین، تقسیم بر ATR (قابل مقایسه بین کوین‌ها)
- شباهت: فاصله اقلیدسی (KMeans)
- آمار: وقوع‌های هم‌پوشان حذف می‌شوند (فاصله حداقل DEDUP_GAP کندل)
- اعتبارسنجی: شکل‌ها روی ۷۰٪ اول تاریخ انتخاب و روی ۳۰٪ آخر تست می‌شوند
"""
import os
import time
import requests
import ccxt
import numpy as np
import pandas as pd
from datetime import timedelta
from numpy.lib.stride_tricks import sliding_window_view
from sklearn.cluster import KMeans

# ---------------- تنظیمات ----------------
DEFAULT_SYMBOLS = "BTC/USDT,ETH/USDT,SOL/USDT,XRP/USDT,BNB/USDT,DOGE/USDT,ADA/USDT,LINK/USDT,AVAX/USDT,LTC/USDT"
SYMBOLS = [s.strip() for s in os.getenv("SYMBOLS", DEFAULT_SYMBOLS).split(",") if s.strip()]
TIMEFRAME = "4h"
HORIZON = int(os.getenv("HORIZON", "100"))          # کندل‌های بعد از هر مورد
K = int(os.getenv("K", "40"))                       # تعداد گروه شکل کندل
ATR_N = 14
DEDUP_GAP = int(os.getenv("DEDUP_GAP", str(HORIZON)))  # حداقل فاصله بین دو وقوع شمارش‌شده
MIN_TRAIN = int(os.getenv("MIN_TRAIN", "60"))       # حداقل وقوع در دوره انتخاب
MIN_TEST = int(os.getenv("MIN_TEST", "20"))         # حداقل وقوع در دوره تست
MIN_WIN_TEST = float(os.getenv("MIN_WIN_TEST", "55"))  # حداقل وین‌ریت تست برای تایید (٪)
TOP_N = int(os.getenv("TOP_N", "3"))
START_DATE = os.getenv("START_DATE", "2020-01-01")
MIN_DD = 0.001                                      # کف ریزش برای محاسبه RR
FEATS = ["body_a", "up_a", "lo_a"]

TG_TOKEN = os.getenv("TELEGRAM_TOKEN", "")
TG_CHAT = os.getenv("TELEGRAM_CHAT_ID", "")


# ---------------- دریافت داده ----------------
def fetch_all(ex, symbol):
    since = ex.parse8601(START_DATE + "T00:00:00Z")
    tf_ms = ex.parse_timeframe(TIMEFRAME) * 1000
    rows = []
    while True:
        batch = ex.fetch_ohlcv(symbol, TIMEFRAME, since=since, limit=1000)
        if not batch:
            break
        rows += batch
        last = batch[-1][0]
        if last + tf_ms >= ex.milliseconds() or len(batch) < 2:
            break
        since = last + tf_ms
        time.sleep(ex.rateLimit / 1000)
    df = pd.DataFrame(rows, columns=["ts", "o", "h", "l", "c", "v"])
    df = df.drop_duplicates("ts").sort_values("ts").reset_index(drop=True)
    df = df[df["ts"] + tf_ms <= ex.milliseconds()].reset_index(drop=True)  # فقط کندل‌های بسته
    df["time"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    return df


# ---------------- ویژگی‌ها و نتیجه‌ی ۱۰۰ کندل بعد ----------------
def build(df, symbol):
    df = df.copy()
    n, H = len(df), HORIZON
    prev_c = df["c"].shift(1)
    tr = pd.concat([df["h"] - df["l"], (df["h"] - prev_c).abs(), (df["l"] - prev_c).abs()], axis=1).max(axis=1)
    atr = tr.rolling(ATR_N).mean() / df["c"]

    top = df[["o", "c"]].max(axis=1)
    bot = df[["o", "c"]].min(axis=1)
    df["body_r"] = (df["c"] - df["o"]) / df["o"]
    df["up_r"] = (df["h"] - top) / df["o"]
    df["lo_r"] = (bot - df["l"]) / df["o"]
    for f in ["body", "up", "lo"]:
        df[f + "_a"] = (df[f + "_r"] / atr).clip(-6, 6)

    mfe = np.full(n, np.nan)
    idd = np.full(n, np.nan)    # ریزش اولیه: عمیق‌ترین افت قبل از رسیدن به سقف ۱۰۰ کندل
    fret = np.full(n, np.nan)   # بازده بسته شدن کندل صدم
    if n > H + 1:
        h, l, c = df["h"].values, df["l"].values, df["c"].values
        hw = sliding_window_view(h[1:], H)
        lw = sliding_window_view(l[1:], H)
        m = n - H
        ci = c[:m]
        peak = hw.argmax(axis=1)
        cummin = np.minimum.accumulate(lw, axis=1)
        dd_peak = cummin[np.arange(m), peak]
        mfe[:m] = np.clip((hw.max(axis=1) - ci) / ci, 0, None)
        idd[:m] = np.clip((ci - dd_peak) / ci, 0, None)
        fret[:m] = (c[H:H + m] - ci) / ci
    df["mfe"], df["idd"], df["fret"] = mfe, idd, fret
    df["rr"] = df["mfe"] / np.maximum(df["idd"], MIN_DD)
    df["win"] = df["fret"] > 0
    df["symbol"] = symbol
    df["idx"] = np.arange(n)
    return df


# ---------------- آمار ----------------
def dedup(d):
    keep = []
    for _, g in d.groupby(["symbol", "cl"], sort=False):
        last = -10 ** 9
        for i, ix in zip(g.index, g["idx"].values):
            if ix - last >= DEDUP_GAP:
                keep.append(i)
                last = ix
    return d.loc[keep]


def agg(d):
    return d.groupby("cl").agg(n=("win", "size"), win=("win", "mean"), ret=("fret", "mean"),
                               idd=("idd", "median"), rr=("rr", "median"))


def fmt(t):
    th = t + timedelta(hours=3, minutes=30)
    return f"{t:%Y-%m-%d %H:%M} UTC ({th:%m-%d %H:%M} تهران)"


# ---------------- تحلیل اصلی ----------------
def analyze(pool):
    tf_ms = 4 * 3600 * 1000
    pool = pool.dropna(subset=FEATS).reset_index(drop=True)
    cutoff = pool["ts"].quantile(0.7)

    # دوره انتخاب (با فاصله ایمنی تا دوره تست) و دوره تست
    train_mask = pool["ts"] < cutoff - HORIZON * tf_ms
    test_mask = pool["ts"] >= cutoff

    km = KMeans(n_clusters=K, n_init=5, random_state=0).fit(pool.loc[train_mask, FEATS].values)
    pool["cl"] = km.predict(pool[FEATS].values)

    lab = pool.dropna(subset=["fret"])
    tr = dedup(lab[train_mask.loc[lab.index]])
    te = dedup(lab[test_mask.loc[lab.index]])
    s = agg(tr).join(agg(te), lsuffix="_tr", rsuffix="_te", how="inner")
    s = s[(s["n_tr"] >= MIN_TRAIN) & (s["n_te"] >= MIN_TEST)].copy()
    if s.empty:
        return ["شکلی با تعداد وقوع کافی پیدا نشد (K یا MIN_TRAIN / MIN_TEST را کمتر کنید)."]

    s["score"] = (s["win_tr"].rank(ascending=False)
                  + s["idd_tr"].rank(ascending=True)
                  + s["rr_tr"].rank(ascending=False))
    s["ok"] = (s["win_te"] * 100 >= MIN_WIN_TEST) & (s["ret_te"] > 0)
    s = s.sort_values("score")

    allev = dedup(lab)
    shown = list(s.head(TOP_N).index)
    out = [f"🔎 {len(SYMBOLS)} کوین | {TIMEFRAME} | افق {HORIZON} کندل | {len(lab):,} کندل بررسی‌شده",
           f"انتخاب شکل: تا {fmt(pd.to_datetime(cutoff, unit='ms', utc=True))} | تست: بعد از آن", ""]

    for rank, cl in enumerate(shown, 1):
        r = s.loc[cl]
        g = allev[allev["cl"] == cl]
        b, u, lo = g["body_r"].mean() * 100, g["up_r"].mean() * 100, g["lo_r"].mean() * 100
        flag = "✅ در تست هم تایید شد" if r["ok"] else "⚠️ در تست تایید نشد"
        out += [
            f"#{rank} کندل {'سبز' if b > 0 else 'قرمز'} | بدنه {b:+.2f}% | سایه بالا {u:.2f}% | سایه پایین {lo:.2f}%",
            f"{flag}",
            f"انتخاب → وقوع {int(r.n_tr)} | وین‌ریت {r.win_tr*100:.1f}% | میانه ریزش اولیه {r.idd_tr*100:.2f}% | میانه RR {r.rr_tr:.2f}",
            f"تست → وقوع {int(r.n_te)} | وین‌ریت {r.win_te*100:.1f}% | میانه ریزش اولیه {r.idd_te*100:.2f}% | میانه RR {r.rr_te:.2f} | میانگین بازده {r.ret_te*100:+.1f}%",
            "بهترین موارد (برنده با کمترین ریزش اولیه):",
        ]
        for _, x in g[g["win"]].sort_values("idd").head(3).iterrows():
            out.append(f"  {x.symbol} | {fmt(x.time)} | DD {x.idd*100:.2f}% | RR {x.rr:.1f}")
        x = g.sort_values("ts").iloc[-1]
        out += [f"آخرین وقوع: {x.symbol} | {fmt(x.time)}", ""]

    # سیگنال زنده: آخرین کندل بسته‌شده‌ی هر کوین
    out.append("📡 آخرین کندل بسته‌شده:")
    last_rows = pool.sort_values("idx").groupby("symbol").tail(1)
    hit = False
    good = [c for c in shown if s.loc[c, "ok"]]
    for _, x in last_rows.iterrows():
        if x["cl"] in good:
            hit = True
            out.append(f"🚨 {x.symbol} | {fmt(x.time)} → شکل تاییدشده #{shown.index(x['cl']) + 1}")
    if not hit:
        out.append("هیچ‌کدام جزو شکل‌های تاییدشده نیست.")
    out += ["", "تعریف‌ها: ریزش اولیه = عمیق‌ترین افت قبل از رسیدن به سقف ۱۰۰ کندل؛ RR = بیشترین سود ÷ ریزش اولیه؛ برد = بسته شدن کندل صدم بالاتر از ورود (close کندل شکل)."]
    return out


def send(text):
    print(text)
    if not (TG_TOKEN and TG_CHAT):
        return
    for i in range(0, len(text), 4000):
        requests.post(f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage",
                      data={"chat_id": TG_CHAT, "text": text[i:i + 4000]}, timeout=30)


def main():
    ex = ccxt.lbank({"enableRateLimit": True})
    frames = []
    for sym in SYMBOLS:
        try:
            d = fetch_all(ex, sym)
            if len(d) < HORIZON + 200:
                print(f"{sym}: داده کم، رد شد")
                continue
            frames.append(build(d, sym))
            print(f"{sym}: {len(d)} candles")
        except Exception as e:
            print(f"{sym}: خطا — {e}")
    if not frames:
        send("هیچ داده‌ای دریافت نشد. (ممکن است LBank آی‌پی گیت‌هاب را بلاک کرده باشد)")
        return
    send("\n".join(analyze(pd.concat(frames, ignore_index=True))))


if __name__ == "__main__":
    main()
