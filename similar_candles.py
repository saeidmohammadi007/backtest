import os
import numpy as np
import pandas as pd
import yfinance as yf
import requests

# ---------------- تنظیمات ----------------
REF_SYMBOL = os.getenv("REF_SYMBOL", "ETH-USD")
REF_DATE = os.getenv("REF_DATE", "2024-11-05")

# توجه: با تغییر به فضای 5 بُعدی، آستانه باید بازکالیبره شود.
# حداکثر فاصله ممکن ≈ 0.47 است. آستانه 0.15 یعنی ~32٪ از حداکثر.
THRESHOLD = float(os.getenv("THRESHOLD", "0.15"))
SIZE_TOL = float(os.getenv("SIZE_TOL", "2"))
HORIZON = int(os.getenv("HORIZON", "60"))
MIN_SAMPLES = int(os.getenv("MIN_SAMPLES", "10"))
MIN_RANGE_PCT = float(os.getenv("MIN_RANGE_PCT", "0.001"))  # حداقل رنج نسبت به open

# وزن ویژگی‌ها (جمعشان = 1.0)
WEIGHTS = np.array([
    float(os.getenv("W_OPEN",   "0.25")),   # موقعیت open
    float(os.getenv("W_CLOSE",  "0.30")),   # موقعیت close (مهم‌تر)
    float(os.getenv("W_UPWICK", "0.15")),   # سایه بالایی
    float(os.getenv("W_LOWICK", "0.15")),   # سایه پایینی
    float(os.getenv("W_DIR",    "0.15")),   # جهت بدنه (صعودی/نزولی)
])

COINS = [
    "BTC-USD", "ETH-USD", "XRP-USD", "BNB-USD", "SOL-USD",
    "DOGE-USD", "TRX-USD", "ADA-USD", "AVAX-USD", "LINK-USD",
]
TARGETS = [5, 10, 20, 30, 50, 100]
DDS = [2, 3, 4]

TG_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TG_CHAT = os.getenv("TELEGRAM_CHAT_ID")


def flatten(df):
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df


def features(o, h, l, c):
    """
    ویژگی‌های یک کندل (همه بین 0 و 1):
      [0] موقعیت open در رنج
      [1] موقعیت close در رنج
      [2] سایه بالایی
      [3] سایه پایینی
      [4] جهت بدنه (1=صعودی، 0=نزولی)
    """
    r = h - l
    if r <= 0:
        return None
    o_pos = (o - l) / r
    c_pos = (c - l) / r
    up_wick = (h - max(o, c)) / r
    lo_wick = (min(o, c) - l) / r
    direction = 1.0 if c >= o else 0.0
    return np.array([o_pos, c_pos, up_wick, lo_wick, direction], dtype=float)


def batch_features(o, h, l, c):
    """نسخه برداری‌شده features برای آرایه‌های numpy"""
    rng = h - l
    valid = rng > 0
    safe = np.where(valid, rng, 1.0)
    o_pos = (o - l) / safe
    c_pos = (c - l) / safe
    up_wick = (h - np.maximum(o, c)) / safe
    lo_wick = (np.minimum(o, c) - l) / safe
    direction = (c >= o).astype(float)
    feats = np.column_stack([o_pos, c_pos, up_wick, lo_wick, direction])
    feats[~valid] = np.nan
    return feats


def get_reference():
    d = pd.Timestamp(REF_DATE)
    df = yf.download(REF_SYMBOL, start=d, end=d + pd.Timedelta(days=1),
                     interval="1d", progress=False, auto_adjust=False)
    df = flatten(df)
    if df.empty:
        raise RuntimeError("کندل مرجع از یاهو دریافت نشد")
    r = df.iloc[0]
    o, h, l, c = float(r["Open"]), float(r["High"]), float(r["Low"]), float(r["Close"])
    ref_feat = features(o, h, l, c)
    if ref_feat is None:
        raise RuntimeError("کندل مرجع رنج صفر دارد")
    return ref_feat, (o, h, l, c)


def get_4h(symbol):
    # یاهو 4 ساعته ندارد؛ از 1 ساعته (حداکثر ~2 سال) ساخته می‌شود
    df = yf.download(symbol, period="729d", interval="1h",
                     progress=False, auto_adjust=False)
    df = flatten(df)
    if df.empty:
        return None
    df = df.resample("4h").agg(
        {"Open": "first", "High": "max", "Low": "min", "Close": "last"}
    ).dropna()
    return df


def analyze(df, ref_feat, ref_rng_pct):
    o, h, l, c = (df[k].values.astype(float) for k in ["Open", "High", "Low", "Close"])
    n = len(df)

    # حذف کندل‌های با رنج خیلی کوچک (نویز)
    rng = h - l
    valid_range = (rng / np.where(o == 0, np.nan, o)) >= MIN_RANGE_PCT
    valid_range &= rng > 0

    feats = batch_features(o, h, l, c)
    diff = feats - ref_feat                        # shape (n, 5)
    weighted = diff * WEIGHTS                       # وزن‌دهی
    dist = np.linalg.norm(weighted, axis=1)         # فاصله اقلیدسی وزن‌دار
    dist = np.nan_to_num(dist, nan=9.0)

    mask = (dist <= THRESHOLD) & valid_range

    if SIZE_TOL > 0:
        rp = rng / np.where(o == 0, np.nan, o)
        mask &= (rp >= ref_rng_pct / SIZE_TOL) & (rp <= ref_rng_pct * SIZE_TOL)

    mask = np.nan_to_num(mask, nan=False).astype(bool)
    idx = np.where(mask)[0]

    found = len(idx)   # همه کندل‌های مشابه
    total = 0          # فقط آن‌هایی که افق کامل دارند
    succ = {(t, d): 0 for t in TARGETS for d in DDS}

    for i in idx:
        if i + HORIZON >= n:
            continue
        total += 1
        entry = c[i]
        fh = h[i + 1:i + 1 + HORIZON]
        fl = l[i + 1:i + 1 + HORIZON]
        runmin = np.minimum.accumulate(fl)
        for t in TARGETS:
            hits = np.where(fh >= entry * (1 + t / 100))[0]
            if len(hits) == 0:
                continue
            worst = runmin[hits[0]]
            for d in DDS:
                if worst > entry * (1 - d / 100):
                    succ[(t, d)] += 1

    return found, total, succ


def pct(a, b):
    return 0.0 if b == 0 else 100.0 * a / b


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
    ref_feat, (ro, rh, rl, rc) = get_reference()
    ref_rng_pct = (rh - rl) / ro

    grand_found = 0
    grand_total = 0
    grand = {(t, d): 0 for t in TARGETS for d in DDS}
    per_coin = {}

    for sym in COINS:
        try:
            df = get_4h(sym)
            if df is None or len(df) < HORIZON + 10:
                print(f"{sym}: داده کافی نیست")
                continue
            found, total, succ = analyze(df, ref_feat, ref_rng_pct)
            per_coin[sym] = (found, total, succ)
            grand_found += found
            grand_total += total
            for k, v in succ.items():
                grand[k] += v
        except Exception as e:
            print(f"{sym}: خطا {e}")

    lines = [
        f"کندل مرجع: {REF_SYMBOL} {REF_DATE} (روزانه)",
        f"O={ro:.2f} H={rh:.2f} L={rl:.2f} C={rc:.2f}",
        f"رنج: {ref_rng_pct*100:.2f}%  |  جهت: {'صعودی' if rc>=ro else 'نزولی'}",
        f"ویژگی مرجع (o_pos, c_pos, up_wick, lo_wick, dir): "
        f"{ref_feat[0]:.2f}, {ref_feat[1]:.2f}, {ref_feat[2]:.2f}, "
        f"{ref_feat[3]:.2f}, {ref_feat[4]:.0f}",
        f"وزن‌ها: open={WEIGHTS[0]} close={WEIGHTS[1]} up_w={WEIGHTS[2]} "
        f"lo_w={WEIGHTS[3]} dir={WEIGHTS[4]}",
        f"آستانه فاصله وزن‌دار: {THRESHOLD} | فیلتر اندازه: x{SIZE_TOL} | "
        f"افق: {HORIZON} کندل 4h | حداقل رنج: {MIN_RANGE_PCT*100:.2f}%",
        "",
        f"کندل مشابه پیدا شده: {grand_found}",
        f"استفاده شده در محاسبه (افق کامل دارند): {grand_total}",
        f"کنار گذاشته شده (افق ناقص): {grand_found - grand_total}",
        "",
        "تعداد پیدا شده در هر ارز:",
    ]
    for sym, (found, total, _) in per_coin.items():
        lines.append(f"{sym:<9} پیدا شده={found:<4} محاسبه={total}")
    lines.append("")

    if grand_total == 0:
        lines.append("نمونه‌ای برای محاسبه نبود. THRESHOLD یا SIZE_TOL را بزرگ‌تر کن.")
        send_telegram("\n".join(lines))
        return

    lines.append("درصد رسیدن به هدف قبل از ریزش بیش از حد مجاز:")
    lines.append("هدف |  ریزش<2%  ریزش<3%  ریزش<4%")
    for t in TARGETS:
        row = "  ".join(f"{pct(grand[(t, d)], grand_total):6.1f}%" for d in DDS)
        lines.append(f"{t:>3}% | {row}")
    lines.append("")

    ranking = []
    for sym, (found, total, succ) in per_coin.items():
        if total < MIN_SAMPLES:
            continue
        avg = np.mean([pct(v, total) for v in succ.values()])
        ranking.append((avg, sym, total, pct(succ[(5, 2)], total)))
    ranking.sort(reverse=True)

    if ranking:
        lines.append("رتبه ارزها (میانگین موفقیت همه سناریوها):")
        for avg, sym, total, r52 in ranking:
            lines.append(f"{sym:<9} n={total:<4} میانگین={avg:5.1f}%  هدف5/ریزش2={r52:5.1f}%")
        lines.append("")
        lines.append(f"بهترین ارز: {ranking[0][1]}")
    else:
        lines.append(f"هیچ ارزی حداقل {MIN_SAMPLES} نمونه نداشت.")

    send_telegram("\n".join(lines))


if __name__ == "__main__":
    main()
