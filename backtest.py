import os
import sys
import traceback

import numpy as np
import pandas as pd
import requests
import yfinance as yf


def log(m):
    print(m, flush=True)


# ==================== تنظیمات ====================
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
TELEGRAM_CHAT_ID   = os.getenv("TELEGRAM_CHAT_ID", "").strip()

TARGET_SYMBOL = "DOGE-USD"
TARGET_DATE   = "2019-01-21"

SEARCH_PERIOD = "max"
PATTERN_LEN = 1

SYMBOLS = [
    "BTC-USD", "ETH-USD", "LTC-USD", "XRP-USD",
    "XMR-USD", "ZEC-USD", "DASH-USD", "ETC-USD",
    "XLM-USD", "REP-USD",
]

# ─── بازه‌ی کاندیدها ───
# کف تاریخی (yfinance قبل از این تاریخ سایه‌های کریپتو را ناقص می‌دهد)
MIN_CANDIDATE_DATE = "2019-01-01"

# کاندیدهای بعد از تاریخ هدف هم مجاز باشند
ALLOW_CANDIDATES_AFTER_TARGET = True

# فقط دقیقاً خودِ کندل مرجع حذف شود
EXCLUDE_EXACT_TARGET_CANDLE = True

# کل ماه هدف حذف نشود
EXCLUDE_SAME_MONTH_AS_TARGET = False

# ─── حالت ورود ───
#   "next_open" → ورود روی Open کندل هفته‌ی بعد (واقع‌گرایانه)
#   "ref_close" → ورود روی Close کندل الگو (خوش‌بینانه)
ENTRY_MODE = "next_open"

# ─── پارامترهای جستجو ───
LOOKAHEAD        = 26
K_PER_SYMBOL     = 20
TOTAL_TOP        = 100
EXCLUDE_TEMPORAL = 4

TARGETS       = [20.0, 50.0]
DD_THRESHOLDS = [2.0, 3.0]

WEEK_START = "MON"   # "MON" یا "SUN"

MAX_DATE_DRIFT_DAYS = 7

# از بین کاندیدها فقط یک کندل در هر ماه تقویمی
ONE_CANDIDATE_PER_MONTH = True


# ==================== توابع ====================
def fetch_weekly(symbol):
    """دریافت روزانه از yfinance و رزمپل به هفتگی."""
    log(f"📥 دریافت هفتگی {symbol}")
    try:
        df = yf.download(symbol, interval="1d", period=SEARCH_PERIOD,
                         progress=False, auto_adjust=True, timeout=60)
        if df is None or df.empty:
            return None
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df[["Open", "High", "Low", "Close"]].dropna()
        if df.empty:
            return None

        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
        if df.index.tz is not None:
            df.index = df.index.tz_convert("UTC").tz_localize(None)
        df.index = df.index.normalize()

        rule = "W-MON" if WEEK_START == "MON" else "W-SUN"
        df_wk = df.resample(rule, label="left", closed="left").agg({
            "Open":  "first",
            "High":  "max",
            "Low":   "min",
            "Close": "last",
        }).dropna()

        return df_wk if not df_wk.empty else None
    except Exception as e:
        log(f"❌ fetch_weekly {symbol}: {e}")
        traceback.print_exc()
        return None


def candle_features(i, o, h, l, c):
    op, hi, lo, cl = o[i], h[i], l[i], c[i]
    if op <= 0:
        return None
    body        = (cl - op) / op
    upper_wick  = (hi - max(op, cl)) / op
    lower_wick  = (min(op, cl) - lo) / op
    total_range = (hi - lo) / op
    return np.array([body, upper_wick, lower_wick, total_range], dtype=float)


def pattern_features(end_idx, o, h, l, c, pattern_len=PATTERN_LEN):
    start = end_idx - pattern_len + 1
    if start < 0:
        return None
    feats = []
    for j in range(start, end_idx + 1):
        f = candle_features(j, o, h, l, c)
        if f is None:
            return None
        feats.append(f)
    return np.concatenate(feats)


def analyze_forward(entry_idx, o, h, l, c, n, lookahead):
    """ورود طبق ENTRY_MODE، سپس سنجش بازده و drawdown."""
    if ENTRY_MODE == "next_open":
        if entry_idx + 1 >= n:
            return None
        entry_price = float(o[entry_idx + 1])
        start = entry_idx + 1
    else:  # ref_close
        entry_price = float(c[entry_idx])
        start = entry_idx + 1

    if entry_price <= 0:
        return None

    end = start + lookahead
    if end > n:
        return None

    fut_c = c[start:end]
    fut_h = h[start:end]
    fut_l = l[start:end]
    if len(fut_c) == 0:
        return None

    final_price = float(fut_c[-1])
    total_ret = (final_price - entry_price) / entry_price * 100.0

    out = {"ret": total_ret}
    for T in TARGETS:
        target_price = entry_price * (1 + T / 100.0)
        hits = np.where(fut_h >= target_price)[0]
        if len(hits) == 0:
            out[f"hit_{T}"] = False
            for D in DD_THRESHOLDS:
                out[f"hit_{T}_dd{D}"] = False
        else:
            out[f"hit_{T}"] = True
            k = int(hits[0])
            dd = (float(np.min(fut_l[:k + 1])) - entry_price) / entry_price * 100.0
            for D in DD_THRESHOLDS:
                out[f"hit_{T}_dd{D}"] = dd >= -D
    return out


def find_similar_weekly(target_feat, target_date_naive, dfs):
    """
    جستجوی الگوهای مشابه با استفاده از mask بولی.
    کاندیدها می‌توانند قبل یا بعد از تاریخ هدف باشند
    (به‌جز خودِ کندل مرجع در صورت EXCLUDE_EXACT_TARGET_CANDLE).
    """
    candidates_all = []
    sym_floor_global = pd.Timestamp(MIN_CANDIDATE_DATE)

    for sym in SYMBOLS:
        df = dfs.get(sym)
        if df is None:
            continue

        o = df["Open"].values.astype(float)
        h = df["High"].values.astype(float)
        l = df["Low"].values.astype(float)
        c = df["Close"].values.astype(float)
        n = len(df)

        idx_arr = np.arange(n)

        # ─── کف تاریخی ───
        mask = df.index >= sym_floor_global

        # ─── آیا فقط قبل از تاریخ هدف؟ ───
        if not ALLOW_CANDIDATES_AFTER_TARGET:
            mask = mask & (df.index < target_date_naive)

        # ─── حذف دقیقاً خودِ کندل مرجع ───
        if EXCLUDE_EXACT_TARGET_CANDLE:
            mask = mask & (df.index != target_date_naive)

        # ─── حذف هم‌ماه (اختیاری) ───
        if EXCLUDE_SAME_MONTH_AS_TARGET:
            same_month = (
                (df.index.year  == target_date_naive.year) &
                (df.index.month == target_date_naive.month)
            )
            mask = mask & (~same_month)

        # ─── باید حداقل PATTERN_LEN کندل قبل داشته باشد ───
        mask = mask & (idx_arr >= PATTERN_LEN - 1)

        # ─── باید LOOKAHEAD+1 کندل بعد داشته باشد (برای next_open) ───
        mask = mask & (idx_arr <= n - LOOKAHEAD - 2)

        valid_idxs = np.where(mask)[0]
        if len(valid_idxs) == 0:
            log(f"⚠️ {sym}: هیچ کندل معتبری در بازه‌ی مجاز ندارد → رد شد.")
            continue

        # ساخت فیچرها فقط برای ایندکس‌های معتبر
        feats, idxs = [], []
        for i in valid_idxs:
            f = pattern_features(int(i), o, h, l, c)
            if f is None:
                continue
            feats.append(f)
            idxs.append(int(i))

        if not feats:
            continue

        feats = np.array(feats)
        dists = np.sqrt(np.sum((feats - target_feat) ** 2, axis=1))

        k = min(K_PER_SYMBOL * 5, len(idxs))
        order = np.argsort(dists)[:k]
        for oi in order:
            cand_idx = int(idxs[oi])
            candidates_all.append({
                "symbol": sym,
                "idx":    cand_idx,
                "dist":   float(dists[oi]),
                "date":   df.index[cand_idx],
            })

    candidates_all.sort(key=lambda x: x["dist"])

    unique_candidates = []
    seen_months = set()

    for c_ in candidates_all:
        if ONE_CANDIDATE_PER_MONTH:
            ym = (c_["date"].year, c_["date"].month)
            if ym in seen_months:
                continue

        too_close = False
        for u in unique_candidates:
            if u["symbol"] != c_["symbol"]:
                continue
            if abs((u["date"] - c_["date"]).days) < EXCLUDE_TEMPORAL * 7:
                too_close = True
                break
        if too_close:
            continue

        unique_candidates.append(c_)
        if ONE_CANDIDATE_PER_MONTH:
            seen_months.add((c_["date"].year, c_["date"].month))

        if len(unique_candidates) >= TOTAL_TOP:
            break

    log(f"🧹 بعد از حذف هم‌نمادهای نزدیک و هم‌ماه‌ها: "
        f"{len(unique_candidates)} الگو")
    return unique_candidates


def send_message(text, chunk_size=3500):
    if not (TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID):
        log("⚠️ توکن یا chat_id موجود نیست.")
        return
    lines = text.split("\n")
    chunks, cur = [], ""
    for ln in lines:
        if len(cur) + len(ln) + 1 > chunk_size and cur:
            chunks.append(cur)
            cur = ln
        else:
            cur = (cur + "\n" + ln) if cur else ln
    if cur:
        chunks.append(cur)

    for i, chunk in enumerate(chunks, 1):
        try:
            r = requests.post(
                f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage",
                json={"chat_id": TELEGRAM_CHAT_ID, "text": chunk,
                      "parse_mode": "HTML"},
                timeout=30,
            )
            log(f"📤 telegram part {i}/{len(chunks)}: {r.status_code}")
            if r.status_code != 200:
                log(f"   err: {r.text[:200]}")
        except Exception as e:
            log(f"❌ send part {i}: {e}")


# ==================== گزارش ====================
def build_report(results, target_info, baseline):
    n = len(results)
    if n == 0:
        return "بدون داده"

    wins = sum(1 for r in results if r["ret"] > 0)
    win_pct = wins / n * 100.0

    lines = []
    lines.append(f"🕯 <b>الگوی هفتگی {TARGET_SYMBOL}</b>")
    lines.append("━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    lines.append(f"📅 کندل مرجع: <code>{target_info['date']}</code>")
    lines.append(f"   O=<code>{target_info['o']:.4f}</code> "
                 f"H=<code>{target_info['h']:.4f}</code> "
                 f"L=<code>{target_info['l']:.4f}</code> "
                 f"C=<code>{target_info['c']:.4f}</code>")
    lines.append(f"   بدنه <code>{target_info['body']*100:+.2f}٪</code> | "
                 f"سایه‌بالا <code>{target_info['up']*100:.2f}٪</code> | "
                 f"سایه‌پایین <code>{target_info['dn']*100:.2f}٪</code> | "
                 f"دامنه <code>{target_info['rng']*100:.2f}٪</code>")
    lines.append(f"🔎 الگوهای مشابه: <code>{n}</code> | "
                 f"Lookahead: <code>{LOOKAHEAD}w</code> | "
                 f"ورود: <code>{ENTRY_MODE}</code>")
    lines.append(f"🗓 کف تاریخی: <code>{MIN_CANDIDATE_DATE}</code>")
    lines.append("")

    lines.append(f"💰 <b>سوددهی کلی:</b> <code>{win_pct:.1f}٪</code> "
                 f"(پایه <code>{baseline['win']:.1f}٪</code> | "
                 f"edge <code>{win_pct - baseline['win']:+.1f}٪</code>)")
    lines.append("")

    for D in DD_THRESHOLDS:
        lines.append(f"📉 <b>ریزش زیر {D:.0f}٪</b>")
        for T in TARGETS:
            cnt = sum(1 for r in results if r[f"hit_{T}_dd{D}"])
            pct = cnt / n * 100.0
            base = baseline[(T, D)]
            lines.append(
                f"├ +{T:.0f}٪: <code>{pct:.1f}٪</code> "
                f"(پایه {base:.1f}٪ | edge <code>{pct - base:+.1f}٪</code>)"
            )
        lines.append("")

    lines.append("📋 <b>۱۰ کندل هفتگی مشابه برتر:</b>")
    for r in results[:10]:
        f0 = r["feat"][0]
        lines.append(
            f"├ <b>{r['symbol']}</b> @ <code>{r['date']}</code> | "
            f"d=<code>{r['dist']:.4f}</code> | "
            f"بدنه <code>{f0*100:+.2f}٪</code>"
        )

    hit_50_dd2 = [r for r in results if r.get("hit_50.0_dd2.0", False)]
    hit_50_dd3 = [r for r in results if r.get("hit_50.0_dd3.0", False)]

    lines.append("")
    lines.append(f"✅ <b>+۵۰٪ / DD≤۲٪</b> — {len(hit_50_dd2)} مورد:")
    if hit_50_dd2:
        for r in hit_50_dd2:
            lines.append(f"│ <b>{r['symbol']}</b> @ {r['date']}")
    else:
        lines.append("│ (هیچ)")

    lines.append("")
    lines.append(f"✅ <b>+۵۰٪ / DD≤۳٪</b> — {len(hit_50_dd3)} مورد:")
    if hit_50_dd3:
        for r in hit_50_dd3:
            lines.append(f"│ <b>{r['symbol']}</b> @ {r['date']}")
    else:
        lines.append("│ (هیچ)")

    return "\n".join(lines).rstrip()


# ==================== اجرای اصلی ====================
def main():
    log("=" * 60)
    log(f"🚀 مشابه‌یابی: هدف=هفتگی {TARGET_SYMBOL} | ورود={ENTRY_MODE}")
    log(f"   کف تاریخی کاندیدها: {MIN_CANDIDATE_DATE}")
    log(f"   کاندیدهای بعد از هدف: {ALLOW_CANDIDATES_AFTER_TARGET}")
    log("=" * 60)

    # ۱) کندل مرجع هفتگی
    df_target = fetch_weekly(TARGET_SYMBOL)
    if df_target is None or len(df_target) < PATTERN_LEN:
        raise RuntimeError(f"داده‌ی هفتگی {TARGET_SYMBOL} موجود نیست.")

    log(f"   محدوده‌ی داده: {df_target.index[0].date()} → "
        f"{df_target.index[-1].date()}")

    deltas = np.abs((df_target.index - pd.Timestamp(TARGET_DATE)).total_seconds())
    target_idx = int(np.argmin(deltas))
    actual_date = df_target.index[target_idx]
    drift_days = abs((actual_date - pd.Timestamp(TARGET_DATE)).days)

    if drift_days > MAX_DATE_DRIFT_DAYS:
        raise RuntimeError(
            f"⛔ کندل مرجع ({actual_date.date()}) "
            f"{drift_days} روز از TARGET_DATE فاصله دارد."
        )

    log(f"🎯 کندل مرجع هفتگی: {actual_date.date()} ({actual_date.day_name()}) | "
        f"انحراف: {drift_days} روز")

    o_t = df_target["Open"].values.astype(float)
    h_t = df_target["High"].values.astype(float)
    l_t = df_target["Low"].values.astype(float)
    c_t = df_target["Close"].values.astype(float)

    target_feat = pattern_features(target_idx, o_t, h_t, l_t, c_t)
    if target_feat is None:
        raise RuntimeError("ویژگی کندل مرجع محاسبه نشد.")

    log(f"   O={o_t[target_idx]:.4f} H={h_t[target_idx]:.4f} "
        f"L={l_t[target_idx]:.4f} C={c_t[target_idx]:.4f}")
    log(f"   بدنه={target_feat[0]*100:+.2f}٪ | "
        f"سایه‌بالا={target_feat[1]*100:.2f}٪ | "
        f"سایه‌پایین={target_feat[2]*100:.2f}٪ | "
        f"دامنه={target_feat[3]*100:.2f}٪")

    target_date_naive = actual_date

    # ۲) داده‌ی هفتگی نمادها
    dfs = {}
    for sym in SYMBOLS:
        df = fetch_weekly(sym)
        if df is not None and len(df) > LOOKAHEAD + 10:
            dfs[sym] = df
            log(f"   ↳ {sym}: {len(df)} کندل "
                f"({df.index[0].date()} → {df.index[-1].date()})")

    missing = [s for s in SYMBOLS if s not in dfs]
    if missing:
        log(f"⚠️ بدون داده: {missing}")
    if not dfs:
        raise RuntimeError("هیچ نمادی داده‌ی هفتگی ندارد.")

    # ۳) نرخ پایه
    log("\n📊 محاسبه‌ی نرخ پایه (هفتگی)...")
    all_outcomes = []
    for sym, df in dfs.items():
        o = df["Open"].values.astype(float)
        h = df["High"].values.astype(float)
        l = df["Low"].values.astype(float)
        c = df["Close"].values.astype(float)
        n = len(df)
        for i in range(n):
            out = analyze_forward(i, o, h, l, c, n, LOOKAHEAD)
            if out is not None:
                all_outcomes.append(out)

    n_base = len(all_outcomes)
    if n_base == 0:
        raise RuntimeError("نمونه‌ی پایه صفر است.")
    baseline = {
        "win": sum(1 for o in all_outcomes if o["ret"] > 0) / n_base * 100.0
    }
    for T in TARGETS:
        for D in DD_THRESHOLDS:
            cnt = sum(1 for o in all_outcomes if o[f"hit_{T}_dd{D}"])
            baseline[(T, D)] = cnt / n_base * 100.0

    log(f"   نمونه‌های پایه: {n_base}")
    log(f"   سوددهی پایه: {baseline['win']:.2f}٪")
    for T in TARGETS:
        for D in DD_THRESHOLDS:
            log(f"   +{T:.0f}٪ / DD≤{D:.0f}٪: {baseline[(T, D)]:.2f}٪")

    # ۴) جستجوی الگوهای مشابه
    log("\n🔍 جستجوی الگوهای هفتگی مشابه...")
    candidates = find_similar_weekly(target_feat, target_date_naive, dfs)
    if not candidates:
        raise RuntimeError("هیچ الگوی مشابهی یافت نشد.")

    # ۵) تحلیل فوروارد
    results = []
    for c_ in candidates:
        sym = c_["symbol"]
        df = dfs[sym]
        o = df["Open"].values.astype(float)
        h = df["High"].values.astype(float)
        l = df["Low"].values.astype(float)
        cc = df["Close"].values.astype(float)
        n = len(df)
        r = analyze_forward(c_["idx"], o, h, l, cc, n, LOOKAHEAD)
        if r is None:
            continue
        r["symbol"] = sym
        r["date"]   = c_["date"].strftime("%Y-%m-%d (%a)")
        r["dist"]   = c_["dist"]
        r["feat"]   = pattern_features(c_["idx"], o, h, l, cc)
        results.append(r)

    if not results:
        raise RuntimeError("نتیجه‌ای محاسبه نشد.")

    results.sort(key=lambda x: x["dist"])

    target_info = {
        "date": actual_date.strftime("%Y-%m-%d (%a)"),
        "o":    o_t[target_idx],
        "h":    h_t[target_idx],
        "l":    l_t[target_idx],
        "c":    c_t[target_idx],
        "body": target_feat[0],
        "up":   target_feat[1],
        "dn":   target_feat[2],
        "rng":  target_feat[3],
    }

    # ۶) گزارش
    report = build_report(results, target_info, baseline)
    plain = (report.replace("<b>", "").replace("</b>", "")
                   .replace("<code>", "").replace("</code>", ""))
    log("\n" + plain)
    send_message(report)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        log(f"❌ خطا: {e}")
        traceback.print_exc()
        sys.exit(1)
