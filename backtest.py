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

# 🕯 نماد مرجع (4h)
TARGET_SYMBOL = "API3-USD"

# 🕯 کندل مرجع روی تایم‌فریم 4h (فرمت YYYY-MM-DD HH:MM — UTC)
TARGET_DATE   = "2025-02-03 04:00"

# ─── داده‌ی کاندیدها (4h) ───
SEARCH_PERIOD = "730d"      # حداکثر بازه‌ی مجاز برای دیتای 1h
INTERVAL_SRC  = "1h"        # داده‌ی خام
RESAMPLE_RULE = "4h"        # تبدیل به 4h
PATTERN_LEN   = 1

SYMBOLS = [
    "API3-USD",
    "BTC-USD", "ETH-USD", "LTC-USD", "XRP-USD",
    "XMR-USD", "ZEC-USD", "DASH-USD", "ETC-USD",
    "XLM-USD", "REP-USD",
]

# ─── بازه‌ی کاندیدها (بر مبنای تاریخ کندل 4h) ───
MIN_CANDIDATE_DATE = "2024-10-01"
MAX_CANDIDATE_DATE = "2026-12-31"

ALLOW_CANDIDATES_AFTER_TARGET = True
EXCLUDE_EXACT_TARGET_CANDLE   = True   # کندل 4h دقیقاً برابر کندل مرجع در همان نماد حذف می‌شود
EXCLUDE_SAME_MONTH_AS_TARGET  = False

# ─── حالت ورود ───
ENTRY_MODE = "next_open"   # "next_open" یا "ref_close"

# ─── پارامترهای جستجو (همه بر حسب کندل 4h) ───
LOOKAHEAD              = 42    # 42 کندل 4h ≈ 7 روز
K_PER_SYMBOL           = 20
TOTAL_TOP              = 100
EXCLUDE_TEMPORAL_HOURS = 168   # 7 روز

TARGETS       = [2.0, 20.0, 50.0]        # ← 2.0 اضافه شد
DD_THRESHOLDS = [1.0, 2.0, 3.0]          # ← 1.0 اضافه شد

# حداکثر انحراف مجاز کندل مرجع 4h (ساعت)
MAX_DATE_DRIFT_HOURS = 4

ONE_CANDIDATE_PER_WEEK = True


# ==================== دریافت داده ====================
def fetch_4h(symbol):
    """دریافت 1h از yfinance و رزمپل به 4h."""
    log(f"📥 دریافت 4h {symbol}")
    try:
        df = yf.download(symbol, interval=INTERVAL_SRC, period=SEARCH_PERIOD,
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
        df.index = df.index.floor("h")

        df_4h = df.resample(RESAMPLE_RULE, label="left",
                            closed="left").agg({
            "Open":  "first",
            "High":  "max",
            "Low":   "min",
            "Close": "last",
        }).dropna()

        return df_4h if not df_4h.empty else None
    except Exception as e:
        log(f"❌ fetch_4h {symbol}: {e}")
        traceback.print_exc()
        return None


# ==================== ویژگی‌ها ====================
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


# ==================== تحلیل فوروارد (4h) ====================
def analyze_forward(entry_idx, o, h, l, c, n, lookahead):
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


# ==================== جستجوی مشابه‌ها (4h) ====================
def find_similar(target_feat, target_dt_actual, dfs):
    """
    target_dt_actual: تایم‌استمپ دقیق کندل 4h مرجع.
    کاندیدها کندل‌های 4h در بازه‌ی [MIN_CANDIDATE_DATE, MAX_CANDIDATE_DATE) هستند.
    """
    candidates_all = []
    floor_global = pd.Timestamp(MIN_CANDIDATE_DATE)
    cap_global   = pd.Timestamp(MAX_CANDIDATE_DATE)

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
        mask = df.index >= floor_global
        mask = mask & (df.index < cap_global)

        if not ALLOW_CANDIDATES_AFTER_TARGET:
            mask = mask & (df.index < target_dt_actual)

        # 🕯 حذف دقیقاً همان کندل 4h مرجع در همان نماد هدف
        if EXCLUDE_EXACT_TARGET_CANDLE and sym == TARGET_SYMBOL:
            same_candle = (df.index == target_dt_actual)
            mask = mask & (~same_candle)

        if EXCLUDE_SAME_MONTH_AS_TARGET:
            same_month = (
                (df.index.year  == target_dt_actual.year) &
                (df.index.month == target_dt_actual.month)
            )
            mask = mask & (~same_month)

        mask = mask & (idx_arr >= PATTERN_LEN - 1)
        mask = mask & (idx_arr <= n - LOOKAHEAD - 2)

        valid_idxs = np.where(mask)[0]
        if len(valid_idxs) == 0:
            log(f"⚠️ {sym}: هیچ کندل 4h معتبری در بازه‌ی مجاز ندارد → رد شد.")
            continue

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
    seen_weeks = set()

    for c_ in candidates_all:
        if ONE_CANDIDATE_PER_WEEK:
            iso = c_["date"].isocalendar()
            yw = (int(iso.year), int(iso.week))
            if yw in seen_weeks:
                continue

        too_close = False
        for u in unique_candidates:
            if u["symbol"] != c_["symbol"]:
                continue
            dt_hours = abs((u["date"] - c_["date"]).total_seconds()) / 3600.0
            if dt_hours < EXCLUDE_TEMPORAL_HOURS:
                too_close = True
                break
        if too_close:
            continue

        unique_candidates.append(c_)
        if ONE_CANDIDATE_PER_WEEK:
            iso = c_["date"].isocalendar()
            seen_weeks.add((int(iso.year), int(iso.week)))

        if len(unique_candidates) >= TOTAL_TOP:
            break

    log(f"🧹 بعد از حذف هم‌نمادهای نزدیک و هم‌هفته‌ها: "
        f"{len(unique_candidates)} الگو")
    return unique_candidates


# ==================== تلگرام ====================
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
    lines.append(f"🕯 <b>مرجع 4h {TARGET_SYMBOL} → کاندید 4h</b>")
    lines.append("━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    lines.append(f"📅 کندل مرجع (4h): <code>{target_info['date']}</code>")
    lines.append(f"   O=<code>{target_info['o']:.4f}</code> "
                 f"H=<code>{target_info['h']:.4f}</code> "
                 f"L=<code>{target_info['l']:.4f}</code> "
                 f"C=<code>{target_info['c']:.4f}</code>")
    lines.append(f"   بدنه <code>{target_info['body']*100:+.2f}٪</code> | "
                 f"سایه‌بالا <code>{target_info['up']*100:.2f}٪</code> | "
                 f"سایه‌پایین <code>{target_info['dn']*100:.2f}٪</code> | "
                 f"دامنه <code>{target_info['rng']*100:.2f}٪</code>")
    lines.append(f"🔎 الگوهای مشابه (4h): <code>{n}</code> | "
                 f"Lookahead: <code>{LOOKAHEAD}×4h</code> | "
                 f"ورود: <code>{ENTRY_MODE}</code>")
    lines.append(f"🗓 بازه‌ی کاندیدها: "
                 f"<code>{MIN_CANDIDATE_DATE} → {MAX_CANDIDATE_DATE}</code>")
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

    lines.append("📋 <b>۱۰ کندل 4h مشابه برتر:</b>")
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

    # 🎯 هایلایت اختصاصی: رشد ۲٪ با ریزش حداکثر ۱٪
    hit_2_dd1  = [r for r in results if r.get("hit_2.0_dd1.0", False)]
    pct_2_dd1  = len(hit_2_dd1) / n * 100.0
    base_2_dd1 = baseline.get((2.0, 1.0), 0.0)
    edge_2_dd1 = pct_2_dd1 - base_2_dd1
    exp_val    = pct_2_dd1 * 2.0 - (100.0 - pct_2_dd1) * 1.0

    lines.append("")
    lines.append("━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    lines.append("🎯 <b>رشد +۲٪ با ریزش زیر ۱٪</b>")
    lines.append(
        f"├ احتمال: <code>{pct_2_dd1:.1f}٪</code> "
        f"(پایه <code>{base_2_dd1:.1f}٪</code> | "
        f"edge <code>{edge_2_dd1:+.1f}٪</code>)"
    )
    lines.append(f"├ تعداد موارد: <code>{len(hit_2_dd1)}</code> از <code>{n}</code>")
    lines.append(f"├ نسبت ریسک/ریوارد: <code>1:2</code>")
    lines.append(f"└ امید ریاضی: <code>{exp_val:+.1f}٪</code>")
    if hit_2_dd1:
        lines.append("")
        lines.append("   <b>موارد موفق:</b>")
        for r in hit_2_dd1[:15]:
            lines.append(f"   │ <b>{r['symbol']}</b> @ {r['date']}")
        if len(hit_2_dd1) > 15:
            lines.append(f"   │ ... و {len(hit_2_dd1) - 15} مورد دیگر")

    return "\n".join(lines).rstrip()


# ==================== اجرای اصلی ====================
def main():
    log("=" * 60)
    log(f"🚀 مرجع=4h {TARGET_SYMBOL}@{TARGET_DATE} | کاندید=4h | ورود={ENTRY_MODE}")
    log(f"   بازه‌ی دریافت 4h: {SEARCH_PERIOD} | اینتروال خام: {INTERVAL_SRC}")
    log(f"   کف تاریخی کاندیدها: {MIN_CANDIDATE_DATE}")
    log(f"   سقف تاریخی کاندیدها: {MAX_CANDIDATE_DATE}")
    log(f"   کاندیدهای بعد از هدف: {ALLOW_CANDIDATES_AFTER_TARGET}")
    log("=" * 60)

    target_dt = pd.Timestamp(TARGET_DATE)

    if target_dt >= pd.Timestamp(MAX_CANDIDATE_DATE):
        raise RuntimeError(
            f"⛔ TARGET_DATE ({TARGET_DATE}) باید قبل از "
            f"MAX_CANDIDATE_DATE ({MAX_CANDIDATE_DATE}) باشد."
        )
    if pd.Timestamp(MIN_CANDIDATE_DATE) >= pd.Timestamp(MAX_CANDIDATE_DATE):
        raise RuntimeError(
            "⛔ MIN_CANDIDATE_DATE باید قبل از MAX_CANDIDATE_DATE باشد."
        )

    # ۱) دریافت داده‌ی 4h همه‌ی نمادها (شامل نماد مرجع)
    dfs = {}
    for sym in SYMBOLS:
        df = fetch_4h(sym)
        if df is not None and len(df) > LOOKAHEAD + 10:
            dfs[sym] = df
            log(f"   ↳ {sym}: {len(df)} کندل 4h "
                f"({df.index[0]} → {df.index[-1]})")

    missing = [s for s in SYMBOLS if s not in dfs]
    if missing:
        log(f"⚠️ بدون داده: {missing}")
    if not dfs:
        raise RuntimeError("هیچ نمادی داده‌ی 4h ندارد.")

    if TARGET_SYMBOL not in dfs:
        raise RuntimeError(f"⛔ داده‌ی 4h برای {TARGET_SYMBOL} موجود نیست.")

    # ۲) یافتن کندل مرجع 4h در داده‌ی همان نماد
    df_target = dfs[TARGET_SYMBOL]
    log(f"   محدوده‌ی داده‌ی 4h {TARGET_SYMBOL}: "
        f"{df_target.index[0]} → {df_target.index[-1]}")

    deltas = np.abs((df_target.index - target_dt).total_seconds())
    target_idx_4h = int(np.argmin(deltas))
    actual_dt = df_target.index[target_idx_4h]
    drift_hours = abs((actual_dt - target_dt).total_seconds()) / 3600.0

    if drift_hours > MAX_DATE_DRIFT_HOURS:
        raise RuntimeError(
            f"⛔ نزدیک‌ترین کندل 4h ({actual_dt}) "
            f"{drift_hours:.1f} ساعت از TARGET_DATE ({target_dt}) فاصله دارد."
        )

    log(f"🎯 کندل مرجع 4h: {actual_dt} "
        f"({actual_dt.day_name()}) | انحراف: {drift_hours:.1f} ساعت")

    o_t = df_target["Open"].values.astype(float)
    h_t = df_target["High"].values.astype(float)
    l_t = df_target["Low"].values.astype(float)
    c_t = df_target["Close"].values.astype(float)

    target_feat = pattern_features(target_idx_4h, o_t, h_t, l_t, c_t)
    if target_feat is None:
        raise RuntimeError("ویژگی کندل 4h مرجع محاسبه نشد.")

    log(f"   O={o_t[target_idx_4h]:.4f} H={h_t[target_idx_4h]:.4f} "
        f"L={l_t[target_idx_4h]:.4f} C={c_t[target_idx_4h]:.4f}")
    log(f"   بدنه={target_feat[0]*100:+.2f}٪ | "
        f"سایه‌بالا={target_feat[1]*100:.2f}٪ | "
        f"سایه‌پایین={target_feat[2]*100:.2f}٪ | "
        f"دامنه={target_feat[3]*100:.2f}٪")

    # ۳) نرخ پایه (4h)
    log("\n📊 محاسبه‌ی نرخ پایه (4h)...")
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

    # ۴) جستجوی الگوهای 4h مشابه
    log("\n🔍 جستجوی الگوهای 4h مشابه با کندل 4h مرجع...")
    candidates = find_similar(target_feat, actual_dt, dfs)
    if not candidates:
        raise RuntimeError("هیچ الگوی مشابهی یافت نشد.")

    # ۵) تحلیل فوروارد روی 4h
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
        r["date"]   = c_["date"].strftime("%Y-%m-%d %H:%M (%a)")
        r["dist"]   = c_["dist"]
        r["feat"]   = pattern_features(c_["idx"], o, h, l, cc)
        results.append(r)

    if not results:
        raise RuntimeError("نتیجه‌ای محاسبه نشد.")

    results.sort(key=lambda x: x["dist"])

    target_info = {
        "date": actual_dt.strftime("%Y-%m-%d %H:%M (%a)"),
        "o":    o_t[target_idx_4h],
        "h":    h_t[target_idx_4h],
        "l":    l_t[target_idx_4h],
        "c":    c_t[target_idx_4h],
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
