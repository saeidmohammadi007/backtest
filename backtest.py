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

# 🕯 تاریخ کندل مرجع روی تایم‌فریم روزانه (فرمت YYYY-MM-DD)
TARGET_DATE   = "2025-01-15"

# ─── داده‌ی کاندیدها (4h) ───
SEARCH_PERIOD = "730d"      # حداکثر بازه‌ی مجاز برای دیتای 1h
INTERVAL_SRC  = "1h"        # داده‌ی خام
RESAMPLE_RULE = "4h"        # تبدیل به 4h
PATTERN_LEN   = 1

SYMBOLS = [
    "BTC-USD", "ETH-USD", "LTC-USD", "XRP-USD",
    "XMR-USD", "ZEC-USD", "DASH-USD", "ETC-USD",
    "XLM-USD", "REP-USD",
]

# ─── بازه‌ی کاندیدها (بر مبنای تاریخ کندل 4h) ───
MIN_CANDIDATE_DATE = "2024-10-01"
MAX_CANDIDATE_DATE = "2026-12-31"

ALLOW_CANDIDATES_AFTER_TARGET = True
EXCLUDE_EXACT_TARGET_CANDLE   = True   # تمام کندل‌های 4h هم‌روز با کندل روزانه‌ی مرجع (در همان نماد) حذف می‌شوند
EXCLUDE_SAME_MONTH_AS_TARGET  = False

# ─── حالت ورود ───
ENTRY_MODE = "next_open"   # "next_open" یا "ref_close"

# ─── پارامترهای جستجو (همه بر حسب کندل 4h) ───
LOOKAHEAD              = 42    # 42 کندل 4h ≈ 7 روز
K_PER_SYMBOL           = 20
TOTAL_TOP              = 100
EXCLUDE_TEMPORAL_HOURS = 168   # 7 روز

TARGETS       = [20.0, 50.0]
DD_THRESHOLDS = [2.0, 3.0]

# حداکثر انحراف مجاز کندل روزانه‌ی مرجع (روز)
MAX_DATE_DRIFT_DAYS = 1

ONE_CANDIDATE_PER_WEEK = True


# ==================== دریافت داده ====================
def fetch_daily(symbol):
    """دریافت روزانه‌ی خام — فقط برای استخراج کندل مرجع."""
    log(f"📥 دریافت روزانه {symbol}")
    try:
        df = yf.download(symbol, interval="1d", period="max",
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
        return df if not df.empty else None
    except Exception as e:
        log(f"❌ fetch_daily {symbol}: {e}")
        traceback.print_exc()
        return None


def fetch_4h(symbol):
    """دریافت 1h از yfinance و رزمپل به 4h — برای کاندیدها."""
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
def find_similar(target_feat, target_day, dfs):
    """
    target_day: تاریخ روزانه‌ی کندل مرجع (Timestamp نرمال‌شده).
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
            mask = mask & (df.index < target_day)

        # 🕯 حذف روزِ کندل مرجع در همان نماد هدف
        if EXCLUDE_EXACT_TARGET_CANDLE and sym == TARGET_SYMBOL:
            same_day = (df.index.normalize() == target_day)
            mask = mask & (~same_day)

        if EXCLUDE_SAME_MONTH_AS_TARGET:
            same_month = (
                (df.index.year  == target_day.year) &
                (df.index.month == target_day.month)
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
    lines.append(f"🕯 <b>مرجع روزانه {TARGET_SYMBOL} → کاندید ۴ساعته</b>")
    lines.append("━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    lines.append(f"📅 کندل مرجع (روزانه): <code>{target_info['date']}</code>")
    lines.append(f"   O=<code>{target_info['o']:.4f}</code> "
                 f"H=<code>{target_info['h']:.4f}</code> "
                 f"L=<code>{target_info['l']:.4f}</code> "
                 f"C=<code>{target_info['c']:.4f}</code>")
    lines.append(f"   بدنه <code>{target_info['body']*100:+.2f}٪</code> | "
                 f"سایه‌بالا <code>{target_info['up']*100:.2f}٪</code> | "
                 f"سایه‌پایین <code>{target_info['dn']*100:.2f}٪</code> | "
                 f"دامنه <code>{target_info['rng']*100:.2f}٪</code>")
    lines.append(f"🔎 الگوهای مشابه (۴ساعته): <code>{n}</code> | "
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

    lines.append("📋 <b>۱۰ کندل ۴ساعته‌ی مشابه برتر:</b>")
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
    log(f"🚀 مرجع=روزانه {TARGET_SYMBOL} | کاندید=4h | ورود={ENTRY_MODE}")
    log(f"   بازه‌ی دریافت 4h: {SEARCH_PERIOD} | اینتروال خام: {INTERVAL_SRC}")
    log(f"   کف تاریخی کاندیدها: {MIN_CANDIDATE_DATE}")
    log(f"   سقف تاریخی کاندیدها: {MAX_CANDIDATE_DATE}")
    log(f"   کاندیدهای بعد از هدف: {ALLOW_CANDIDATES_AFTER_TARGET}")
    log("=" * 60)

    target_day = pd.Timestamp(TARGET_DATE).normalize()

    if target_day >= pd.Timestamp(MAX_CANDIDATE_DATE):
        raise RuntimeError(
            f"⛔ TARGET_DATE ({TARGET_DATE}) باید قبل از "
            f"MAX_CANDIDATE_DATE ({MAX_CANDIDATE_DATE}) باشد."
        )
    if pd.Timestamp(MIN_CANDIDATE_DATE) >= pd.Timestamp(MAX_CANDIDATE_DATE):
        raise RuntimeError(
            "⛔ MIN_CANDIDATE_DATE باید قبل از MAX_CANDIDATE_DATE باشد."
        )

    # ۱) کندل مرجع از داده‌ی روزانه
    df_daily = fetch_daily(TARGET_SYMBOL)
    if df_daily is None or len(df_daily) < 1:
        raise RuntimeError(f"داده‌ی روزانه‌ی {TARGET_SYMBOL} موجود نیست.")

    log(f"   محدوده‌ی داده‌ی روزانه: {df_daily.index[0].date()} → "
        f"{df_daily.index[-1].date()}")

    deltas = np.abs((df_daily.index - target_day).total_seconds())
    target_idx_d = int(np.argmin(deltas))
    actual_day = df_daily.index[target_idx_d]
    drift_days = abs((actual_day - target_day).days)

    if drift_days > MAX_DATE_DRIFT_DAYS:
        raise RuntimeError(
            f"⛔ کندل روزانه‌ی مرجع ({actual_day.date()}) "
            f"{drift_days} روز از TARGET_DATE فاصله دارد."
        )

    log(f"🎯 کندل مرجع روزانه: {actual_day.date()} "
        f"({actual_day.day_name()}) | انحراف: {drift_days} روز")

    o_d = df_daily["Open"].values.astype(float)
    h_d = df_daily["High"].values.astype(float)
    l_d = df_daily["Low"].values.astype(float)
    c_d = df_daily["Close"].values.astype(float)

    target_feat = pattern_features(target_idx_d, o_d, h_d, l_d, c_d)
    if target_feat is None:
        raise RuntimeError("ویژگی کندل روزانه‌ی مرجع محاسبه نشد.")

    log(f"   O={o_d[target_idx_d]:.4f} H={h_d[target_idx_d]:.4f} "
        f"L={l_d[target_idx_d]:.4f} C={c_d[target_idx_d]:.4f}")
    log(f"   بدنه={target_feat[0]*100:+.2f}٪ | "
        f"سایه‌بالا={target_feat[1]*100:.2f}٪ | "
        f"سایه‌پایین={target_feat[2]*100:.2f}٪ | "
        f"دامنه={target_feat[3]*100:.2f}٪")

    # ۲) داده‌ی 4h نمادها
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
    log("\n🔍 جستجوی الگوهای 4h مشابه با کندل روزانه‌ی مرجع...")
    candidates = find_similar(target_feat, actual_day, dfs)
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
        "date": actual_day.strftime("%Y-%m-%d (%a)"),
        "o":    o_d[target_idx_d],
        "h":    h_d[target_idx_d],
        "l":    l_d[target_idx_d],
        "c":    c_d[target_idx_d],
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
