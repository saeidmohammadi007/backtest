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

TARGET_SYMBOL = "ETH-USD"      # نماد هدف
TARGET_DATE   = "2025-04-24"    # تاریخ کندل مرجع (روزانه)

TARGET_INTERVAL = "1d"          # کندل مرجع
SEARCH_PERIOD_WEEKLY = "max"    # بازه‌ی داده‌ی هفتگی (max = کل تاریخچه)

# اگر True باشد فقط کندل‌های قبل از تاریخ مرجع بررسی می‌شوند.
STRICT_BEFORE_TARGET = False

PATTERN_LEN = 1                 # تعداد کندل الگو (روی کندل مرجع)

SYMBOLS = [
    "LUMIA-USD", "API3-USD", "FIL-USD",
    "DOGE-USD", "ONDO-USD", "SHIB-USD",
    "BLESS-USD", "ROSE-USD", "LINEA-USD", "JTO-USD",
]

LOOKAHEAD    = 30               # تعداد کندل هفتگی بعدی (~۷ ماه)
K_PER_SYMBOL = 20
TOTAL_TOP    = 100

TARGETS = [20.0, 50.0]          # اهداف سود (درصد)
DD_THRESHOLDS = [2.0, 3.0]      # آستانه‌ی ریزش (درصد)


# ==================== توابع ====================
def fetch(symbol, interval, period):
    log(f"📥 دریافت {symbol} ({interval})")
    try:
        df = yf.download(symbol, interval=interval, period=period,
                         progress=False, auto_adjust=True, timeout=60)
        if df is None or df.empty:
            return None
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        return df[['Open', 'High', 'Low', 'Close']].dropna()
    except Exception as e:
        log(f"❌ fetch {symbol}: {e}")
        return None


def fetch_weekly(symbol):
    log(f"📥 دریافت {symbol} (1wk)")
    try:
        df = yf.download(symbol, interval="1wk", period=SEARCH_PERIOD_WEEKLY,
                         progress=False, auto_adjust=True, timeout=60)
        if df is None or df.empty:
            return None
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df[['Open', 'High', 'Low', 'Close']].dropna()
        return df if not df.empty else None
    except Exception as e:
        log(f"❌ fetch_weekly {symbol}: {e}")
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
    if entry_idx + 1 + lookahead > n:
        return None
    entry_price = float(c[entry_idx])
    end = entry_idx + 1 + lookahead
    fut_c = c[entry_idx + 1:end]
    fut_h = h[entry_idx + 1:end]
    fut_l = l[entry_idx + 1:end]
    if len(fut_c) == 0:
        return None

    final_price = float(fut_c[-1])
    total_ret   = (final_price - entry_price) / entry_price * 100.0

    out = {'ret': total_ret}

    for T in TARGETS:
        target_price = entry_price * (1 + T / 100.0)
        hits = np.where(fut_h >= target_price)[0]
        if len(hits) == 0:
            out[f'hit_{T}'] = False
            for D in DD_THRESHOLDS:
                out[f'hit_{T}_dd{D}'] = False
        else:
            out[f'hit_{T}'] = True
            k = int(hits[0])
            dd = (float(np.min(fut_l[:k + 1])) - entry_price) / entry_price * 100.0
            for D in DD_THRESHOLDS:
                out[f'hit_{T}_dd{D}'] = dd >= -D
    return out


def find_similar_candles(target_feat, target_date_naive, dfs):
    candidates_all = []
    skipped_by_date = []

    for sym in SYMBOLS:
        df = dfs.get(sym)
        if df is None:
            continue
        o = df['Open'].values.astype(float)
        h = df['High'].values.astype(float)
        l = df['Low'].values.astype(float)
        c = df['Close'].values.astype(float)
        n = len(df)

        max_i = n - LOOKAHEAD - 1
        if max_i < PATTERN_LEN - 1:
            continue

        idx_naive = df.index
        if getattr(idx_naive, 'tz', None) is not None:
            idx_naive = idx_naive.tz_localize(None)

        mask_before = idx_naive < target_date_naive

        if mask_before.any():
            last_before = int(np.where(mask_before)[0][-1])
            max_i_strict = min(max_i, last_before)
            if max_i_strict >= PATTERN_LEN - 1:
                max_i = max_i_strict
            elif STRICT_BEFORE_TARGET:
                skipped_by_date.append(sym)
                continue
            else:
                log(f"⚠️ {sym}: کندل کافی قبل از {target_date_naive.date()} "
                    f"نیست → از کل داده استفاده می‌شود.")
        else:
            if STRICT_BEFORE_TARGET:
                skipped_by_date.append(sym)
                continue
            log(f"⚠️ {sym}: هیچ کندلی قبل از {target_date_naive.date()} "
                f"وجود ندارد → از کل داده استفاده می‌شود.")

        if max_i < PATTERN_LEN - 1:
            continue

        feats, idxs = [], []
        for i in range(PATTERN_LEN - 1, max_i + 1):
            f = pattern_features(i, o, h, l, c)
            if f is None:
                continue
            feats.append(f)
            idxs.append(i)

        if not feats:
            continue

        feats = np.array(feats)
        dists = np.sqrt(np.sum((feats - target_feat) ** 2, axis=1))

        k = min(K_PER_SYMBOL * 5, len(idxs))
        order = np.argsort(dists)[:k]
        for oi in order:
            cand_idx = int(idxs[oi])
            dt = df.index[cand_idx]
            if getattr(dt, 'tz', None) is not None:
                dt = dt.tz_localize(None)
            candidates_all.append({
                'symbol':   sym,
                'idx':      cand_idx,
                'dist':     float(dists[oi]),
                'date_key': dt.normalize(),
            })

    if skipped_by_date:
        log(f"⏭️  نمادهای رد شده به‌دلیل نبود داده‌ی قبل از تاریخ مرجع: "
            f"{skipped_by_date}")

    candidates_all.sort(key=lambda x: x['dist'])

    seen_dates = set()
    unique_candidates = []
    for c_ in candidates_all:
        key = (c_['symbol'], c_['date_key'])
        if key in seen_dates:
            continue
        seen_dates.add(key)
        unique_candidates.append(c_)
        if len(unique_candidates) >= TOTAL_TOP:
            break

    log(f"🧹 بعد از حذف هم‌تاریخی‌ها: {len(unique_candidates)} الگوی یکتا")
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
def build_report(results):
    n = len(results)
    if n == 0:
        return "بدون داده"

    wins = sum(1 for r in results if r['ret'] > 0)
    win_pct = wins / n * 100.0

    lines = []
    lines.append(f"🕯 <b>الگوی {TARGET_SYMBOL} | {TARGET_DATE}</b>")
    lines.append("━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    lines.append(f"🔎 تعداد الگوهای مشابه: <code>{n}</code>")
    lines.append("")
    lines.append(f"💰 <b>سوددهی کلی:</b> <code>{win_pct:.1f}٪</code> "
                 f"({wins} از {n})")
    lines.append("")

    for D in DD_THRESHOLDS:
        lines.append(f"📉 <b>ریزش زیر {D:.0f}٪</b>")
        for T in TARGETS:
            cnt = sum(1 for r in results if r[f'hit_{T}_dd{D}'])
            pct = cnt / n * 100.0
            lines.append(f"├ رسیدن به هدف +{T:.0f}٪: "
                         f"<code>{pct:.1f}٪</code> ({cnt} از {n})")
        lines.append("")

    return "\n".join(lines).rstrip()


# ==================== اجرای اصلی ====================
def main():
    log("=" * 60)
    log(f"🚀 مشابه‌یابی الگوی {TARGET_SYMBOL}")
    log("=" * 60)

    # ۱) داده‌ی روزانه‌ی نماد هدف
    df_target_daily = fetch(TARGET_SYMBOL, TARGET_INTERVAL, "10y")
    if df_target_daily is None or len(df_target_daily) < PATTERN_LEN:
        raise RuntimeError(f"داده‌ی روزانه‌ی {TARGET_SYMBOL} موجود نیست.")

    idx_naive = df_target_daily.index
    if getattr(idx_naive, 'tz', None) is not None:
        idx_naive = idx_naive.tz_localize(None)
    deltas = np.abs((idx_naive - pd.Timestamp(TARGET_DATE)).total_seconds())
    target_idx = int(np.argmin(deltas))
    if target_idx < PATTERN_LEN - 1:
        raise RuntimeError(f"کندل مرجع ({TARGET_DATE}) معتبر نیست.")

    log(f"🎯 کندل مرجع: {df_target_daily.index[target_idx].date()}")

    o_t = df_target_daily['Open'].values.astype(float)
    h_t = df_target_daily['High'].values.astype(float)
    l_t = df_target_daily['Low'].values.astype(float)
    c_t = df_target_daily['Close'].values.astype(float)

    target_feat = pattern_features(target_idx, o_t, h_t, l_t, c_t)
    if target_feat is None:
        raise RuntimeError("ویژگی کندل مرجع محاسبه نشد.")

    target_date_naive = df_target_daily.index[target_idx]
    if getattr(target_date_naive, 'tz', None) is not None:
        target_date_naive = target_date_naive.tz_localize(None)

    # ۲) داده‌ی هفتگی همه‌ی نمادها (مستقیم از yfinance)
    dfs = {}
    for sym in SYMBOLS:
        df = fetch_weekly(sym)
        if df is not None and len(df) > LOOKAHEAD + 10:
            dfs[sym] = df
            log(f"   ↳ {sym}: {len(df)} کندل هفتگی "
                f"({df.index[0].date()} → {df.index[-1].date()})")

    missing = [s for s in SYMBOLS if s not in dfs]
    if missing:
        log(f"⚠️ بدون داده‌ی هفتگی: {missing}")
    if not dfs:
        raise RuntimeError("هیچ نمادی داده‌ی هفتگی ندارد.")

    # ۳) جستجوی الگوهای مشابه
    log("\n🔍 جستجوی الگوهای هفتگی مشابه...")
    candidates = find_similar_candles(target_feat, target_date_naive, dfs)
    if not candidates:
        raise RuntimeError(
            "هیچ الگوی مشابهی یافت نشد. "
            "احتمالاً داده‌ی هفتگی کافی وجود ندارد."
        )
    log(f"🏆 {len(candidates)} الگوی مشابه برتر.")

    # ۴) تحلیل فوروارد
    results = []
    for c_ in candidates:
        sym = c_['symbol']; df = dfs[sym]
        o = df['Open'].values.astype(float)
        h = df['High'].values.astype(float)
        l = df['Low'].values.astype(float)
        cc = df['Close'].values.astype(float)
        n = len(df)
        r = analyze_forward(c_['idx'], o, h, l, cc, n, LOOKAHEAD)
        if r is None:
            continue
        r['symbol'] = sym
        results.append(r)

    if not results:
        raise RuntimeError("نتیجه‌ای محاسبه نشد.")

    # ۵) گزارش
    report = build_report(results)
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
