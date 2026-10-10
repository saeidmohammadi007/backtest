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
TARGET_DATE   = "2025-07-04"

TARGET_INTERVAL = "1d"
SEARCH_PERIOD_HOURLY = "730d"

PATTERN_LEN = 1

SYMBOLS = [
    "LUMIA-USD", "API3-USD", "FIL-USD",
    "DOGE-USD", "ONDO-USD", "SHIB-USD",
    "BLESS-USD", "ROSE-USD", "LINEA-USD", "JTO-USD",
]

LOOKAHEAD    = 100
K_PER_SYMBOL = 20
TOTAL_TOP    = 100

TARGETS       = [20.0, 50.0]
DD_THRESHOLDS = [2.0]           # فقط ۲٪ (با اهرم ۳۰x)

# ---------- مدیریت سرمایه ----------
MARGIN_PCT   = 1.0
LEVERAGE     = 30.0
NOTIONAL_PCT = MARGIN_PCT * LEVERAGE

# ---------- کارمزد و هزینه‌ها ----------
FEE_OPEN_PCT  = 0.05           # کارمزد ورود (٪ نهional)
FEE_CLOSE_PCT = 0.05           # کارمزد خروج (٪ نهional)
FUNDING_PER_8H = 0.01          # funding تقریبی هر ۸ ساعت (٪ نهional)
# توجه: funding بسته به جهت پوزیشن می‌تونه مثبت یا منفی باشه؛
# محافظه‌کارانه همیشه مثبت (هزینه) فرض می‌کنیم.


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


def fetch_4h(symbol):
    log(f"📥 دریافت {symbol} (1h→4h)")
    try:
        df = yf.download(symbol, interval="1h", period=SEARCH_PERIOD_HOURLY,
                         progress=False, auto_adjust=True, timeout=60)
        if df is None or df.empty:
            return None
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df[['Open', 'High', 'Low', 'Close']].dropna()
        if df.empty:
            return None
        df_4h = df.resample('4h').agg({
            'Open':  'first',
            'High':  'max',
            'Low':   'min',
            'Close': 'last',
        }).dropna()
        return df_4h if not df_4h.empty else None
    except Exception as e:
        log(f"❌ fetch_4h {symbol}: {e}")
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


def cost_on_capital(bars_held):
    """
    هزینه‌ی کل روی سرمایه (٪):
    کارمزد ورود + خروج + funding برای مدت نگه‌داری.
    هر کندل ۴ساعته = ۴ ساعت → هر ۸ ساعت یک دوره funding.
    """
    fee_open  = NOTIONAL_PCT * (FEE_OPEN_PCT  / 100.0)   # ٪ سرمایه
    fee_close = NOTIONAL_PCT * (FEE_CLOSE_PCT / 100.0)  # ٪ سرمایه
    periods_8h = bars_held * 4.0 / 8.0
    funding   = NOTIONAL_PCT * (FUNDING_PER_8H / 100.0) * periods_8h
    return fee_open + fee_close + funding


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

    out = {'ret': total_ret, 'entry': entry_price, 'final': final_price}

    for D in DD_THRESHOLDS:
        sl_price = entry_price * (1 - D / 100.0)
        for T in TARGETS:
            tp_price = entry_price * (1 + T / 100.0)
            outcome = None
            exit_ret = 0.0
            bars_held = 0
            for k in range(len(fut_c)):
                lo = float(fut_l[k]); hi = float(fut_h[k])
                if lo <= sl_price:
                    outcome = 'sl'; exit_ret = -D; bars_held = k + 1; break
                if hi >= tp_price:
                    outcome = 'tp'; exit_ret = +T; bars_held = k + 1; break
            if outcome is None:
                outcome = 'time'
                bars_held = len(fut_c)
                exit_ret = (float(fut_c[-1]) - entry_price) / entry_price * 100.0

            # بازده ناخالص روی سرمایه
            gross_cap_ret = NOTIONAL_PCT * (exit_ret / 100.0)
            # هزینه روی سرمایه
            cost = cost_on_capital(bars_held)
            # بازده خالص
            net_cap_ret = gross_cap_ret - cost

            # لیکویید تقریبی؟ (وقتی قیمت ~3.3٪ بره خلاف پوزیشن)
            liq_pct = 100.0 / LEVERAGE
            liq_hit = False
            if liq_pct <= D:
                # SL قبل از لیکویید می‌خوره، پس لیکویید نمی‌شیم
                pass
            else:
                # اگه SL بالاتر از لیکویید باشه، قبل از SL لیکویید می‌شیم
                # اما اینجا D≤2 و liq=3.33 پس همیشه SL اوله
                pass

            out[f'{D}_{T}'] = {
                'outcome':     outcome,
                'exit_ret':    exit_ret,
                'gross':       gross_cap_ret,
                'cost':        cost,
                'cap_ret':     net_cap_ret,
                'bars_held':   bars_held,
            }
    return out


def find_similar_candles(target_feat, target_date_naive, dfs):
    candidates_all = []
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
        if not mask_before.any():
            continue
        last_before = int(np.where(mask_before)[0][-1])
        max_i = min(max_i, last_before)

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
    liq_pct = 100.0 / LEVERAGE

    # هزینه‌ی تیپیک (برای ۵۰ کندل نگه‌داری)
    sample_cost = cost_on_capital(50)

    lines = []
    lines.append(f"🕯 <b>الگوی {TARGET_SYMBOL} | {TARGET_DATE}</b>")
    lines.append("━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    lines.append(f"🔎 تعداد الگوهای مشابه: <code>{n}</code>")
    lines.append(f"💰 سوددهی کلی (قیمتی): "
                 f"<code>{win_pct:.1f}٪</code> ({wins} از {n})")
    lines.append("")
    lines.append(f"⚙️ <b>مارجین {MARGIN_PCT:.0f}٪ × اهرم "
                 f"{LEVERAGE:.0f}x  (نotional ≈ "
                 f"{NOTIONAL_PCT:.0f}٪ سرمایه)</b>")
    lines.append(f"💸 کارمزد رفت+برگشت: "
                 f"<code>−{NOTIONAL_PCT*(FEE_OPEN_PCT+FEE_CLOSE_PCT)/100:.1f}٪</code>"
                 f" سرمایه در هر معامله")
    lines.append(f"💸 هزینه‌ی ۵۰ کندل (~۸ روز): "
                 f"<code>−{sample_cost:.1f}٪</code> سرمایه")
    lines.append(f"⚠️ لیکویید تقریبی: حدود <code>−{liq_pct:.1f}٪</code>")
    lines.append("")

    for D in DD_THRESHOLDS:
        warn = " ⚠️ خیلی نزدیک به لیکویید" if D >= liq_pct * 0.9 else ""
        lines.append(f"📉 <b>SL = {D:.0f}٪</b>{warn}")
        for T in TARGETS:
            tp = sum(1 for r in results if r[f'{D}_{T}']['outcome'] == 'tp')
            sl = sum(1 for r in results if r[f'{D}_{T}']['outcome'] == 'sl')
            tm = n - tp - sl
            cap_rets = [r[f'{D}_{T}']['cap_ret'] for r in results]
            ev = float(np.mean(cap_rets))
            total = float(np.sum(cap_rets))

            tp_pct = tp / n * 100.0
            sl_pct = sl / n * 100.0
            tm_pct = tm / n * 100.0

            lines.append(f"🎯 هدف +{T:.0f}٪  →  "
                         f"<code>{tp_pct:.1f}٪</code> TP  |  "
                         f"<code>{sl_pct:.1f}٪</code> SL  |  "
                         f"<code>{tm_pct:.1f}٪</code> زمانی")
            lines.append(f"   ├ میانگین هر معامله (خالص): "
                         f"<code>{ev:+.2f}٪</code> سرمایه")
            lines.append(f"   └ بازده کل ({n} معامله): "
                         f"<code>{total:+.0f}٪</code> سرمایه")
        lines.append("")

    return "\n".join(lines).rstrip()


# ==================== اجرای اصلی ====================
def main():
    log("=" * 60)
    log(f"🚀 مشابه‌یابی الگوی {TARGET_SYMBOL}")
    log(f"   مارجین {MARGIN_PCT}٪ × اهرم {LEVERAGE}x  "
        f"(نotional ≈ {NOTIONAL_PCT}٪ سرمایه)")
    log("=" * 60)

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

    dfs = {}
    for sym in SYMBOLS:
        df = fetch_4h(sym)
        if df is not None and len(df) > LOOKAHEAD + 10:
            dfs[sym] = df

    missing = [s for s in SYMBOLS if s not in dfs]
    if missing:
        log(f"⚠️ بدون داده‌ی ۴ساعته: {missing}")
    if not dfs:
        raise RuntimeError("هیچ نمادی داده‌ی ۴ساعته ندارد.")

    log("\n🔍 جستجوی الگوهای ۴ساعته‌ی مشابه...")
    candidates = find_similar_candles(target_feat, target_date_naive, dfs)
    if not candidates:
        raise RuntimeError("هیچ الگوی مشابهی یافت نشد.")
    log(f"🏆 {len(candidates)} الگوی مشابه برتر.")

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
