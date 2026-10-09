import os
import sys
import io
import traceback

import numpy as np
import pandas as pd
import requests
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import yfinance as yf


def log(m):
    print(m, flush=True)


# ==================== تنظیمات ====================
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
TELEGRAM_CHAT_ID   = os.getenv("TELEGRAM_CHAT_ID", "").strip()

TARGET_SYMBOL = "DOGE-USD"       # ← نماد هدف
TARGET_DATE   = "2025-07-04"     # ← تاریخ کندل مرجع (روزانه)

TARGET_INTERVAL = "1d"           # کندل مرجع
SEARCH_INTERVAL = "4h"           # جستجوی مشابه‌ها
SEARCH_PERIOD_HOURLY = "730d"    # بازه‌ی داده‌ی ۱ ساعته (حداکثر Yahoo)

PATTERN_LEN = 1                  # تعداد کندل الگو (روی کندل مرجع)

SYMBOLS = [
    "LUMIA-USD", "API3-USD", "FIL-USD",
    "DOGE-USD", "ONDO-USD", "SHIB-USD",
    "BLESS-USD", "ROSE-USD", "LINEA-USD", "JTO-USD",
]

LOOKAHEAD    = 100               # تعداد کندل ۴ساعته بعدی برای بررسی آینده
K_PER_SYMBOL = 20
TOTAL_TOP    = 100

LOW_DD_THRESHOLD = 3.0


# ---------- سطل‌های ۱٪ ----------
def build_buckets(start, end, step=1.0):
    edges = []
    v = start
    while v < end:
        edges.append((v, v + step))
        v += step
    return edges

DD_BUCKETS  = build_buckets(-50, 0, 1.0); DD_BUCKETS[0] = (-1e9, -50)
UP_BUCKETS  = build_buckets(0, 50, 1.0);  UP_BUCKETS.append((50, 1e9))
RET_BUCKETS = build_buckets(-50, 50, 1.0)
RET_BUCKETS.insert(0, (-1e9, -50)); RET_BUCKETS.append((50, 1e9))


# ==================== توابع ====================
def fetch(symbol, interval, period):
    """دریافت داده‌ی روزانه (برای کندل مرجع)."""
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
    """دریافت ۱ ساعته از Yahoo و تبدیل به ۴ ساعته."""
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
        # تبدیل 1h → 4h
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
    max_up      = (float(np.max(fut_h)) - entry_price) / entry_price * 100.0
    max_dd      = (float(np.min(fut_l)) - entry_price) / entry_price * 100.0

    peak_idx = int(np.argmax(fut_h))
    bars_to_peak = peak_idx + 1
    dd_before_peak = (float(np.min(fut_l[:peak_idx + 1])) - entry_price) \
                     / entry_price * 100.0

    above = fut_c >= entry_price
    if above.any():
        rec_idx = int(np.argmax(above))
        bars_to_recovery = rec_idx + 1
        dd_before_recovery = (float(np.min(fut_l[:rec_idx + 1])) - entry_price) \
                             / entry_price * 100.0
    else:
        bars_to_recovery = None
        dd_before_recovery = max_dd

    path = np.concatenate([[entry_price], fut_c]) / entry_price * 100.0

    return {
        'entry':              entry_price,
        'final':              final_price,
        'ret':                total_ret,
        'max_up':             max_up,
        'max_dd':             max_dd,
        'dd_before_peak':     dd_before_peak,
        'dd_before_recovery': dd_before_recovery,
        'bars_to_peak':       bars_to_peak,
        'bars_to_recovery':   bars_to_recovery,
        'direction':          'up' if total_ret > 0 else 'down',
        'path':               path,
    }


def bucketize(values, buckets):
    total = len(values)
    if total == 0:
        return []
    out = []
    for lo, hi in buckets:
        if lo <= -1e8:
            label = f"≤ {hi:+.0f}٪"
        elif hi >= 1e8:
            label = f"≥ {lo:+.0f}٪"
        else:
            label = f"{lo:+.0f}٪ تا {hi:+.0f}٪"
        cnt = sum(1 for v in values if lo <= v < hi)
        if cnt > 0:
            out.append((label, cnt, cnt / total * 100.0))
    return out


def compress_buckets(buckets_result, max_lines=15, min_pct=3.0):
    big   = [b for b in buckets_result if b[2] >= min_pct]
    small = [b for b in buckets_result if b[2] <  min_pct]
    if small:
        big.append(("سایر سطل‌ها",
                    sum(b[1] for b in small),
                    sum(b[2] for b in small)))
    return big


def fmt_buckets(buckets_result, emoji="", max_bar=20):
    lines = []
    if not buckets_result:
        return lines
    max_pct = max(p for _, _, p in buckets_result)
    for label, cnt, pct in buckets_result:
        bar = "█" * int(round(pct / max_pct * max_bar))
        lines.append(f"├ {label}: <code>{cnt}</code> مورد "
                     f"({pct:.1f}٪) {emoji}{bar}")
    return lines


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


def send_photo(buf, caption):
    if not (TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID):
        return
    try:
        buf.seek(0)
        r = requests.post(
            f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendPhoto",
            files={"photo": ("chart.png", buf, "image/png")},
            data={"chat_id": TELEGRAM_CHAT_ID, "caption": caption,
                  "parse_mode": "HTML"},
            timeout=60,
        )
        log(f"📤 telegram photo: {r.status_code}")
    except Exception as e:
        log(f"❌ send photo: {e}")


# ==================== جستجوی الگوهای مشابه (روی ۴ساعته) ====================
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
        log(f"   ✔️ {sym}: {k} الگوی ۴ساعته مشابه "
            f"(کمترین فاصله: {dists[order[0]]:.4f})")

    candidates_all.sort(key=lambda x: x['dist'])

    seen_dates = set()
    unique_candidates = []
    for c in candidates_all:
        key = (c['symbol'], c['date_key'])
        if key in seen_dates:
            continue
        seen_dates.add(key)
        unique_candidates.append(c)
        if len(unique_candidates) >= TOTAL_TOP:
            break

    log(f"🧹 بعد از حذف هم‌تاریخی‌ها: {len(unique_candidates)} الگوی یکتا")
    return unique_candidates


# ==================== نمودار ====================
def build_chart(details, target_forward, target_date_str):
    results = details['results']
    ups     = details['ups']
    downs   = details['downs']

    fig, axes = plt.subplots(2, 3, figsize=(20, 11))
    fig.patch.set_facecolor('#f5f5f5')

    symbols_used = sorted({r['symbol'] for r in results})
    palette = plt.cm.tab10(np.linspace(0, 1, max(len(symbols_used), 1)))
    color_map = {s: palette[i % len(palette)] for i, s in enumerate(symbols_used)}
    x = np.arange(LOOKAHEAD + 1)

    ax = axes[0, 0]
    for r in results:
        ax.plot(x, r['path'], color=color_map[r['symbol']],
                alpha=0.20, linewidth=0.8)
    if target_forward is not None:
        ax.plot(x, target_forward, color='#000000', linewidth=3.0,
                label=f'Target {TARGET_SYMBOL}')
    ax.axhline(100, color='#888', linestyle=':', linewidth=1)
    ax.set_title(f"All {len(results)} forward paths (4h)", fontweight='bold')
    ax.set_xlabel('Bars after pattern (4h)'); ax.set_ylabel('Normalized price')
    ax.grid(alpha=0.3); ax.legend(loc='upper left')

    ax = axes[0, 1]
    paths = np.array([r['path'] for r in results])
    ax.fill_between(x, np.percentile(paths, 10, axis=0),
                    np.percentile(paths, 90, axis=0),
                    color='#2196F3', alpha=0.15, label='10–90٪ band')
    ax.plot(x, np.percentile(paths, 50, axis=0), color='#1976D2',
            linewidth=2.0, label='Median')
    ax.plot(x, paths.mean(axis=0), color='#0D47A1',
            linewidth=2.5, label='Mean')
    if target_forward is not None:
        ax.plot(x, target_forward, color='#000000', linewidth=2.5,
                linestyle='--', label='Target actual')
    ax.axhline(100, color='#888', linestyle=':', linewidth=1)
    ax.set_title("Overall mean / median / band", fontweight='bold')
    ax.set_xlabel('Bars after pattern (4h)'); ax.set_ylabel('Normalized price')
    ax.grid(alpha=0.3); ax.legend(loc='upper left')

    ax = axes[0, 2]
    if ups:
        ax.plot(x, np.mean([r['path'] for r in ups], axis=0),
                color='#26a69a', linewidth=2.5, label=f"Mean UP ({len(ups)})")
        ax.fill_between(x, np.min([r['path'] for r in ups], axis=0),
                        np.max([r['path'] for r in ups], axis=0),
                        color='#26a69a', alpha=0.10)
    if downs:
        ax.plot(x, np.mean([r['path'] for r in downs], axis=0),
                color='#ef5350', linewidth=2.5, label=f"Mean DOWN ({len(downs)})")
        ax.fill_between(x, np.min([r['path'] for r in downs], axis=0),
                        np.max([r['path'] for r in downs], axis=0),
                        color='#ef5350', alpha=0.10)
    if target_forward is not None:
        ax.plot(x, target_forward, color='#000000', linewidth=2.5,
                linestyle='--', label='Target actual')
    ax.axhline(100, color='#888', linestyle=':', linewidth=1)
    ax.set_title("Mean path UP vs DOWN", fontweight='bold')
    ax.set_xlabel('Bars after pattern (4h)'); ax.set_ylabel('Normalized price')
    ax.grid(alpha=0.3); ax.legend(loc='upper left')

    dd_vals = details['all_max_dd']
    ax = axes[1, 0]
    ax.hist(dd_vals, bins=40, color='#ef5350', alpha=0.7, edgecolor='white')
    ax.axvline(np.mean(dd_vals), color='#b71c1c', linestyle='--',
               label=f"Mean {np.mean(dd_vals):.1f}٪")
    ax.axvline(np.min(dd_vals), color='#000000', linestyle=':',
               label=f"Worst {np.min(dd_vals):.1f}٪")
    ax.set_title("Distribution of max drawdown", fontweight='bold')
    ax.set_xlabel('Max drawdown (٪)'); ax.set_ylabel('Count')
    ax.grid(alpha=0.3); ax.legend(loc='upper left')

    up_vals = details['all_max_up']
    ax = axes[1, 1]
    ax.hist(up_vals, bins=40, color='#26a69a', alpha=0.7, edgecolor='white')
    ax.axvline(np.mean(up_vals), color='#004d40', linestyle='--',
               label=f"Mean +{np.mean(up_vals):.1f}٪")
    ax.axvline(np.max(up_vals), color='#000000', linestyle=':',
               label=f"Best +{np.max(up_vals):.1f}٪")
    ax.set_title("Distribution of max upside", fontweight='bold')
    ax.set_xlabel('Max upside (٪)'); ax.set_ylabel('Count')
    ax.grid(alpha=0.3); ax.legend(loc='upper left')

    ret_vals = details['all_ret']
    ax = axes[1, 2]
    ax.hist(ret_vals, bins=40, color='#2196F3', alpha=0.7, edgecolor='white')
    ax.axvline(0, color='#000', linestyle=':')
    ax.axvline(np.mean(ret_vals), color='#0D47A1', linestyle='--',
               label=f"Mean {np.mean(ret_vals):+.1f}٪")
    ax.set_title("Distribution of final return", fontweight='bold')
    ax.set_xlabel('Final return (٪)'); ax.set_ylabel('Count')
    ax.grid(alpha=0.3); ax.legend(loc='upper left')

    plt.suptitle(f"Top {len(results)} similar PAST 4h patterns | "
                 f"Target (daily): {TARGET_SYMBOL} {target_date_str}",
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100, facecolor='#f5f5f5')
    plt.close()
    buf.seek(0)
    return buf


# ==================== گزارش خیلی خلاصه ====================
def report_minimal(ups, n_total):
    """فقط دو عدد: زیر ۲٪ ریزش + ۲۰٪ رشد، و زیر ۳٪ ریزش + ۲۰٪ رشد."""
    def joint(th, target=20.0):
        low = [r for r in ups
               if r['dd_before_recovery'] is not None
               and r['dd_before_recovery'] >= -th]
        cnt = sum(1 for r in low if r['max_up'] >= target)
        pct_total = cnt / n_total * 100.0 if n_total else 0.0
        return cnt, len(low), pct_total

    c2, n2, p2 = joint(2.0)
    c3, n3, p3 = joint(3.0)

    lines = []
    lines.append(f"🕯 <b>الگوی {TARGET_SYMBOL} | {TARGET_DATE}</b>")
    lines.append("━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    lines.append("📊 <b>نتیجه:</b>")
    lines.append(f"├ ریزش زیر ۲٪ + رشد ۲۰٪: "
                 f"<code>{p2:.1f}٪</code>  ({c2} از {n_total})")
    lines.append(f"└ ریزش زیر ۳٪ + رشد ۲۰٪: "
                 f"<code>{p3:.1f}٪</code>  ({c3} از {n_total})")
    return "\n".join(lines)


# ==================== اجرای اصلی ====================
def main():
    pat_label = "تک‌کندلی روزانه"
    log("=" * 60)
    log(f"🚀 مشابه‌یابی الگوی {pat_label} {TARGET_SYMBOL}")
    log(f"   کندل مرجع: روزانه  |  جستجو: ۴ ساعته")
    log("=" * 60)

    # ---------- ۱) دریافت داده‌ی روزانه‌ی نماد هدف ----------
    df_target_daily = fetch(TARGET_SYMBOL, TARGET_INTERVAL, "10y")
    if df_target_daily is None or len(df_target_daily) < PATTERN_LEN:
        raise RuntimeError(f"داده‌ی روزانه‌ی {TARGET_SYMBOL} موجود نیست.")

    idx_naive = df_target_daily.index
    if getattr(idx_naive, 'tz', None) is not None:
        idx_naive = idx_naive.tz_localize(None)
    deltas = np.abs((idx_naive - pd.Timestamp(TARGET_DATE)).total_seconds())
    target_idx = int(np.argmin(deltas))
    if target_idx < PATTERN_LEN - 1:
        raise RuntimeError(f"کندل مرجع ({TARGET_DATE}) باید "
                           f"حداقل {PATTERN_LEN - 1} کندل قبل از خود داشته باشد.")

    log(f"🎯 کندل مرجع (روزانه): "
        f"{df_target_daily.index[target_idx].date()}")

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

    # ---------- ۲) دریافت داده‌ی ۴ساعته‌ی همه‌ی نمادها ----------
    dfs = {}
    for sym in SYMBOLS:
        df = fetch_4h(sym)
        if df is not None and len(df) > LOOKAHEAD + 10:
            dfs[sym] = df

    missing = [s for s in SYMBOLS if s not in dfs]
    if missing:
        log(f"⚠️ نمادهای بدون داده‌ی ۴ساعته: {missing}")

    if not dfs:
        raise RuntimeError("هیچ نمادی داده‌ی ۴ساعته ندارد.")

    # ---------- ۳) جستجوی الگوهای ۴ساعته‌ی مشابه ----------
    log(f"\n🔍 جستجوی الگوهای ۴ساعته‌ی مشابه در گذشته...")
    candidates = find_similar_candles(target_feat, target_date_naive, dfs)
    if not candidates:
        raise RuntimeError("هیچ الگوی مشابهی یافت نشد.")
    log(f"\n🏆 {len(candidates)} الگوی ۴ساعته‌ی مشابه برتر انتخاب شد.")

    # ---------- ۴) تحلیل فوروارد روی ۴ساعته ----------
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
        r['idx']    = c_['idx']
        r['date']   = df.index[c_['idx']]
        r['dist']   = c_['dist']
        results.append(r)

    if not results:
        raise RuntimeError("نتیجه‌ای محاسبه نشد.")

    ups   = [r for r in results if r['direction'] == 'up']
    downs = [r for r in results if r['direction'] == 'down']
    n_total = len(results)
    n_up = len(ups); n_down = n_total - n_up
    win_rate = n_up / n_total * 100.0

    target_forward = None

    details = {
        'results':       results,
        'ups':           ups,
        'downs':         downs,
        'n_total':       n_total,
        'n_up':          n_up,
        'n_down':        n_down,
        'win_rate':      win_rate,
        'all_ret':       [r['ret']    for r in results],
        'all_max_up':    [r['max_up'] for r in results],
        'all_max_dd':    [r['max_dd'] for r in results],
        'target_actual': None,
    }

    # ---------- ۵) گزارش خیلی خلاصه ----------
    report = report_minimal(ups, n_total)
    plain = (report.replace("<b>", "").replace("</b>", "")
                   .replace("<code>", "").replace("</code>", ""))
    log("\n" + plain)
    send_message(report)

    # ---------- ۶) نمودار ----------
    try:
        target_date_str = df_target_daily.index[target_idx].strftime('%Y-%m-%d')
        buf = build_chart(details, target_forward, target_date_str)
        caption = (f"🕯 کندل مرجع (روزانه): {TARGET_SYMBOL} "
                   f"{target_date_str}\n"
                   f"🔍 الگوهای مشابه: ۴ ساعته\n"
                   f"تعداد: {n_total} | صعودی: {win_rate:.0f}٪")
        send_photo(buf, caption)
    except Exception as e:
        log(f"❌ chart: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        log(f"❌ خطا: {e}")
        traceback.print_exc()
        sys.exit(1)
