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

TARGET_SYMBOL = "ETH-USD"
TARGET_DATE   = "2025-07-04"     # ← کندل الگو (تک‌کندلی)

PATTERN_LEN   = 1                # ← الگو تک‌کندلی

SYMBOLS = [
    "BTC-USD", "ETH-USD", "BNB-USD", "SOL-USD", "XRP-USD",
    "ADA-USD", "DOGE-USD", "AVAX-USD", "DOT-USD", "LINK-USD",
]

INTERVAL = "1d"
PERIOD   = "10y"

LOOKAHEAD    = 100
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
    log(f"📥 دریافت {symbol}")
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
    """
    ویژگی‌های الگوی چندکندلی که به کندل end_idx ختم می‌شود.
    خروجی: بردار 4*pattern_len بعدی.
    """
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


# ==================== جستجوی الگوهای مشابه ====================
def find_similar_candles(df_target, target_idx, dfs):
    o_t = df_target['Open'].values.astype(float)
    h_t = df_target['High'].values.astype(float)
    l_t = df_target['Low'].values.astype(float)
    c_t = df_target['Close'].values.astype(float)

    target_feat = pattern_features(target_idx, o_t, h_t, l_t, c_t)
    if target_feat is None:
        raise RuntimeError("ویژگی الگوی هدف محاسبه نشد.")

    target_date = df_target.index[target_idx]
    target_date_naive = target_date
    if getattr(target_date_naive, 'tz', None) is not None:
        target_date_naive = target_date_naive.tz_localize(None)

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

        if sym == TARGET_SYMBOL:
            max_i = min(max_i, target_idx - 1)

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

        # استخر بزرگ‌تر می‌گیریم چون بخشی از کاندیدها در مرحله حذف
        # هم‌تاریخی حذف می‌شوند.
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
        log(f"   ✔️ {sym}: {k} الگوی مشابه "
            f"(کمترین فاصله: {dists[order[0]]:.4f})")

    candidates_all.sort(key=lambda x: x['dist'])

    # ↓ حذف هم‌تاریخی‌ها: هیچ دو کندلی از یک تاریخ انتخاب نمی‌شود
    seen_dates = set()
    unique_candidates = []
    for c in candidates_all:
        if c['date_key'] in seen_dates:
            continue
        seen_dates.add(c['date_key'])
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
    ax.set_title(f"All {len(results)} forward paths", fontweight='bold')
    ax.set_xlabel('Bars after pattern'); ax.set_ylabel('Normalized price')
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
    ax.set_xlabel('Bars after pattern'); ax.set_ylabel('Normalized price')
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
    ax.set_xlabel('Bars after pattern'); ax.set_ylabel('Normalized price')
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

    pat_label = "single candle" if PATTERN_LEN == 1 else f"{PATTERN_LEN} candles"
    plt.suptitle(f"Top {len(results)} similar PAST patterns "
                 f"({pat_label}) | "
                 f"Target: {TARGET_SYMBOL} {target_date_str}",
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100, facecolor='#f5f5f5')
    plt.close()
    buf.seek(0)
    return buf


# ==================== تحلیل مسیر گروه صعودی ====================
def report_up_group_path(ups):
    if not ups:
        return ["<b>🟢 گروه صعودی‌ها:</b>", "└ داده‌ای نیست.", ""]

    n = len(ups)
    dd_peak  = [r['dd_before_peak']     for r in ups]
    dd_rec   = [r['dd_before_recovery'] for r in ups
                if r['dd_before_recovery'] is not None]
    bars_rec = [r['bars_to_recovery']   for r in ups
                if r['bars_to_recovery'] is not None]
    bars_pk  = [r['bars_to_peak']       for r in ups]

    lines = []
    lines.append(f"<b>🔬 تحلیل مسیر {n} صعودی (اول ریزش، بعد صعود):</b>")
    lines.append("")

    if dd_rec:
        lines.append("📉 <b>ریزش قبل از سربه‌سر شدن:</b>")
        lines.append(f"├ میانگین: <code>{np.mean(dd_rec):+.2f}٪</code>")
        lines.append(f"├ میانه: <code>{np.median(dd_rec):+.2f}٪</code>")
        lines.append(f"└ بدترین: <code>{np.min(dd_rec):+.2f}٪</code>")

        def pct_above(vals, threshold):
            if not vals:
                return 0.0
            return sum(1 for v in vals if v >= -threshold) / len(vals) * 100.0

        lines.append("")
        lines.append("📊 <b>چند درصد قبل از سربه‌سر شدن، "
                     "کمتر از ... ریزش داشتن؟</b>")
        lines.append(f"├ کمتر از ۵٪: <code>{pct_above(dd_rec, 5):.1f}٪</code>")
        lines.append(f"├ کمتر از ۱۰٪: <code>{pct_above(dd_rec, 10):.1f}٪</code>")
        lines.append(f"├ کمتر از ۲۰٪: <code>{pct_above(dd_rec, 20):.1f}٪</code>")
        lines.append(f"├ کمتر از ۳۰٪: <code>{pct_above(dd_rec, 30):.1f}٪</code>")
        lines.append(f"└ کمتر از ۵۰٪: <code>{pct_above(dd_rec, 50):.1f}٪</code>")

    lines.append("")
    lines.append("🏔 <b>ریزش قبل از رسیدن به سقف:</b>")
    lines.append(f"├ میانگین: <code>{np.mean(dd_peak):+.2f}٪</code>")
    lines.append(f"├ میانه: <code>{np.median(dd_peak):+.2f}٪</code>")
    lines.append(f"└ بدترین: <code>{np.min(dd_peak):+.2f}٪</code>")

    lines.append("")
    lines.append("⏱ <b>زمان‌بندی (تعداد کندل):</b>")
    if bars_rec:
        lines.append(f"├ میانگین رسیدن به سربه‌سر: "
                     f"<code>{np.mean(bars_rec):.0f}</code> کندل")
        lines.append(f"├ میانه رسیدن به سربه‌سر: "
                     f"<code>{np.median(bars_rec):.0f}</code> کندل")
    lines.append(f"├ میانگین رسیدن به سقف: "
                 f"<code>{np.mean(bars_pk):.0f}</code> کندل")
    lines.append(f"└ میانه رسیدن به سقف: "
                 f"<code>{np.median(bars_pk):.0f}</code> کندل")

    if dd_rec:
        lines.append("")
        lines.append("📊 <b>توزیع ریزش اولیه (قبل از سربه‌سر):</b>")
        b = bucketize(dd_rec, DD_BUCKETS)
        lines.extend(fmt_buckets(compress_buckets(b, max_lines=10), emoji="🟥"))
    return lines


# ==================== تحلیل اهداف سود ====================
def report_target_reach(ups):
    lines = []
    if not ups:
        return lines

    n_up = len(ups)
    max_ups = [r['max_up'] for r in ups]

    targets = [5, 10, 15, 20, 30, 50, 75, 100, 150, 200]

    lines.append(f"<b>🎯 از {n_up} مورد صعودی، "
                 f"چند درصد به این اهداف رسیدن؟</b>")
    lines.append("<i>(یعنی حداکثر رشدی که در ۱۰۰ کندل بعد تجربه کردن، "
                 "حداقل چقدر بود)</i>")
    lines.append("")
    for t in targets:
        cnt = sum(1 for v in max_ups if v >= t)
        pct = cnt / n_up * 100.0
        bar = "█" * int(round(pct / 5))
        lines.append(f"├ رسیدن به <b>+{t}٪</b>: "
                     f"<code>{cnt:>2}</code> مورد ({pct:5.1f}٪) 🟢{bar}")
    lines.append("")

    lines.append("📊 <b>آمار بیشترین رشد در صعودی‌ها:</b>")
    lines.append(f"├ میانگین: <code>+{np.mean(max_ups):.1f}٪</code>")
    lines.append(f"├ میانه: <code>+{np.median(max_ups):.1f}٪</code>")
    lines.append(f"├ بیشترین: <code>+{np.max(max_ups):.1f}٪</code>")
    lines.append(f"└ کمترین: <code>+{np.min(max_ups):.1f}٪</code>")
    lines.append("")

    lines.append("📈 <b>توزیع بیشترین رشد در صعودی‌ها:</b>")
    b = bucketize(max_ups, UP_BUCKETS)
    lines.extend(fmt_buckets(compress_buckets(b, max_lines=12), emoji="🟩"))
    return lines


def report_target_reach_down(downs):
    lines = []
    if not downs:
        return lines

    n = len(downs)
    max_dds = [abs(r['max_dd']) for r in downs]
    targets = [5, 10, 20, 30, 50, 70]

    lines.append(f"<b>📉 از {n} مورد نزولی، "
                 f"چند درصد ریزش این‌قدری داشتن؟</b>")
    for t in targets:
        cnt = sum(1 for v in max_dds if v >= t)
        pct = cnt / n * 100.0
        bar = "█" * int(round(pct / 5))
        lines.append(f"├ ریزش بیش از <b>{t}٪</b>: "
                     f"<code>{cnt:>2}</code> مورد ({pct:5.1f}٪) 🟥{bar}")
    return lines


# ==================== تحلیل برنده‌های کم‌ریسک ====================
def report_low_dd_winners(ups, n_total, dd_threshold=3.0):
    lines = []
    if not ups:
        return lines

    low_dd = [r for r in ups
              if r['dd_before_recovery'] is not None
              and r['dd_before_recovery'] >= -dd_threshold]

    lines.append(f"<b>💎 برنده‌هایی با ریزش اولیه کمتر از "
                 f"{dd_threshold:.0f}٪ (کم‌ریسک‌ها):</b>")

    if not low_dd:
        lines.append("└ موردی یافت نشد.")
        return lines

    n_low = len(low_dd)
    n_up  = len(ups)

    rets   = [r['ret']    for r in low_dd]
    maxups = [r['max_up'] for r in low_dd]
    maxdds = [r['max_dd'] for r in low_dd]
    dds_r  = [r['dd_before_recovery'] for r in low_dd]
    bars_r = [r['bars_to_recovery'] for r in low_dd
              if r['bars_to_recovery'] is not None]

    def hit(t):
        return sum(1 for v in maxups if v >= t) / n_low * 100.0

    lines.append(f"├ تعداد: <code>{n_low}</code> از "
                 f"<code>{n_up}</code> برنده "
                 f"({n_low / n_up * 100:.1f}٪ از برنده‌ها)")
    lines.append(f"└ سهم از کل {n_total} الگوی مشابه: "
                 f"<code>{n_low / n_total * 100:.1f}٪</code>")
    lines.append("")

    lines.append("📈 <b>سود نهایی (بعد از ۱۰۰ کندل):</b>")
    lines.append(f"├ میانگین: <code>{np.mean(rets):+.2f}٪</code>")
    lines.append(f"├ میانه: <code>{np.median(rets):+.2f}٪</code>")
    lines.append(f"├ بیشترین: <code>{np.max(rets):+.2f}٪</code>")
    lines.append(f"└ کمترین: <code>{np.min(rets):+.2f}٪</code>")
    lines.append("")

    lines.append("🚀 <b>بیشترین رشد (سقف مسیر):</b>")
    lines.append(f"├ میانگین: <code>{np.mean(maxups):+.2f}٪</code>")
    lines.append(f"├ میانه: <code>{np.median(maxups):+.2f}٪</code>")
    lines.append(f"├ بیشترین: <code>{np.max(maxups):+.2f}٪</code>")
    lines.append(f"└ کمترین: <code>{np.min(maxups):+.2f}٪</code>")
    lines.append("")

    lines.append("📉 <b>ریزش تجربه‌شده در کل مسیر:</b>")
    lines.append(f"├ میانگین بیشترین ریزش: <code>{np.mean(maxdds):+.2f}٪</code>")
    lines.append(f"├ میانگین ریزش اولیه: <code>{np.mean(dds_r):+.2f}٪</code>")
    lines.append(f"└ بدترین ریزش: <code>{np.min(maxdds):+.2f}٪</code>")
    lines.append("")

    if bars_r:
        lines.append("⏱ <b>زمان رسیدن به سربه‌سر:</b>")
        lines.append(f"├ میانگین: <code>{np.mean(bars_r):.0f}</code> کندل")
        lines.append(f"└ میانه: <code>{np.median(bars_r):.0f}</code> کندل")
        lines.append("")

    lines.append(f"🎯 <b>چند درصد این زیرگروه به اهداف سود رسیدن:</b>")
    for t in [5, 10, 20, 30, 50, 75, 100, 150, 200]:
        p = hit(t)
        bar = "█" * int(round(p / 5))
        lines.append(f"├ +{t}٪: <code>{p:5.1f}٪</code> 🟢{bar}")
    return lines


# ==================== گزارش اصلی ====================
def report_single(details, df_target, target_idx, target_forward):
    res = details['results']
    ups = details['ups']; downs = details['downs']
    n_total = details['n_total']
    n_up = details['n_up']; n_down = details['n_down']
    win_rate = details['win_rate']
    all_ret    = details['all_ret']
    all_max_up = details['all_max_up']
    all_max_dd = details['all_max_dd']
    target_actual = details.get('target_actual')

    def sub(lst, key):
        return [x[key] for x in lst] if lst else []

    avg_ret_all = float(np.mean(all_ret))
    avg_max_up  = float(np.mean(all_max_up))
    avg_max_dd  = float(np.mean(all_max_dd))

    if ups:
        avg_ret_up    = float(np.mean(sub(ups, 'ret')))
        avg_max_up_up = float(np.mean(sub(ups, 'max_up')))
        avg_dd_up     = float(np.mean(sub(ups, 'max_dd')))
        worst_dd_up   = float(np.min(sub(ups, 'max_dd')))
    else:
        avg_ret_up = avg_max_up_up = avg_dd_up = worst_dd_up = 0.0

    if downs:
        avg_ret_down  = float(np.mean(sub(downs, 'ret')))
        avg_dd_down   = float(np.mean(sub(downs, 'max_dd')))
        worst_dd_down = float(np.min(sub(downs, 'max_dd')))
        worst_ret_down= float(np.min(sub(downs, 'ret')))
    else:
        avg_ret_down = avg_dd_down = worst_dd_down = worst_ret_down = 0.0

    dd_buckets    = bucketize(all_max_dd, DD_BUCKETS)
    up_buckets    = bucketize(all_max_up, UP_BUCKETS)
    ret_buckets   = bucketize(all_ret,    RET_BUCKETS)

    def pct_between(vals, lo, hi):
        if not vals:
            return 0.0
        return sum(1 for v in vals if lo <= v < hi) / len(vals) * 100.0

    pct_dd_lt_2  = pct_between(all_max_dd, -1e9, -2)
    pct_dd_lt_5  = pct_between(all_max_dd, -1e9, -5)
    pct_dd_lt_10 = pct_between(all_max_dd, -1e9, -10)
    pct_dd_lt_20 = pct_between(all_max_dd, -1e9, -20)

    sym_counts = {}
    for r in res:
        sym_counts[r['symbol']] = sym_counts.get(r['symbol'], 0) + 1

    o_t = df_target['Open'].values.astype(float)
    h_t = df_target['High'].values.astype(float)
    l_t = df_target['Low'].values.astype(float)
    c_t = df_target['Close'].values.astype(float)

    pat_label = "تک‌کندلی" if PATTERN_LEN == 1 else f"{PATTERN_LEN}-کندلی"

    lines = []
    lines.append(f"🕯 <b>تحلیل الگوی {pat_label} مشابه {TARGET_SYMBOL}</b>")
    lines.append("━━━━━━━━━━━━━━━━━━━━━━━━━━━")

    # نمایش کندل‌های الگو
    lines.append(f"🗓 <b>کندل‌های الگو ({PATTERN_LEN} کندل):</b>")
    pattern_start = target_idx - PATTERN_LEN + 1
    for k, j in enumerate(range(pattern_start, target_idx + 1), 1):
        op = o_t[j]; cl = c_t[j]; hi = h_t[j]; lo = l_t[j]
        if op <= 0:
            continue
        body_pct  = (cl - op) / op * 100
        range_pct = (hi - lo) / op * 100
        color = "🟢" if cl >= op else "🔴"
        d_str = df_target.index[j].date()
        lines.append(
            f"├ کندل {k} ({d_str}) {color}: "
            f"O=<code>{op:.2f}</code> H=<code>{hi:.2f}</code> "
            f"L=<code>{lo:.2f}</code> C=<code>{cl:.2f}</code>"
        )
        lines.append(
            f"│   بدنه: <code>{body_pct:+.2f}٪</code> | "
            f"دامنه: <code>{range_pct:.2f}٪</code>"
        )

    entry_price = c_t[target_idx]
    lines.append(f"└ 💵 قیمت ورود (Close آخرین کندل): "
                 f"<code>{entry_price:.2f}</code>")
    lines.append("")
    lines.append(f"⏭ بررسی <code>{LOOKAHEAD}</code> کندل بعدی از این نقطه")
    lines.append(f"✅ فقط الگوهای گذشته (بدون نگاه به آینده)")
    lines.append(f"🌐 تعداد الگوی مشابه: <code>{n_total}</code>")
    lines.append("")

    lines.append("<b>📈 آمار کلی:</b>")
    lines.append(f"├ 🟢 صعودی: <code>{n_up}</code> مورد ({win_rate:.1f}٪)")
    lines.append(f"├ 🔴 نزولی: <code>{n_down}</code> مورد ({100 - win_rate:.1f}٪)")
    lines.append(f"├ میانگین بازدهی: <code>{avg_ret_all:+.2f}٪</code>")
    lines.append(f"├ میانگین بیشترین رشد: <code>{avg_max_up:+.2f}٪</code>")
    lines.append(f"└ میانگین بیشترین ریزش: <code>{avg_max_dd:+.2f}٪</code>")
    lines.append("")

    lines.append("<b>📌 خلاصه‌ی احتمال ریزش:</b>")
    lines.append(f"├ بیش از ۲٪ ریزش: <code>{pct_dd_lt_2:.1f}٪</code>")
    lines.append(f"├ بیش از ۵٪ ریزش: <code>{pct_dd_lt_5:.1f}٪</code>")
    lines.append(f"├ بیش از ۱۰٪ ریزش: <code>{pct_dd_lt_10:.1f}٪</code>")
    lines.append(f"└ بیش از ۲۰٪ ریزش: <code>{pct_dd_lt_20:.1f}٪</code>")
    lines.append("")

    lines.append("<b>📉 توزیع حداکثر ریزش:</b>")
    lines.extend(fmt_buckets(compress_buckets(dd_buckets, max_lines=15), emoji="🟥"))
    lines.append(f"└ 📌 بدترین ریزش: <code>{min(all_max_dd):+.2f}٪</code>")
    lines.append("")

    lines.append("<b>📈 توزیع حداکثر رشد:</b>")
    lines.extend(fmt_buckets(compress_buckets(up_buckets, max_lines=15), emoji="🟩"))
    lines.append(f"└ 📌 بهترین رشد: <code>{max(all_max_up):+.2f}٪</code>")
    lines.append("")

    lines.append(f"<b>🎯 توزیع بازدهی نهایی (بعد از {LOOKAHEAD} کندل):</b>")
    lines.extend(fmt_buckets(compress_buckets(ret_buckets, max_lines=15), emoji="🔵"))
    lines.append("")

    if ups:
        lines.append("<b>🟢 گروه صعودی‌ها (نهایتاً مثبت):</b>")
        lines.append(f"├ میانگین بازدهی: <code>{avg_ret_up:+.2f}٪</code>")
        lines.append(f"├ میانگین بیشترین رشد: <code>{avg_max_up_up:+.2f}٪</code>")
        lines.append(f"├ میانگین بیشترین ریزش: <code>{avg_dd_up:+.2f}٪</code>")
        lines.append(f"└ 📌 بدترین ریزش در صعودی‌ها: "
                     f"<code>{worst_dd_up:+.2f}٪</code>")
        lines.append("")

        lines.extend(report_up_group_path(ups))
        lines.append("")

        lines.extend(report_low_dd_winners(ups, n_total,
                                           dd_threshold=LOW_DD_THRESHOLD))
        lines.append("")

        lines.extend(report_target_reach(ups))
        lines.append("")

    if downs:
        lines.append("<b>🔴 گروه نزولی‌ها (نهایتاً منفی):</b>")
        lines.append(f"├ میانگین بازدهی: <code>{avg_ret_down:+.2f}٪</code>")
        lines.append(f"├ میانگین بیشترین ریزش: <code>{avg_dd_down:+.2f}٪</code>")
        lines.append(f"├ 📉 بدترین ریزش: <code>{worst_dd_down:+.2f}٪</code>")
        lines.append(f"└ بدترین بازدهی: <code>{worst_ret_down:+.2f}٪</code>")
        lines.append("")

        lines.extend(report_target_reach_down(downs))
        lines.append("")

    if target_actual:
        lines.append("<b>🔵 مسیر واقعی همین الگو تا الان:</b>")
        lines.append(f"├ بازدهی: <code>{target_actual['ret']:+.2f}٪</code>")
        lines.append(f"├ بیشترین رشد: <code>{target_actual['max_up']:+.2f}٪</code>")
        lines.append(f"└ بیشترین ریزش: <code>{target_actual['max_dd']:+.2f}٪</code>")
        lines.append("")

    lines.append("<b>🌐 توزیع ارزها:</b>")
    for s, cnt in sorted(sym_counts.items(), key=lambda x: -x[1]):
        lines.append(f"• <code>{s}</code>: {cnt} مورد")
    lines.append("")

    lines.append(f"<b>🔝 ۱۰ الگوی نزدیک‌تر (کندل آخر الگو):</b>")
    for r in res[:10]:
        d = r['date'].strftime('%Y-%m-%d') \
            if hasattr(r['date'], 'strftime') else str(r['date'])
        lines.append(f"• <code>{r['symbol']}</code> | {d}")
        lines.append(f"   بازدهی: <code>{r['ret']:+.1f}٪</code> | "
                     f"رشد: <code>{r['max_up']:+.1f}٪</code> | "
                     f"ریزش: <code>{r['max_dd']:+.1f}٪</code>")
    lines.append("")
    lines.append("⚠️ صرفاً تحلیلی است؛ توصیه‌ی معاملاتی نیست.")

    return "\n".join(lines)


# ==================== اجرای اصلی ====================
def main():
    pat_label = "تک‌کندلی" if PATTERN_LEN == 1 else f"{PATTERN_LEN}-کندلی"
    log("=" * 60)
    log(f"🚀 مشابه‌یابی الگوی {pat_label} {TARGET_SYMBOL}")
    log("=" * 60)

    dfs = {}
    for sym in SYMBOLS:
        df = fetch(sym, INTERVAL, PERIOD)
        if df is not None and len(df) > LOOKAHEAD + 10:
            dfs[sym] = df

    if TARGET_SYMBOL not in dfs:
        raise RuntimeError(f"داده‌ی {TARGET_SYMBOL} موجود نیست.")

    df_target = dfs[TARGET_SYMBOL]
    n_t = len(df_target)
    idx_naive = df_target.index
    if getattr(idx_naive, 'tz', None) is not None:
        idx_naive = idx_naive.tz_localize(None)
    deltas = np.abs((idx_naive - pd.Timestamp(TARGET_DATE)).total_seconds())
    target_idx = int(np.argmin(deltas))

    if target_idx < PATTERN_LEN - 1:
        raise RuntimeError(
            f"کندل هدف ({TARGET_DATE}) باید حداقل "
            f"{PATTERN_LEN - 1} کندل قبل از خود داشته باشد."
        )

    log(f"🎯 کندل آخر الگو: {df_target.index[target_idx].date()}")
    if PATTERN_LEN > 1:
        log(f"   کندل اول الگو: "
            f"{df_target.index[target_idx - PATTERN_LEN + 1].date()}")

    o_t = df_target['Open'].values.astype(float)
    h_t = df_target['High'].values.astype(float)
    l_t = df_target['Low'].values.astype(float)
    c_t = df_target['Close'].values.astype(float)
    target_actual = analyze_forward(target_idx, o_t, h_t, l_t, c_t, n_t, LOOKAHEAD)
    target_forward = target_actual['path'] if target_actual else None

    log(f"\n🔍 جستجوی الگوهای {pat_label} مشابه در گذشته...")
    candidates = find_similar_candles(df_target, target_idx, dfs)
    if not candidates:
        raise RuntimeError("هیچ الگوی مشابهی یافت نشد.")
    log(f"\n🏆 {len(candidates)} الگوی مشابه برتر انتخاب شد.")

    results = []
    for c_ in candidates:
        sym = c_['symbol']; df = dfs[sym]
        o = df['Open'].values.astype(float)
        h = df['High'].values.astype(float)
        l = df['Low'].values.astype(float)
        cc = df['Close'].values.astype(float)
        n = len(df)
        # entry_idx = کندل آخر الگو (پس ورود از Close همان کندل)
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
        'target_actual': target_actual,
    }

    report = report_single(details, df_target, target_idx, target_forward)
    plain = (report.replace("<b>", "").replace("</b>", "")
                   .replace("<code>", "").replace("</code>", ""))
    log("\n" + plain)
    send_message(report)

    try:
        target_date_str = df_target.index[target_idx].strftime('%Y-%m-%d')
        buf = build_chart(details, target_forward, target_date_str)
        caption = (f"🕯 الگوی {pat_label} مشابه {TARGET_SYMBOL} | "
                   f"{df_target.index[target_idx].date()}\n"
                   f"الگوهای مشابه: {n_total} | صعودی: {win_rate:.0f}٪")
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
