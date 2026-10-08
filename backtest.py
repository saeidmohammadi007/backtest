import os
import sys
import io
import traceback
from datetime import datetime

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
TARGET_DATE   = "2025-07-04"

SYMBOLS = [
    "BTC-USD", "ETH-USD", "BNB-USD", "SOL-USD", "XRP-USD",
    "ADA-USD", "DOGE-USD", "AVAX-USD", "DOT-USD", "LINK-USD",
]

INTERVAL     = "1d"
PERIOD       = "10y"

PATTERN_LEN  = 5
LOOKAHEAD    = 100
K_PER_SYMBOL = 20
TOTAL_TOP    = 100
GAP          = PATTERN_LEN + 5

# ---------- سطل‌های ریز (۱٪) ----------
def build_buckets(start, end, step=1.0):
    """ساخت سطل‌های [start, start+step), ... , [end-step, end)"""
    edges = []
    v = start
    while v < end:
        edges.append((v, v + step))
        v += step
    return edges

# ریزش‌ها: از -50% تا 0% با گام 1٪
DD_BUCKETS = build_buckets(-50, 0, 1.0)
DD_BUCKETS[0] = (-1e9, -50)  # اولین سطل: هر چیزی کمتر از -50
# رشدها: از 0% تا 50% با گام 1٪ + یک سطل باز برای ≥50
UP_BUCKETS = build_buckets(0, 50, 1.0)
UP_BUCKETS.append((50, 1e9))
# بازدهی نهایی: از -50% تا +50% با گام 1٪ + دو سطل باز
RET_BUCKETS = build_buckets(-50, 50, 1.0)
RET_BUCKETS.insert(0, (-1e9, -50))
RET_BUCKETS.append((50, 1e9))


# ==================== توابع ====================
def fetch(symbol, interval, period):
    log(f"📥 دریافت {symbol} | {interval} | {period}")
    try:
        df = yf.download(symbol, interval=interval, period=period,
                         progress=False, auto_adjust=True, timeout=60)
        if df is None or df.empty:
            log(f"⚠️ {symbol}: داده‌ای دریافت نشد")
            return None
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df[['Open', 'High', 'Low', 'Close']].dropna()
        return df
    except Exception as e:
        log(f"❌ fetch {symbol}: {e}")
        return None


def make_vec(o, h, l, c, start, length):
    vec = np.concatenate([
        o[start:start + length],
        h[start:start + length],
        l[start:start + length],
        c[start:start + length],
    ]).astype(float)
    std = vec.std()
    if std == 0:
        return vec
    return (vec - vec.mean()) / std


def analyze_forward(entry_idx, o, h, l, c, n, lookahead):
    if entry_idx + 1 + lookahead > n:
        return None
    entry_price = float(c[entry_idx])
    end = entry_idx + 1 + lookahead
    future_c = c[entry_idx + 1:end]
    future_h = h[entry_idx + 1:end]
    future_l = l[entry_idx + 1:end]
    if len(future_c) == 0:
        return None

    final_price = float(future_c[-1])
    total_ret   = (final_price - entry_price) / entry_price * 100.0
    max_up      = (float(np.max(future_h)) - entry_price) / entry_price * 100.0
    max_dd      = (float(np.min(future_l)) - entry_price) / entry_price * 100.0

    path = np.concatenate([[entry_price], future_c])
    norm_path = path / entry_price * 100.0

    return {
        'entry':     entry_price,
        'final':     final_price,
        'ret':       total_ret,
        'max_up':    max_up,
        'max_dd':    max_dd,
        'direction': 'up' if total_ret > 0 else 'down',
        'path':      norm_path,
    }


def bucketize(values, buckets):
    """برمی‌گرداند لیست (label, count, pct) — فقط سطل‌های غیرخالی"""
    total = len(values)
    if total == 0:
        return []
    out = []
    for lo, hi in buckets:
        # برچسب‌گذاری
        if lo <= -1e8:
            label = f"≤ {hi:+.0f}%"
        elif hi >= 1e8:
            label = f"≥ {lo:+.0f}%"
        else:
            label = f"{lo:+.0f}% تا {hi:+.0f}%"
        cnt = sum(1 for v in values if lo <= v < hi)
        if cnt > 0:  # فقط سطل‌های غیرخالی
            out.append((label, cnt, cnt / total * 100.0))
    return out


def fmt_bucket_lines(buckets_result, emoji="", max_bar=20):
    """نمایش سطرها با نوار متناسب"""
    lines = []
    if not buckets_result:
        return lines
    max_pct = max(p for _, _, p in buckets_result)
    for label, cnt, pct in buckets_result:
        bar_len = int(round(pct / max_pct * max_bar)) if max_pct > 0 else 0
        bar = "█" * bar_len
        lines.append(f"├ {label:<14} : <code>{cnt:>3}</code> ({pct:5.1f}%) {emoji}{bar}")
    return lines


def send_message(text):
    if not (TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID):
        log("⚠️ توکن یا chat_id موجود نیست.")
        return
    try:
        r = requests.post(
            f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage",
            json={"chat_id": TELEGRAM_CHAT_ID, "text": text, "parse_mode": "HTML"},
            timeout=30,
        )
        log(f"📤 telegram message: {r.status_code}")
    except Exception as e:
        log(f"❌ send msg: {e}")


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


def build_chart(results, target_forward, target_date_str):
    fig, axes = plt.subplots(2, 3, figsize=(20, 11))
    fig.patch.set_facecolor('#f5f5f5')

    symbols_used = sorted({r['symbol'] for r in results})
    palette = plt.cm.tab10(np.linspace(0, 1, max(len(symbols_used), 1)))
    color_map = {s: palette[i % len(palette)] for i, s in enumerate(symbols_used)}

    x = np.arange(LOOKAHEAD + 1)

    # (0,0)
    ax = axes[0, 0]
    for r in results:
        ax.plot(x, r['path'], color=color_map[r['symbol']], alpha=0.20, linewidth=0.8)
    if target_forward is not None:
        ax.plot(x, target_forward, color='#000000', linewidth=3.0,
                label=f'Target {TARGET_SYMBOL}')
    ax.axhline(100, color='#888', linestyle=':', linewidth=1)
    ax.set_title(f"All {len(results)} forward paths (entry=100)", fontweight='bold')
    ax.set_xlabel('Bars after pattern'); ax.set_ylabel('Normalized price')
    ax.grid(alpha=0.3); ax.legend(loc='upper left')

    # (0,1)
    ax = axes[0, 1]
    paths = np.array([r['path'] for r in results])
    mean_all = paths.mean(axis=0)
    p10 = np.percentile(paths, 10, axis=0)
    p50 = np.percentile(paths, 50, axis=0)
    p90 = np.percentile(paths, 90, axis=0)
    ax.fill_between(x, p10, p90, color='#2196F3', alpha=0.15, label='10–90% band')
    ax.plot(x, p50, color='#1976D2', linewidth=2.0, label='Median')
    ax.plot(x, mean_all, color='#0D47A1', linewidth=2.5, label='Mean')
    if target_forward is not None:
        ax.plot(x, target_forward, color='#000000', linewidth=2.5,
                linestyle='--', label='Target actual')
    ax.axhline(100, color='#888', linestyle=':', linewidth=1)
    ax.set_title("Overall mean / median / band", fontweight='bold')
    ax.set_xlabel('Bars after pattern'); ax.set_ylabel('Normalized price')
    ax.grid(alpha=0.3); ax.legend(loc='upper left')

    # (0,2)
    ax = axes[0, 2]
    ups   = [r for r in results if r['direction'] == 'up']
    downs = [r for r in results if r['direction'] == 'down']
    if ups:
        mean_up = np.mean([r['path'] for r in ups], axis=0)
        ax.plot(x, mean_up, color='#26a69a', linewidth=2.5, label=f"Mean UP ({len(ups)})")
        ax.fill_between(x, np.min([r['path'] for r in ups], axis=0),
                        np.max([r['path'] for r in ups], axis=0),
                        color='#26a69a', alpha=0.10)
    if downs:
        mean_down = np.mean([r['path'] for r in downs], axis=0)
        ax.plot(x, mean_down, color='#ef5350', linewidth=2.5, label=f"Mean DOWN ({len(downs)})")
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

    # (1,0) توزیع max_dd — ریزتر
    ax = axes[1, 0]
    dd_vals = [r['max_dd'] for r in results]
    ax.hist(dd_vals, bins=40, color='#ef5350', alpha=0.7, edgecolor='white')
    ax.axvline(np.mean(dd_vals), color='#b71c1c', linestyle='--',
               label=f"Mean {np.mean(dd_vals):.1f}%")
    ax.axvline(np.min(dd_vals), color='#000000', linestyle=':',
               label=f"Worst {np.min(dd_vals):.1f}%")
    ax.set_title("Distribution of max drawdown (40 bins)", fontweight='bold')
    ax.set_xlabel('Max drawdown (%)'); ax.set_ylabel('Count')
    ax.grid(alpha=0.3); ax.legend(loc='upper left')

    # (1,1)
    ax = axes[1, 1]
    up_vals = [r['max_up'] for r in results]
    ax.hist(up_vals, bins=40, color='#26a69a', alpha=0.7, edgecolor='white')
    ax.axvline(np.mean(up_vals), color='#004d40', linestyle='--',
               label=f"Mean +{np.mean(up_vals):.1f}%")
    ax.axvline(np.max(up_vals), color='#000000', linestyle=':',
               label=f"Best +{np.max(up_vals):.1f}%")
    ax.set_title("Distribution of max upside (40 bins)", fontweight='bold')
    ax.set_xlabel('Max upside (%)'); ax.set_ylabel('Count')
    ax.grid(alpha=0.3); ax.legend(loc='upper left')

    # (1,2)
    ax = axes[1, 2]
    ret_vals = [r['ret'] for r in results]
    ax.hist(ret_vals, bins=40, color='#2196F3', alpha=0.7, edgecolor='white')
    ax.axvline(0, color='#000', linestyle=':')
    ax.axvline(np.mean(ret_vals), color='#0D47A1', linestyle='--',
               label=f"Mean {np.mean(ret_vals):+.1f}%")
    ax.set_title("Distribution of final return (40 bins)", fontweight='bold')
    ax.set_xlabel('Final return (%)'); ax.set_ylabel('Count')
    ax.grid(alpha=0.3); ax.legend(loc='upper left')

    plt.suptitle(f"Top {len(results)} similar patterns across 10 cryptos | "
                 f"Target: {TARGET_SYMBOL} {target_date_str}",
                 fontsize=13, fontweight='bold')
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100, facecolor='#f5f5f5')
    plt.close()
    buf.seek(0)
    return buf


# ==================== اجرای اصلی ====================
def main():
    log("=" * 60)
    log(f"🚀 تحلیل مشابه‌یابی الگوی {TARGET_SYMBOL} در {len(SYMBOLS)} ارز")
    log("=" * 60)

    df_target = fetch(TARGET_SYMBOL, INTERVAL, PERIOD)
    if df_target is None:
        raise RuntimeError(f"داده‌ی {TARGET_SYMBOL} دریافت نشد.")

    n_t = len(df_target)
    idx_naive = df_target.index
    if getattr(idx_naive, 'tz', None) is not None:
        idx_naive = idx_naive.tz_localize(None)
    target_ts = pd.Timestamp(TARGET_DATE)
    deltas = np.abs((idx_naive - target_ts).total_seconds())
    target_idx = int(np.argmin(deltas))
    log(f"🎯 کندل هدف {TARGET_SYMBOL}: {df_target.index[target_idx].date()}")

    o_t = df_target['Open'].values.astype(float)
    h_t = df_target['High'].values.astype(float)
    l_t = df_target['Low'].values.astype(float)
    c_t = df_target['Close'].values.astype(float)

    pat_start = target_idx - PATTERN_LEN + 1
    target_vec = make_vec(o_t, h_t, l_t, c_t, pat_start, PATTERN_LEN)
    target_actual = analyze_forward(target_idx, o_t, h_t, l_t, c_t, n_t, LOOKAHEAD)
    target_forward = target_actual['path'] if target_actual else None

    candidates_all = []
    dfs = {}

    for sym in SYMBOLS:
        df = fetch(sym, INTERVAL, PERIOD)
        if df is None or len(df) < PATTERN_LEN + LOOKAHEAD + 5:
            continue
        dfs[sym] = df
        o = df['Open'].values.astype(float)
        h = df['High'].values.astype(float)
        l = df['Low'].values.astype(float)
        c = df['Close'].values.astype(float)
        n = len(df)

        max_start = n - PATTERN_LEN - LOOKAHEAD
        if max_start <= 0:
            continue
        starts = np.arange(0, max_start)
        if sym == TARGET_SYMBOL:
            keep = np.array([abs(s + PATTERN_LEN - 1 - target_idx) >= GAP
                             for s in starts])
            starts = starts[keep]
        if len(starts) == 0:
            continue

        vecs = np.array([make_vec(o, h, l, c, int(s), PATTERN_LEN) for s in starts])
        dists = np.sqrt(np.sum((vecs - target_vec) ** 2, axis=1))

        k = min(K_PER_SYMBOL, len(starts))
        order = np.argsort(dists)[:k]
        for oi in order:
            candidates_all.append({
                'symbol': sym,
                'start':  int(starts[oi]),
                'dist':   float(dists[oi]),
            })
        log(f"   ✔️ {sym}: {k} الگو")

    candidates_all.sort(key=lambda x: x['dist'])
    top = candidates_all[:TOTAL_TOP]
    log(f"🏆 {len(top)} الگوی برتر انتخاب شد")

    results = []
    for c_ in top:
        sym = c_['symbol']; df = dfs[sym]
        o = df['Open'].values.astype(float)
        h = df['High'].values.astype(float)
        l = df['Low'].values.astype(float)
        cc = df['Close'].values.astype(float)
        n = len(df)
        end_idx = c_['start'] + PATTERN_LEN - 1
        r = analyze_forward(end_idx, o, h, l, cc, n, LOOKAHEAD)
        if r is None:
            continue
        r['symbol']  = sym
        r['start']   = c_['start']
        r['end_idx'] = end_idx
        r['date']    = df.index[end_idx]
        r['dist']    = c_['dist']
        results.append(r)

    if not results:
        raise RuntimeError("هیچ نتیجه‌ای محاسبه نشد.")

    ups   = [r for r in results if r['direction'] == 'up']
    downs = [r for r in results if r['direction'] == 'down']
    n_total = len(results); n_up = len(ups); n_down = len(downs)
    win_rate = n_up / n_total * 100.0

    all_ret    = [r['ret']    for r in results]
    all_max_up = [r['max_up'] for r in results]
    all_max_dd = [r['max_dd'] for r in results]

    avg_ret_all = float(np.mean(all_ret))
    avg_max_up  = float(np.mean(all_max_up))
    avg_max_dd  = float(np.mean(all_max_dd))

    if ups:
        avg_ret_up    = float(np.mean([r['ret']    for r in ups]))
        avg_max_up_up = float(np.mean([r['max_up'] for r in ups]))
        avg_dd_up     = float(np.mean([r['max_dd'] for r in ups]))
        worst_dd_up   = float(np.min([r['max_dd']  for r in ups]))
        best_ret_up   = float(np.max([r['ret']     for r in ups]))
    else:
        avg_ret_up = avg_max_up_up = avg_dd_up = worst_dd_up = best_ret_up = 0.0

    if downs:
        avg_ret_down  = float(np.mean([r['ret']    for r in downs]))
        avg_dd_down   = float(np.mean([r['max_dd'] for r in downs]))
        worst_dd_down = float(np.min([r['max_dd']  for r in downs]))
        worst_ret_down= float(np.min([r['ret']     for r in downs]))
    else:
        avg_ret_down = avg_dd_down = worst_dd_down = worst_ret_down = 0.0

    # ---------- سطل‌های ریز ۱٪ ----------
    dd_buckets    = bucketize(all_max_dd, DD_BUCKETS)
    up_buckets    = bucketize(all_max_up, UP_BUCKETS)
    ret_buckets   = bucketize(all_ret,    RET_BUCKETS)
    dd_up_buckets = bucketize([r['max_dd'] for r in ups], DD_BUCKETS) if ups else []

    # ---------- آمار تجمعی (cumulative) برای ریزش ----------
    # مثال: «چند درصد ریزش بین 2 تا 3 درصد داشتند» — نگاه به سطل دقیق کافی است
    def pct_between(values, lo, hi):
        if not values:
            return 0.0
        cnt = sum(1 for v in values if lo <= v < hi)
        return cnt / len(values) * 100.0

    # شاخص‌های مفید
    pct_dd_lt_2  = pct_between(all_max_dd, -1e9, -2)   # ریزش بیشتر از 2%
    pct_dd_lt_5  = pct_between(all_max_dd, -1e9, -5)   # ریزش بیشتر از 5%
    pct_dd_lt_10 = pct_between(all_max_dd, -1e9, -10)  # ریزش بیشتر از 10%
    pct_dd_lt_20 = pct_between(all_max_dd, -1e9, -20)  # ریزش بیشتر از 20%

    sym_counts = {}
    for r in results:
        sym_counts[r['symbol']] = sym_counts.get(r['symbol'], 0) + 1

    # ---------- پیام ----------
    lines = []
    lines.append(f"📊 <b>مشابه‌یابی الگوی {TARGET_SYMBOL}</b>")
    lines.append(f"🗓 کندل هدف: <code>{df_target.index[target_idx].date()}</code>")
    lines.append(f"🧩 طول الگو: <code>{PATTERN_LEN}</code> | ⏭ بررسی <code>{LOOKAHEAD}</code> کندل بعد")
    lines.append(f"🌐 <code>{len(dfs)}</code> ارز | <code>{n_total}</code> الگوی برتر")
    lines.append("")

    lines.append("<b>📈 آمار کلی:</b>")
    lines.append(f"├ 🟢 صعودی: <code>{n_up}</code> ({win_rate:.1f}%)")
    lines.append(f"├ 🔴 نزولی: <code>{n_down}</code> ({100 - win_rate:.1f}%)")
    lines.append(f"├ میانگین بازدهی: <code>{avg_ret_all:+.2f}%</code>")
    lines.append(f"├ میانگین بیشترین رشد: <code>{avg_max_up:+.2f}%</code>")
    lines.append(f"└ میانگین بیشترین ریزش: <code>{avg_max_dd:+.2f}%</code>")
    lines.append("")

    # خلاصه‌ی ریزش‌های مهم
    lines.append("<b>📌 خلاصه ریزش‌ها:</b>")
    lines.append(f"├ حداقل یک‌بار بیش از ۲٪ ریزش: <code>{pct_dd_lt_2:.1f}%</code> ارزها")
    lines.append(f"├ حداقل یک‌بار بیش از ۵٪ ریزش: <code>{pct_dd_lt_5:.1f}%</code> ارزها")
    lines.append(f"├ حداقل یک‌بار بیش از ۱۰٪ ریزش: <code>{pct_dd_lt_10:.1f}%</code> ارزها")
    lines.append(f"└ حداقل یک‌بار بیش از ۲۰٪ ریزش: <code>{pct_dd_lt_20:.1f}%</code> ارزها")
    lines.append("")

    # ============ توزیع ریزش با گام ۱٪ ============
    lines.append("<b>📉 توزیع حداکثر ریزش (گام ۱٪):</b>")
    lines.extend(fmt_bucket_lines(dd_buckets, emoji="🟥"))
    lines.append(f"└ 📌 بدترین ریزش: <code>{min(all_max_dd):+.2f}%</code>")
    lines.append("")

    # ============ توزیع رشد با گام ۱٪ ============
    lines.append("<b>📈 توزیع حداکثر رشد (گام ۱٪):</b>")
    lines.extend(fmt_bucket_lines(up_buckets, emoji="🟩"))
    lines.append(f"└ 📌 بهترین رشد: <code>{max(all_max_up):+.2f}%</code>")
    lines.append("")

    # ============ توزیع بازدهی نهایی با گام ۱٪ ============
    lines.append("<b>🎯 توزیع بازدهی نهایی بعد از ۱۰۰ کندل (گام ۱٪):</b>")
    lines.extend(fmt_bucket_lines(ret_buckets, emoji="🔵"))
    lines.append("")

    if ups:
        lines.append("<b>🟢 گروه صعودی‌ها – توزیع ریزش (گام ۱٪):</b>")
        lines.append(f"├ میانگین بازدهی: <code>{avg_ret_up:+.2f}%</code>")
        lines.append(f"├ میانگین بیشترین رشد: <code>{avg_max_up_up:+.2f}%</code>")
        lines.append(f"├ میانگین بیشترین ریزش: <code>{avg_dd_up:+.2f}%</code>")
        lines.extend(fmt_bucket_lines(dd_up_buckets, emoji="🟥"))
        lines.append(f"└ 📌 بدترین ریزش در صعودی‌ها: <code>{worst_dd_up:+.2f}%</code>")
        lines.append("")

    if downs:
        lines.append("<b>🔴 گروه نزولی‌ها:</b>")
        lines.append(f"├ میانگین بازدهی: <code>{avg_ret_down:+.2f}%</code>")
        lines.append(f"├ میانگین بیشترین ریزش: <code>{avg_dd_down:+.2f}%</code>")
        lines.append(f"├ 📉 بدترین ریزش: <code>{worst_dd_down:+.2f}%</code>")
        lines.append(f"└ بدترین بازدهی: <code>{worst_ret_down:+.2f}%</code>")
        lines.append("")

    if target_actual:
        lines.append("<b>🔵 مسیر واقعی کندل هدف:</b>")
        lines.append(f"├ بازدهی تا الان: <code>{target_actual['ret']:+.2f}%</code>")
        lines.append(f"├ بیشترین رشد: <code>{target_actual['max_up']:+.2f}%</code>")
        lines.append(f"└ بیشترین ریزش: <code>{target_actual['max_dd']:+.2f}%</code>")
        lines.append("")

    lines.append("<b>🌐 توزیع ارزها:</b>")
    for s, cnt in sorted(sym_counts.items(), key=lambda x: -x[1]):
        lines.append(f"• <code>{s}</code>: {cnt}")
    lines.append("")

    lines.append("<b>🔝 10 الگوی نزدیک‌تر:</b>")
    for r in results[:10]:
        d = r['date'].strftime('%Y-%m-%d') if hasattr(r['date'], 'strftime') else str(r['date'])
        lines.append(f"• <code>{r['symbol']}</code> | {d} | "
                     f"d={r['dist']:.2f} | ret={r['ret']:+.1f}% | "
                     f"↑{r['max_up']:+.1f}% | ↓{r['max_dd']:+.1f}%")
    lines.append("")
    lines.append("⚠️ صرفاً تحلیلی است؛ توصیه‌ی معاملاتی نیست.")

    message = "\n".join(lines)
    plain = (message.replace("<b>", "").replace("</b>", "")
                    .replace("<code>", "").replace("</code>", ""))
    log("\n" + plain)
    send_message(message)

    try:
        target_date_str = df_target.index[target_idx].strftime('%Y-%m-%d')
        buf = build_chart(results, target_forward, target_date_str)
        caption = (f"📈 مشابه‌یابی {TARGET_SYMBOL} در {len(dfs)} ارز\n"
                   f"الگوها: {n_total} | صعودی: {win_rate:.0f}% | "
                   f"میانگین بازدهی: {avg_ret_all:+.1f}%")
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
