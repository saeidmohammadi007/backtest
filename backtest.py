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

SYMBOL       = "ETH-USD"        # ارز
TARGET_DATE  = "2025-07-04"     # کندل هدف
INTERVAL     = "1d"             # تایم‌فریم روزانه
PERIOD       = "10y"            # بازه تاریخی

PATTERN_LEN  = 5                # تعداد کندل‌های الگو (شامل کندل هدف)
LOOKAHEAD    = 100              # تعداد کندل آینده برای بررسی
K_NEIGHBORS  = 20               # تعداد نزدیک‌ترین الگوهای مشابه
GAP          = PATTERN_LEN + 5  # فاصله‌ی حداقلی برای جلوگیری از overlap


# ==================== توابع ====================
def fetch(symbol, interval, period):
    log(f"📥 دریافت داده {symbol} | {interval} | {period}")
    df = yf.download(symbol, interval=interval, period=period,
                     progress=False, auto_adjust=True, timeout=60)
    if df is None or df.empty:
        raise RuntimeError("داده‌ای دریافت نشد")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df[['Open', 'High', 'Low', 'Close']].dropna()
    return df


def make_vec(o, h, l, c, start, length):
    """ساخت بردار نرمال‌شده از کندل‌های یک پنجره"""
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
    """
    ورود در بسته‌شدن کندل entry_idx، بررسی lookahead کندل بعدی.
    """
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


def build_chart(df, target_idx, results, target_forward):
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    fig.patch.set_facecolor('#f5f5f5')

    # --- بالایی: تاریخچه‌ی قیمت ---
    ax = axes[0]
    idx_arr = np.arange(len(df))
    ax.plot(idx_arr, df['Close'].values, color='#333', linewidth=0.9)
    ax.axvline(target_idx, color='#2196F3', linestyle='--', linewidth=2.0,
               label=f'Target: {TARGET_DATE}')
    for r in results:
        ax.axvline(r['end_idx'], color='#FF9800', alpha=0.12, linewidth=0.8)
    ax.set_title(f"{SYMBOL} Daily – Price History (orange lines = similar patterns)",
                 fontweight='bold')
    ax.set_xlabel('Index')
    ax.set_ylabel('Price (USD)')
    ax.grid(alpha=0.3)
    ax.legend(loc='upper left')

    # --- پایینی: مسیرهای آینده ---
    ax = axes[1]
    x = np.arange(LOOKAHEAD + 1)
    for r in results:
        color = '#26a69a' if r['direction'] == 'up' else '#ef5350'
        ax.plot(x, r['path'], color=color, alpha=0.45, linewidth=1.0)
    if target_forward is not None:
        ax.plot(x, target_forward, color='#2196F3', linewidth=2.5,
                label='Target (actual forward)')
    ax.axhline(100, color='#888', linestyle=':', linewidth=1)
    ax.set_title(f"Forward path over next {LOOKAHEAD} candles (entry = 100)",
                 fontweight='bold')
    ax.set_xlabel('Bars after pattern')
    ax.set_ylabel('Normalized price')
    ax.grid(alpha=0.3)
    ax.legend(loc='upper left')

    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=110, facecolor='#f5f5f5')
    plt.close()
    buf.seek(0)
    return buf


def send_message(text):
    if not (TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID):
        log("⚠️ توکن یا chat_id موجود نیست؛ پیام فقط چاپ شد.")
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


# ==================== اجرای اصلی ====================
def main():
    log("=" * 60)
    log(f"🚀 تحلیل الگوی {SYMBOL} – کندل {TARGET_DATE}")
    log("=" * 60)

    df = fetch(SYMBOL, INTERVAL, PERIOD)
    n = len(df)
    log(f"📊 تعداد کندل‌ها: {n} | بازه: {df.index[0].date()} تا {df.index[-1].date()}")

    # پیدا کردن کندل هدف (نزدیک‌ترین کندل به تاریخ موردنظر)
    idx_naive = df.index
    if getattr(idx_naive, 'tz', None) is not None:
        idx_naive = idx_naive.tz_localize(None)
    target_ts = pd.Timestamp(TARGET_DATE)
    deltas = np.abs((idx_naive - target_ts).total_seconds())
    target_idx = int(np.argmin(deltas))
    log(f"🎯 کندل هدف پیدا شد: {df.index[target_idx].date()}  (index={target_idx})")

    if target_idx < PATTERN_LEN - 1:
        raise RuntimeError("قبل از کندل هدف داده کافی وجود ندارد.")

    o = df['Open'].values.astype(float)
    h = df['High'].values.astype(float)
    l = df['Low'].values.astype(float)
    c = df['Close'].values.astype(float)

    # بردار الگوی هدف
    pat_start = target_idx - PATTERN_LEN + 1
    target_vec = make_vec(o, h, l, c, pat_start, PATTERN_LEN)

    # کاندیدهای تاریخی
    valid_starts = []
    max_start = n - PATTERN_LEN - LOOKAHEAD
    for s in range(0, max_start):
        pat_end_s = s + PATTERN_LEN - 1
        if abs(pat_end_s - target_idx) < GAP:   # جلوگیری از overlap با هدف
            continue
        valid_starts.append(s)
    valid_starts = np.array(valid_starts, dtype=int)

    if len(valid_starts) == 0:
        raise RuntimeError("الگوی تاریخی معتبری برای مقایسه وجود ندارد.")
    log(f"🔍 در حال بررسی {len(valid_starts)} الگوی تاریخی...")

    # فاصله اقلیدسی
    vecs = np.array([make_vec(o, h, l, c, s, PATTERN_LEN) for s in valid_starts])
    dists = np.sqrt(np.sum((vecs - target_vec) ** 2, axis=1))

    k = min(K_NEIGHBORS, len(valid_starts))
    order = np.argsort(dists)[:k]
    neighbors = valid_starts[order]
    neighbor_dists = dists[order]

    # تحلیل آینده‌ی هر همسایه
    results = []
    for s, d in zip(neighbors, neighbor_dists):
        end_idx = int(s + PATTERN_LEN - 1)
        r = analyze_forward(end_idx, o, h, l, c, n, LOOKAHEAD)
        if r is None:
            continue
        r['start_idx'] = int(s)
        r['end_idx']   = end_idx
        r['date']      = df.index[end_idx]
        r['dist']      = float(d)
        results.append(r)

    if not results:
        raise RuntimeError("هیچ نتیجه‌ای محاسبه نشد (داده‌ی کافی نبود).")

    log(f"✅ {len(results)} تطابق با داده‌ی کافی برای بررسی {LOOKAHEAD} کندل آینده")

    # ---------- آمار ----------
    ups   = [r for r in results if r['direction'] == 'up']
    downs = [r for r in results if r['direction'] == 'down']
    n_total = len(results)
    n_up    = len(ups)
    n_down  = len(downs)
    win_rate = n_up / n_total * 100.0

    avg_ret_all = float(np.mean([r['ret']    for r in results]))
    avg_max_up  = float(np.mean([r['max_up'] for r in results]))
    avg_max_dd  = float(np.mean([r['max_dd'] for r in results]))

    if ups:
        avg_ret_up    = float(np.mean([r['ret']    for r in ups]))
        avg_max_up_up = float(np.mean([r['max_up'] for r in ups]))
        avg_dd_up     = float(np.mean([r['max_dd'] for r in ups]))
        worst_dd_up   = float(np.min([r['max_dd']  for r in ups]))
        best_ret_up   = float(np.max([r['ret']     for r in ups]))
    else:
        avg_ret_up = avg_max_up_up = avg_dd_up = worst_dd_up = best_ret_up = 0.0

    if downs:
        avg_ret_down    = float(np.mean([r['ret']    for r in downs]))
        avg_dd_down     = float(np.mean([r['max_dd'] for r in downs]))
        worst_dd_down   = float(np.min([r['max_dd']  for r in downs]))
        worst_ret_down  = float(np.min([r['ret']     for r in downs]))
    else:
        avg_ret_down = avg_dd_down = worst_dd_down = worst_ret_down = 0.0

    # مسیر آینده‌ی خود هدف (اگر داده موجود باشد)
    target_actual = None
    target_forward = None
    if target_idx + 1 + LOOKAHEAD <= n:
        target_actual = analyze_forward(target_idx, o, h, l, c, n, LOOKAHEAD)
        if target_actual:
            target_forward = target_actual['path']

    # ---------- پیام تلگرام ----------
    lines = []
    lines.append(f"📊 <b>تحلیل الگوی {SYMBOL}</b>")
    lines.append(f"🗓 کندل هدف: <code>{df.index[target_idx].date()}</code>")
    lines.append(f"🧩 طول الگو: <code>{PATTERN_LEN}</code> کندل | ⏭ بررسی <code>{LOOKAHEAD}</code> کندل بعدی")
    lines.append(f"🔍 <code>{n_total}</code> الگوی تاریخی مشابه (K = {k})")
    lines.append("")
    lines.append("<b>📈 آمار کلی:</b>")
    lines.append(f"├ 🟢 صعودی: <code>{n_up}</code> ({win_rate:.1f}%)")
    lines.append(f"├ 🔴 نزولی: <code>{n_down}</code> ({100 - win_rate:.1f}%)")
    lines.append(f"├ میانگین بازدهی: <code>{avg_ret_all:+.2f}%</code>")
    lines.append(f"├ میانگین بیشترین رشد: <code>{avg_max_up:+.2f}%</code>")
    lines.append(f"└ میانگین بیشترین ریزش: <code>{avg_max_dd:+.2f}%</code>")
    lines.append("")

    if ups:
        lines.append("<b>🟢 گروه صعودی‌ها:</b>")
        lines.append(f"├ میانگین بازدهی: <code>{avg_ret_up:+.2f}%</code>")
        lines.append(f"├ میانگین بیشترین رشد: <code>{avg_max_up_up:+.2f}%</code>")
        lines.append(f"├ میانگین بیشترین ریزش: <code>{avg_dd_up:+.2f}%</code>")
        lines.append(f"├ 📉 بدترین ریزش در صعودی‌ها: <code>{worst_dd_up:+.2f}%</code>")
        lines.append(f"└ بهترین بازدهی: <code>{best_ret_up:+.2f}%</code>")
        lines.append("")

    if downs:
        lines.append("<b>🔴 گروه نزولی‌ها:</b>")
        lines.append(f"├ میانگین بازدهی: <code>{avg_ret_down:+.2f}%</code>")
        lines.append(f"├ میانگین بیشترین ریزش: <code>{avg_dd_down:+.2f}%</code>")
        lines.append(f"├ 📉 بدترین ریزش: <code>{worst_dd_down:+.2f}%</code>")
        lines.append(f"└ بدترین بازدهی: <code>{worst_ret_down:+.2f}%</code>")
        lines.append("")

    if target_actual:
        lines.append("<b>🔵 مسیر واقعی کندل هدف تا الان:</b>")
        lines.append(f"├ بازدهی: <code>{target_actual['ret']:+.2f}%</code>")
        lines.append(f"├ بیشترین رشد: <code>{target_actual['max_up']:+.2f}%</code>")
        lines.append(f"└ بیشترین ریزش: <code>{target_actual['max_dd']:+.2f}%</code>")

    lines.append("")
    lines.append("⚠️ صرفاً تحلیلی است؛ توصیه‌ی معاملاتی نیست.")

    message = "\n".join(lines)

    # لاگ متنی
    plain = (message.replace("<b>", "").replace("</b>", "")
                    .replace("<code>", "").replace("</code>", ""))
    log("\n" + plain)

    send_message(message)

    # ---------- نمودار ----------
    try:
        buf = build_chart(df, target_idx, results, target_forward)
        caption = (f"📈 {SYMBOL} | Target: {df.index[target_idx].date()}\n"
                   f"تطابق: {n_total} | صعودی: {win_rate:.0f}% | "
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
