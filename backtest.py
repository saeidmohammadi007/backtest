import os
import sys
import traceback
from datetime import datetime
import numpy as np

def log(m): print(m, flush=True)

try:
    import yfinance as yf
    import pandas as pd
    from sklearn.metrics import pairwise_distances
    import requests, io
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
except Exception as e:
    log(f"❌ import error: {e}")
    traceback.print_exc()
    sys.exit(1)

# ======================== تنظیمات ========================
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "").strip()

SYMBOL = "BTC-USD"       # نماد مورد نظر
INTERVAL = "1d"          # 🔒 فقط روزانه
PERIOD = "5y"            # ۵ سال داده تاریخی
PATTERN_LEN = 5          # طول الگو
LOOKAHEAD = 60           # بررسی ۶۰ کندل بعد
RR_RATIO = 2.0           # ریسک به ریوارد
TOP_N = 5                # تعداد الگوهای برتر

# ======================== دریافت داده ========================
log(f"📥 دریافت {SYMBOL} | interval={INTERVAL} | period={PERIOD}")
try:
    df = yf.download(SYMBOL, interval=INTERVAL, period=PERIOD,
                     progress=False, auto_adjust=True, timeout=30)
    if df is None or df.empty:
        raise ValueError("داده‌ای دریافت نشد")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df[['Open', 'High', 'Low', 'Close']].dropna()
except Exception as e:
    msg = f"❌ خطا در دریافت داده: {e}"
    log(msg)
    traceback.print_exc()
    if TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID:
        requests.post(f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage",
                      json={"chat_id": TELEGRAM_CHAT_ID, "text": msg}, timeout=15)
    sys.exit(1)

n = len(df)
log(f"✅ دریافت شد: {n} کندل | از {df.index.min().date()} تا {df.index.max().date()}")

MIN_REQUIRED = PATTERN_LEN + LOOKAHEAD + 50
if n < MIN_REQUIRED:
    msg = f"❌ داده کافی نیست. تعداد کندل: {n} (نیاز: {MIN_REQUIRED})"
    log(msg)
    if TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID:
        requests.post(f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage",
                      json={"chat_id": TELEGRAM_CHAT_ID, "text": msg}, timeout=15)
    sys.exit(1)

open_p  = df['Open'].values
high_p  = df['High'].values
low_p   = df['Low'].values
close_p = df['Close'].values

# ======================== ساخت بردار ویژگی ========================
def make_feature_vector(idx_start, length):
    o = open_p[idx_start:idx_start+length]
    h = high_p[idx_start:idx_start+length]
    l = low_p[idx_start:idx_start+length]
    c = close_p[idx_start:idx_start+length]
    vec = np.concatenate([o, h, l, c])
    std = np.std(vec)
    if std == 0:
        return np.zeros_like(vec)
    return (vec - np.mean(vec)) / std

# ======================== یافتن الگوهای مشابه ========================
log("🔍 محاسبه فاصله اقلیدسی...")
target_vec = make_feature_vector(n - PATTERN_LEN, PATTERN_LEN).reshape(1, -1)

candidate_indices = list(range(0, n - PATTERN_LEN - LOOKAHEAD))
candidate_vectors = np.array([make_feature_vector(i, PATTERN_LEN) for i in candidate_indices])
distances = pairwise_distances(target_vec, candidate_vectors, metric='euclidean')[0]

top_indices = np.argsort(distances)[:TOP_N]

# ======================== شبیه‌سازی معامله ========================
def simulate_trade(entry_idx, lookahead=LOOKAHEAD, rr=RR_RATIO):
    if entry_idx + 1 >= n:
        return None
    entry_price = float(open_p[entry_idx + 1])
    stop_loss = float(min(low_p[entry_idx:entry_idx+PATTERN_LEN]))
    risk = entry_price - stop_loss
    if risk <= 0:
        return None
    take_profit = entry_price + rr * risk
    end_idx = min(entry_idx + 1 + lookahead, n)

    for j in range(entry_idx + 1, end_idx):
        if low_p[j] <= stop_loss:
            return {'result': 'loss', 'entry': entry_price, 'exit': stop_loss,
                    'bars': j - entry_idx - 1, 'idx': entry_idx}
        if high_p[j] >= take_profit:
            return {'result': 'win', 'entry': entry_price, 'exit': take_profit,
                    'bars': j - entry_idx - 1, 'idx': entry_idx}

    exit_price = float(close_p[end_idx - 1])
    return {'result': 'win' if exit_price > entry_price else 'loss',
            'entry': entry_price, 'exit': exit_price,
            'bars': end_idx - 1 - entry_idx - 1, 'idx': entry_idx}

results = []
for i in top_indices:
    idx = candidate_indices[i]
    tr = simulate_trade(idx)
    if tr:
        results.append(tr)

# ======================== معیارها ========================
wins = sum(1 for r in results if r['result'] == 'win')
total = len(results)
win_rate = (wins / total * 100) if total > 0 else 0
avg_win = np.mean([r['exit'] - r['entry'] for r in results if r['result'] == 'win']) if wins > 0 else 0
losses = [r['entry'] - r['exit'] for r in results if r['result'] == 'loss']
avg_loss = np.mean(losses) if losses else 0
profit_factor = (avg_win * wins) / (avg_loss * (total - wins)) if (avg_loss * (total - wins)) > 0 else float('inf')

# ======================== پیام تلگرام ========================
now_str = datetime.now().strftime("%Y-%m-%d %H:%M")
message = f"""📊 <b>گزارش تحلیل کندل مشابه</b>
🗓 زمان: <code>{now_str}</code>
💱 نماد: <code>{SYMBOL}</code> | تایم‌فریم: <b>روزانه (1d)</b>
📅 بازه: <code>{df.index.min().date()}</code> → <code>{df.index.max().date()}</code>
📊 کل کندل‌ها: <code>{n}</code>
🔎 طول الگو: <code>{PATTERN_LEN}</code> | بررسی: <code>{LOOKAHEAD}</code> کندل بعد
⚖️ R:R = <code>1:{RR_RATIO}</code>

📈 <b>نتایج شبیه‌سازی:</b>
• تعداد معاملات: <code>{total}</code>
• وین ریت: <code>{win_rate:.1f}%</code>
• میانگین سود: <code>{avg_win:.2f}</code>
• میانگین ضرر: <code>{avg_loss:.2f}</code>
• Profit Factor: <code>{profit_factor:.2f}</code>

🔝 <b>{TOP_N} الگوی مشابه برتر:</b>
"""

for rank, i in enumerate(top_indices, 1):
    idx = candidate_indices[i]
    tr = next((r for r in results if r['idx'] == idx), None)
    date_str = df.index[idx].strftime('%Y-%m-%d')
    if tr:
        emoji = "🟢" if tr['result'] == 'win' else "🔴"
        message += (f"\n{rank}. تاریخ <code>{date_str}</code> | "
                    f"فاصله <code>{distances[i]:.3f}</code> | "
                    f"{emoji} <b>{tr['result'].upper()}</b> | "
                    f"{tr['bars']} کندل")

message += "\n\n⚠️ صرفاً تحلیلی است و توصیه معاملاتی نیست."

if TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID:
    try:
        r = requests.post(f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage",
                          json={"chat_id": TELEGRAM_CHAT_ID, "text": message, "parse_mode": "HTML"},
                          timeout=15)
        log(f"📤 تلگرام (پیام): {r.status_code}")
    except Exception as e:
        log(f"❌ خطا در ارسال پیام: {e}")

# ======================== نمودار ========================
try:
    log("🎨 ساخت نمودار...")
    fig, ax = plt.subplots(figsize=(13, 6))
    plot_start = max(0, n - 250)
    x = np.arange(plot_start, n)

    for i in x:
        color = '#26a69a' if close_p[i] >= open_p[i] else '#ef5350'
        ax.plot([i, i], [low_p[i], high_p[i]], color=color, linewidth=0.8)
        ax.plot([i, i], [open_p[i], close_p[i]], color=color, linewidth=3.5)

    # هایلایت الگوی فعلی
    ax.axvspan(n - PATTERN_LEN, n - 0.5, alpha=0.20, color='#2196F3',
               label=f'الگوی فعلی ({PATTERN_LEN} کندل)')

    # هایلایت الگوهای مشابه
    for i in top_indices:
        idx = candidate_indices[i]
        if idx >= plot_start:
            ax.axvspan(idx, idx + PATTERN_LEN - 0.5, alpha=0.25, color='#FF9800')

    ax.set_title(f"{SYMBOL} [Daily] - Similar Patterns | WinRate: {win_rate:.1f}% | R:R 1:{RR_RATIO}",
                 fontsize=12, fontweight='bold')
    ax.set_xlabel("Index")
    ax.set_ylabel("Price (USD)")
    ax.grid(alpha=0.3)
    ax.legend(loc='upper left')

    buf = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format='png', dpi=110)
    plt.close()
    buf.seek(0)

    if TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID:
        caption = (f"📈 <b>{SYMBOL}</b> [Daily]\n"
                   f"وین ریت: <b>{win_rate:.1f}%</b> | PF: <b>{profit_factor:.2f}</b>")
        r = requests.post(f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendPhoto",
                          files={"photo": ("chart.png", buf, "image/png")},
                          data={"chat_id": TELEGRAM_CHAT_ID, "caption": caption,
                                "parse_mode": "HTML"},
                          timeout=30)
        log(f"📤 تلگرام (نمودار): {r.status_code}")
except Exception as e:
    log(f"⚠️ خطا در ساخت/ارسال نمودار: {e}")
    traceback.print_exc()

log("✅ تمام شد!")
