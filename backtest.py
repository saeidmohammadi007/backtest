import os
import sys
import traceback
from datetime import datetime

def log(msg):
    print(msg, flush=True)

try:
    import yfinance as yf
    import pandas as pd
    import numpy as np
    from sklearn.metrics import pairwise_distances
    import requests
    import io
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    log("✅ کتابخانه‌ها با موفقیت import شدند")
except Exception as e:
    log(f"❌ خطا در import: {e}")
    traceback.print_exc()
    sys.exit(1)

# ======================== تنظیمات ========================
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "").strip()

log(f"🔑 توکن بات: {'موجود' if TELEGRAM_BOT_TOKEN else 'ناموجود ❌'}")
log(f"🔑 Chat ID: {'موجود' if TELEGRAM_CHAT_ID else 'ناموجود ❌'}")

SYMBOL = "BTC-USD"
INTERVAL = "1d"
PATTERN_LEN = 5
LOOKAHEAD = 60
RR_RATIO = 2.0
TOP_N = 5

# ======================== توابع تلگرام ========================
def send_telegram_message(text):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        log("⚠️ تنظیمات تلگرام ناقص است. پیام ارسال نمی‌شود.")
        log(f"متن پیام:\n{text}")
        return False
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    # استفاده از HTML برای جلوگیری از خطای Markdown
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": text, "parse_mode": "HTML"}
    try:
        r = requests.post(url, json=payload, timeout=15)
        log(f"📤 پاسخ تلگرام (sendMessage): {r.status_code} - {r.text[:200]}")
        return r.ok
    except Exception as e:
        log(f"❌ خطا در ارسال پیام: {e}")
        return False

def send_telegram_photo(photo_bytes, caption=""):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        return False
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendPhoto"
    files = {"photo": ("chart.png", photo_bytes, "image/png")}
    data = {"chat_id": TELEGRAM_CHAT_ID, "caption": caption[:1000], "parse_mode": "HTML"}
    try:
        r = requests.post(url, files=files, data=data, timeout=30)
        log(f"📤 پاسخ تلگرام (sendPhoto): {r.status_code} - {r.text[:200]}")
        return r.ok
    except Exception as e:
        log(f"❌ خطا در ارسال عکس: {e}")
        return False

# ======================== دریافت داده ========================
try:
    log("📥 دریافت داده از yfinance...")
    df = yf.download(SYMBOL, interval=INTERVAL, progress=False, auto_adjust=True, timeout=30)
    if df is None or df.empty:
        raise ValueError("داده‌ای از yfinance دریافت نشد. شاید نماد اشتباه است یا API محدود شده.")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df[['Open', 'High', 'Low', 'Close']].dropna()
    log(f"✅ تعداد کندل دریافت‌شده: {len(df)}")
except Exception as e:
    log(f"❌ خطا در دریافت داده: {e}")
    traceback.print_exc()
    send_telegram_message(f"❌ خطا در دریافت داده از yfinance:\n<code>{e}</code>")
    sys.exit(1)

if len(df) < PATTERN_LEN + LOOKAHEAD + 50:
    msg = f"❌ داده کافی نیست. تعداد کندل: {len(df)}"
    log(msg)
    send_telegram_message(msg)
    sys.exit(1)

open_p = df['Open'].values
high_p = df['High'].values
low_p = df['Low'].values
close_p = df['Close'].values
n = len(close_p)

# ======================== تحلیل ========================
try:
    def make_feature_vector(idx_start, length):
        o = open_p[idx_start:idx_start+length]
        h = high_p[idx_start:idx_start+length]
        l = low_p[idx_start:idx_start+length]
        c = close_p[idx_start:idx_start+length]
        vec = np.concatenate([o, h, l, c])
        if np.std(vec) == 0:
            return np.zeros_like(vec)
        return (vec - np.mean(vec)) / np.std(vec)

    target_vec = make_feature_vector(n - PATTERN_LEN, PATTERN_LEN).reshape(1, -1)
    candidate_indices = range(0, n - PATTERN_LEN - LOOKAHEAD)
    candidate_vectors = []
    valid_indices = []
    for idx in candidate_indices:
        candidate_vectors.append(make_feature_vector(idx, PATTERN_LEN))
        valid_indices.append(idx)

    candidate_vectors = np.array(candidate_vectors)
    distances = pairwise_distances(target_vec, candidate_vectors, metric='euclidean')[0]
    sorted_idx = np.argsort(distances)
    top_indices = sorted_idx[:TOP_N]

    def simulate_trade(entry_idx, lookahead=LOOKAHEAD, rr=RR_RATIO):
        if entry_idx + 1 >= n:
            return None
        entry_price = open_p[entry_idx + 1]
        stop_loss = min(low_p[entry_idx:entry_idx+PATTERN_LEN])
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
        exit_price = close_p[end_idx - 1]
        pnl = exit_price - entry_price
        return {'result': 'win' if pnl > 0 else 'loss', 'entry': entry_price,
                'exit': exit_price, 'bars': end_idx - 1 - entry_idx - 1, 'idx': entry_idx}

    results = []
    for i in top_indices:
        idx = valid_indices[i]
        trade = simulate_trade(idx)
        if trade:
            results.append(trade)
    log(f"✅ شبیه‌سازی {len(results)} معامله انجام شد")
except Exception as e:
    log(f"❌ خطا در تحلیل: {e}")
    traceback.print_exc()
    send_telegram_message(f"❌ خطا در تحلیل داده:\n<code>{e}</code>")
    sys.exit(1)

# ======================== محاسبه معیارها ========================
wins = sum(1 for r in results if r['result'] == 'win')
total = len(results)
win_rate = (wins / total * 100) if total > 0 else 0
avg_win = np.mean([r['exit'] - r['entry'] for r in results if r['result'] == 'win']) if wins > 0 else 0
losses_list = [r['entry'] - r['exit'] for r in results if r['result'] == 'loss']
avg_loss = np.mean(losses_list) if losses_list else 0
profit_factor = (avg_win * wins) / (avg_loss * (total - wins)) if (avg_loss * (total - wins)) > 0 else float('inf')

# ======================== ساخت پیام (HTML) ========================
now_str = datetime.now().strftime("%Y-%m-%d %H:%M")

message = f"""📊 <b>گزارش تحلیل کندل مشابه</b>
🗓 زمان: <code>{now_str}</code>
💱 نماد: <code>{SYMBOL}</code> | تایم‌فریم: <code>{INTERVAL}</code>
🔎 طول الگو: <code>{PATTERN_LEN}</code> کندل | بررسی: <code>{LOOKAHEAD}</code> کندل بعد
⚖️ ریسک به ریوارد: <code>1:{RR_RATIO}</code>

📈 <b>نتایج شبیه‌سازی:</b>
• تعداد معاملات: <code>{total}</code>
• وین ریت: <code>{win_rate:.1f}%</code>
• میانگین سود: <code>{avg_win:.2f}</code>
• میانگین ضرر: <code>{avg_loss:.2f}</code>
• Profit Factor: <code>{profit_factor:.2f}</code>

🔝 <b>{TOP_N} الگوی مشابه برتر:</b>
"""

for rank, i in enumerate(top_indices, 1):
    idx = valid_indices[i]
    trade = next((r for r in results if r['idx'] == idx), None)
    if trade:
        emoji = "🟢" if trade['result'] == 'win' else "🔴"
        message += f"\n{rank}. ایندکس <code>{idx}</code> | فاصله <code>{distances[i]:.3f}</code> | {emoji} <b>{trade['result'].upper()}</b> | {trade['bars']} کندل"

message += "\n\n⚠️ این گزارش صرفاً تحلیلی است و توصیه معاملاتی نیست."

# ======================== ارسال ========================
log("📤 ارسال پیام به تلگرام...")
send_telegram_message(message)

# ======================== نمودار ========================
try:
    log("🎨 ساخت نمودار...")
    fig, ax = plt.subplots(figsize=(12, 5))
    plot_start = max(0, n - 200)
    x_axis = range(plot_start, n)
    for i in x_axis:
        color = 'green' if close_p[i] >= open_p[i] else 'red'
        ax.plot([i, i], [low_p[i], high_p[i]], color=color, linewidth=0.7)
        ax.plot([i, i], [open_p[i], close_p[i]], color=color, linewidth=3)
    ax.axvspan(n - PATTERN_LEN, n, alpha=0.15, color='blue', label='Current pattern')
    for i in top_indices:
        idx = valid_indices[i]
        if idx >= plot_start:
            ax.axvspan(idx, idx + PATTERN_LEN, alpha=0.25, color='orange')
    ax.set_title(f"{SYMBOL} - Similar patterns (Euclidean)")
    ax.set_xlabel("Index"); ax.set_ylabel("Price"); ax.grid(alpha=0.3)
    buf = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format='png', dpi=110)
    plt.close()
    buf.seek(0)
    log("📤 ارسال نمودار به تلگرام...")
    send_telegram_photo(buf, caption=f"📈 {SYMBOL} | Win rate: {win_rate:.1f}%")
except Exception as e:
    log(f"⚠️ خطا در ساخت نمودار: {e}")
    traceback.print_exc()

log("✅ تمام شد!")
