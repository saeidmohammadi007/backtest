import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.metrics import pairwise_distances
import requests
from datetime import datetime
import io
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ======================== تنظیمات تلگرام ========================
TELEGRAM_BOT_TOKEN = "YOUR_BOT_TOKEN_HERE"      # توکن بات خود را جایگزین کنید
TELEGRAM_CHAT_ID = "YOUR_CHAT_ID_HERE"          # Chat ID خود را جایگزین کنید

# ======================== تنظیمات استراتژی ========================
SYMBOL = "BTC-USD"
INTERVAL = "1d"
PATTERN_LEN = 5
LOOKAHEAD = 60
RR_RATIO = 2.0
TOP_N = 5

# ======================== توابع تلگرام ========================
def send_telegram_message(text, parse_mode="Markdown"):
    """ارسال پیام متنی به تلگرام"""
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": text,
        "parse_mode": parse_mode
    }
    try:
        r = requests.post(url, json=payload, timeout=15)
        if r.status_code != 200:
            print(f"❌ خطا در ارسال پیام: {r.text}")
        return r.ok
    except Exception as e:
        print(f"❌ خطای شبکه: {e}")
        return False

def send_telegram_photo(photo_bytes, caption=""):
    """ارسال تصویر به تلگرام"""
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendPhoto"
    files = {"photo": ("chart.png", photo_bytes, "image/png")}
    data = {"chat_id": TELEGRAM_CHAT_ID, "caption": caption, "parse_mode": "Markdown"}
    try:
        r = requests.post(url, files=files, data=data, timeout=30)
        if r.status_code != 200:
            print(f"❌ خطا در ارسال عکس: {r.text}")
        return r.ok
    except Exception as e:
        print(f"❌ خطای شبکه: {e}")
        return False

# ======================== دریافت داده ========================
print("📥 دریافت داده از yfinance...")
df = yf.download(SYMBOL, interval=INTERVAL, progress=False)
if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.get_level_values(0)
df = df[['Open', 'High', 'Low', 'Close']].dropna()
print(f"تعداد کندل: {len(df)}")

open_p = df['Open'].values
high_p = df['High'].values
low_p = df['Low'].values
close_p = df['Close'].values
n = len(close_p)

def make_feature_vector(idx_start, length):
    o = open_p[idx_start:idx_start+length]
    h = high_p[idx_start:idx_start+length]
    l = low_p[idx_start:idx_start+length]
    c = close_p[idx_start:idx_start+length]
    vec = np.concatenate([o, h, l, c])
    if np.std(vec) == 0:
        return np.zeros_like(vec)
    return (vec - np.mean(vec)) / np.std(vec)

# ======================== یافتن الگوهای مشابه ========================
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

# ======================== شبیه‌سازی معاملات ========================
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

# ======================== محاسبه معیارها ========================
wins = sum(1 for r in results if r['result'] == 'win')
total = len(results)
win_rate = (wins / total * 100) if total > 0 else 0
avg_win = np.mean([r['exit'] - r['entry'] for r in results if r['result'] == 'win']) if wins > 0 else 0
losses_list = [r['entry'] - r['exit'] for r in results if r['result'] == 'loss']
avg_loss = np.mean(losses_list) if losses_list else 0
profit_factor = (avg_win * wins) / (avg_loss * (total - wins)) if (avg_loss * (total - wins)) > 0 else float('inf')

# ======================== ساخت پیام تلگرام ========================
now_str = datetime.now().strftime("%Y-%m-%d %H:%M")
rr_str = f"1:{RR_RATIO}"

message = f"""📊 *گزارش تحلیل کندل مشابه*
🗓 زمان: `{now_str}`
💱 نماد: `{SYMBOL}` | تایم‌فریم: `{INTERVAL}`
🔎 طول الگو: `{PATTERN_LEN}` کندل | بررسی: `{LOOKAHEAD}` کندل بعد
⚖️ ریسک به ریوارد: `{rr_str}`

📈 *نتایج شبیه‌سازی:*
• تعداد معاملات: `{total}`
• وین ریت: `{win_rate:.1f}%` ✅
• میانگین سود: `{avg_win:.2f}`
• میانگین ضرر: `{avg_loss:.2f}`
• Profit Factor: `{profit_factor:.2f}`

🔝 *{TOP_N} الگوی مشابه برتر:*
"""

for rank, i in enumerate(top_indices, 1):
    idx = valid_indices[i]
    trade = next((r for r in results if r['idx'] == idx), None)
    if trade:
        emoji = "🟢" if trade['result'] == 'win' else "🔴"
        message += f"\n{rank}. ایندکس `{idx}` | فاصله `{distances[i]:.3f}` | {emoji} `{trade['result'].upper()}` | {trade['bars']} کندل"

message += "\n\n_⚠️ این گزارش صرفاً تحلیلی است و توصیه معاملاتی نیست._"

# ======================== ارسال به تلگرام ========================
print("\n📤 ارسال پیام به تلگرام...")
send_telegram_message(message)

# ======================== ساخت نمودار ========================
print("🎨 ساخت نمودار...")
fig, ax = plt.subplots(figsize=(12, 5))
plot_start = max(0, n - 200)
x_axis = range(plot_start, n)

# رسم کندل‌ها
for i in x_axis:
    color = 'green' if close_p[i] >= open_p[i] else 'red'
    ax.plot([i, i], [low_p[i], high_p[i]], color=color, linewidth=0.7)
    ax.plot([i, i], [open_p[i], close_p[i]], color=color, linewidth=3)

# هایلایت الگوی فعلی
ax.axvspan(n - PATTERN_LEN, n, alpha=0.15, color='blue', label='الگوی فعلی')

# هایلایت الگوهای مشابه
for i in top_indices:
    idx = valid_indices[i]
    if idx >= plot_start:
        ax.axvspan(idx, idx + PATTERN_LEN, alpha=0.25, color='orange')

ax.set_title(f"{SYMBOL} - الگوهای مشابه (فاصله اقلیدسی)", fontsize=12)
ax.set_xlabel("Index")
ax.set_ylabel("Price")
ax.grid(alpha=0.3)

buf = io.BytesIO()
plt.tight_layout()
plt.savefig(buf, format='png', dpi=110)
plt.close()
buf.seek(0)

print("📤 ارسال نمودار به تلگرام...")
send_telegram_photo(buf, caption=f"📈 {SYMBOL} - الگوهای مشابه | وین ریت: {win_rate:.1f}%")

print("✅ تمام شد!")
