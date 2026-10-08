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

SYMBOLS = [
    "BTC-USD", "ETH-USD", "BNB-USD", "SOL-USD", "XRP-USD",
    "ADA-USD", "DOGE-USD", "AVAX-USD", "DOT-USD", "LINK-USD",
]

INTERVAL      = "1d"
PERIOD        = "5y"
PATTERN_LEN   = 5         # طول الگو
LOOKAHEAD     = 30        # بررسی N کندل بعد
RR_RATIO      = 2.0       # ریسک به ریوارد
K_NEIGHBORS   = 8         # چند همسایه نزدیک بررسی شود
MIN_MATCHES   = 4         # حداقل تطابق معتبر
MIN_WIN_RATE  = 55.0      # حداقل وین‌ریت
MIN_PF        = 1.3       # حداقل Profit Factor
GAP           = PATTERN_LEN + 10   # فاصله حداقلی بین الگوها (جلوگیری از overlap)
TOP_CHARTS    = 5         # چند نمودار ارسال شود


# ======================== توابع ========================
def make_feature_vector(i, length, o, h, l, c):
    vec = np.concatenate([o[i:i+length], h[i:i+length],
                          l[i:i+length], c[i:i+length]]).astype(float)
    std = np.std(vec)
    if std == 0:
        return np.zeros_like(vec)
    return (vec - np.mean(vec)) / std


def simulate_trade(entry_idx, o, h, l, c, n, lookahead, rr):
    if entry_idx + 1 >= n:
        return None
    entry_price = float(o[entry_idx + 1])
    stop_loss   = float(np.min(l[entry_idx:entry_idx + PATTERN_LEN]))
    risk        = entry_price - stop_loss
    if risk <= 0:
        return None
    take_profit = entry_price + rr * risk
    end_idx     = min(entry_idx + 1 + lookahead, n)
    if end_idx <= entry_idx + 1:
        return None
    for j in range(entry_idx + 1, end_idx):
        if l[j] <= stop_loss:
            return {'result': 'loss', 'entry': entry_price, 'exit': stop_loss,
                    'bars': j - entry_idx - 1, 'idx': entry_idx}
        if h[j] >= take_profit:
            return {'result': 'win', 'entry': entry_price, 'exit': take_profit,
                    'bars': j - entry_idx - 1, 'idx': entry_idx}
    exit_price = float(c[end_idx - 1])
    return {'result': 'win' if exit_price > entry_price else 'loss',
            'entry': entry_price, 'exit': exit_price,
            'bars': end_idx - 2 - entry_idx, 'idx': entry_idx}


def fetch(symbol):
    try:
        df = yf.download(symbol, interval=INTERVAL, period=PERIOD,
                         progress=False, auto_adjust=True, timeout=30)
        if df is None or df.empty:
            return None
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        return df[['Open', 'High', 'Low', 'Close']].dropna()
    except Exception as e:
        log(f"❌ fetch {symbol}: {e}")
        return None


def analyze_symbol(symbol):
    log(f"📥 دریافت {symbol} | {PERIOD}")
    df = fetch(symbol)
    if df is None:
        return None
    n = len(df)
    MIN_REQ = PATTERN_LEN + LOOKAHEAD + 100
    if n < MIN_REQ:
        log(f"⚠️ داده کافی نیست {symbol} ({n})")
        return None

    o = df['Open'].values.astype(float)
    h = df['High'].values.astype(float)
    l = df['Low'].values.astype(float)
    c = df['Close'].values.astype(float)

    max_start = n - PATTERN_LEN - LOOKAHEAD - 1
    if max_start <= GAP * 2:
        return None
    candidates = np.arange(0, max_start)
    log(f"🔍 {symbol}: {len(candidates)} الگو بررسی می‌شود")

    vecs = np.array([make_feature_vector(i, PATTERN_LEN, o, h, l, c)
                     for i in candidates])
    dists = pairwise_distances(vecs, metric='euclidean')

    best = None
    for ci in range(len(candidates)):
        i = candidates[ci]
        valid_mask = np.abs(candidates - i) >= GAP
        valid_idx = np.where(valid_mask)[0]
        if len(valid_idx) < MIN_MATCHES:
            continue

        local = dists[ci][valid_idx]
        order = np.argsort(local)[:K_NEIGHBORS]
        neighbors = candidates[valid_idx[order]]

        results = []
        for nb in neighbors:
            tr = simulate_trade(int(nb), o, h, l, c, n, LOOKAHEAD, RR_RATIO)
            if tr:
                results.append(tr)
        if len(results) < MIN_MATCHES:
            continue

        wins = sum(1 for r in results if r['result'] == 'win')
        total = len(results)
        win_rate = wins / total * 100
        if win_rate < MIN_WIN_RATE:
            continue

        win_pnl  = [r['exit'] - r['entry'] for r in results if r['result'] == 'win']
        loss_pnl = [r['entry'] - r['exit'] for r in results if r['result'] == 'loss']
        gross_win  = float(np.sum(win_pnl)) if win_pnl else 0.0
        gross_loss = float(np.sum(loss_pnl)) if loss_pnl else 0.0
        pf = gross_win / gross_loss if gross_loss > 0 else float('inf')
        if pf < MIN_PF:
            continue

        score = win_rate * min(pf, 5.0)
        if best is None or score > best['score']:
            best = {
                'symbol': symbol,
                'idx': int(i),
                'date': df.index[i],
                'matches': total,
                'win_rate': win_rate,
                'pf': pf,
                'avg_win': float(np.mean(win_pnl)) if win_pnl else 0.0,
                'avg_loss': float(np.mean(loss_pnl)) if loss_pnl else 0.0,
                'avg_dist': float(np.mean(local[order[:len(results)]])),
                'score': score,
                'df': df, 'n': n,
                'neighbors': [int(x) for x in neighbors],
                'opens': o, 'highs': h, 'lows': l, 'closes': c,
            }
    return best


def make_chart(best):
    df = best['df']; o = best['opens']; h = best['highs']
    l = best['lows']; c = best['closes']; n = best['n']

    fig, ax = plt.subplots(figsize=(12, 5))
    plot_start = max(0, n - 200)
    for i in range(plot_start, n):
        col = '#26a69a' if c[i] >= o[i] else '#ef5350'
        ax.plot([i, i], [l[i], h[i]], color=col, linewidth=0.7)
        ax.plot([i, i], [o[i], c[i]], color=col, linewidth=3.0)

    idx = best['idx']
    if idx >= plot_start:
        ax.axvspan(idx, idx + PATTERN_LEN - 0.5, alpha=0.35, color='#2196F3',
                   label=f"الگوی برتر {best['date'].strftime('%Y-%m-%d')}")
    ax.axvspan(n - PATTERN_LEN, n - 0.5, alpha=0.20, color='#FF9800',
               label='الگوی فعلی')

    title = (f"{best['symbol']} [Daily] | {best['date'].strftime('%Y-%m-%d')} | "
             f"Matches: {best['matches']} | WR: {best['win_rate']:.1f}% | "
             f"PF: {best['pf']:.2f}")
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.set_xlabel("Index"); ax.set_ylabel("Price (USD)")
    ax.grid(alpha=0.3); ax.legend(loc='upper left')

    buf = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format='png', dpi=100)
    plt.close()
    buf.seek(0)
    return buf


# ======================== اجرای اصلی ========================
log("=" * 60)
log(f"🚀 شروع تحلیل {len(SYMBOLS)} ارز")
log("=" * 60)

all_bests = []
for sym in SYMBOLS:
    try:
        b = analyze_symbol(sym)
        if b:
            all_bests.append(b)
            log(f"✅ {sym}: {b['date'].date()} | matches={b['matches']} | "
                f"WR={b['win_rate']:.1f}% | PF={b['pf']:.2f}")
        else:
            log(f"⚠️ {sym}: الگوی معتبری یافت نشد")
    except Exception as e:
        log(f"❌ analyze {sym}: {e}")
        traceback.print_exc()

all_bests.sort(key=lambda x: x['score'], reverse=True)

# ======================== پیام تلگرام ========================
now_str = datetime.now().strftime("%Y-%m-%d %H:%M")
message = f"""📊 <b>گزارش الگوهای تکراری در ۱۰ ارز</b>
🗓 <code>{now_str}</code>
🔎 طول الگو: <code>{PATTERN_LEN}</code> کندل | بررسی: <code>{LOOKAHEAD}</code> کندل بعد
⚖️ R:R = <code>1:{RR_RATIO}</code> | حداقل تطابق: <code>{MIN_MATCHES}</code>
📈 آستانه: WR ≥ <code>{MIN_WIN_RATE}%</code> | PF ≥ <code>{MIN_PF}</code>

"""

if not all_bests:
    message += "❌ هیچ الگوی معتبری با شرایط تعریف‌شده یافت نشد.\n"
else:
    message += f"🏆 <b>الگوهای برتر ({len(all_bests)} ارز):</b>\n"
    for rank, b in enumerate(all_bests, 1):
        message += (f"\n{rank}. <b>{b['symbol']}</b> | <code>{b['date'].strftime('%Y-%m-%d')}</code>\n"
                    f"   ├ تطابق‌ها: <code>{b['matches']}</code>\n"
                    f"   ├ وین‌ریت: <code>{b['win_rate']:.1f}%</code>\n"
                    f"   ├ PF: <code>{b['pf']:.2f}</code>\n"
                    f"   ├ میانگین سود/ضرر: <code>{b['avg_win']:.2f} / {b['avg_loss']:.2f}</code>\n"
                    f"   └ امتیاز: <code>{b['score']:.1f}</code>")

message += "\n\n⚠️ صرفاً تحلیلی است و توصیه معاملاتی نیست."

if TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID:
    try:
        r = requests.post(
            f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage",
            json={"chat_id": TELEGRAM_CHAT_ID, "text": message,
                  "parse_mode": "HTML"},
            timeout=20,
        )
        log(f"📤 telegram (message): {r.status_code}")
    except Exception as e:
        log(f"❌ send msg: {e}")

# ======================== ارسال نمودارها ========================
if TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID and all_bests:
    for b in all_bests[:TOP_CHARTS]:
        try:
            buf = make_chart(b)
            caption = (f"📈 <b>{b['symbol']}</b> | {b['date'].strftime('%Y-%m-%d')}\n"
                       f"تطابق: <b>{b['matches']}</b> | WR: <b>{b['win_rate']:.1f}%</b> | "
                       f"PF: <b>{b['pf']:.2f}</b>")
            r = requests.post(
                f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendPhoto",
                files={"photo": ("chart.png", buf, "image/png")},
                data={"chat_id": TELEGRAM_CHAT_ID, "caption": caption,
                      "parse_mode": "HTML"},
                timeout=30,
            )
            log(f"📤 chart {b['symbol']}: {r.status_code}")
        except Exception as e:
            log(f"⚠️ chart {b['symbol']}: {e}")

log("✅ تمام شد!")
