# ================== ایمپورت‌ها ==================
import matplotlib
matplotlib.use('Agg')

import ccxt, time, warnings, signal, os, requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta, timezone

warnings.filterwarnings('ignore')

# ================== پوشه‌ی خروجی ==================
OUTPUT_DIR = 'outputs'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ================== تنظیمات تلگرام ==================
TELEGRAM_BOT_TOKEN = os.environ.get('TELEGRAM_BOT_TOKEN', '').strip()
TELEGRAM_CHAT_ID   = os.environ.get('TELEGRAM_CHAT_ID', '').strip()
TELEGRAM_ENABLED   = bool(TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID)


def _split_text(text, max_len):
    """تقسیم متن طولانی به تکه‌های مجاز (حداکثر 4096 کاراکتر)"""
    lines = text.split('\n')
    chunks, cur = [], ''
    for line in lines:
        if len(cur) + len(line) + 1 > max_len:
            chunks.append(cur)
            cur = line
        else:
            cur = cur + '\n' + line if cur else line
    if cur:
        chunks.append(cur)
    return chunks


def tg_send_message(text, parse_mode='Markdown'):
    """ارسال پیام متنی به تلگرام (بدون عکس و فایل)"""
    if not TELEGRAM_ENABLED:
        print("ℹ️ تلگرام غیرفعال (توکن یا chat_id ست نشده)")
        return False
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    try:
        for chunk in _split_text(text, 4000):
            r = requests.post(url, data={
                'chat_id': TELEGRAM_CHAT_ID,
                'text': chunk,
                'parse_mode': parse_mode,
                'disable_web_page_preview': True,
            }, timeout=20)
            if r.status_code != 200:
                print(f"❌ تلگرام: {r.status_code} - {r.text[:200]}")
                # تلاش دوباره بدون parse_mode (اگه مشکل کاراکتر خاص بود)
                r2 = requests.post(url, data={
                    'chat_id': TELEGRAM_CHAT_ID,
                    'text': chunk,
                }, timeout=20)
                if r2.status_code != 200:
                    return False
        return True
    except Exception as e:
        print(f"❌ خطا در ارسال پیام: {e}")
        return False


# ================== تنظیمات بک‌تست ==================
SYMBOLS = ['BTC/USDT','ETH/USDT','BNB/USDT','SOL/USDT','XRP/USDT',
           'DOGE/USDT','ADA/USDT','AVAX/USDT','LTC/USDT','UNI/USDT']

TIMEFRAME = '4h'
LOOKBACK_DAYS = 730
PATTERN_SYMBOL = 'ETH/USDT'
PATTERN_DATE = '2025-07-04'
SIMILARITY_THRESHOLD = 0.08
FORWARD_CANDLES = 60
USE_MACD_FILTER = False

PER_SYMBOL_TIMEOUT = 45
MAX_FETCH_RETRIES = 2
EXCHANGE_NAME = 'lbank'


# ================== تایم‌اوت ==================
class TimeoutError_(Exception): pass


def _timeout_handler(signum, frame): raise TimeoutError_()


try:
    signal.signal(signal.SIGALRM, _timeout_handler)
    HAS_ALARM = True
except Exception:
    HAS_ALARM = False


# ================== توابع ==================
def connect_exchange():
    try:
        ex = getattr(ccxt, EXCHANGE_NAME)({'enableRateLimit': True, 'timeout': 20000})
        ex.load_markets()
        print(f"✅ اتصال به {EXCHANGE_NAME.upper()}")
        return ex, EXCHANGE_NAME
    except Exception as e:
        raise SystemExit(f"❌ اتصال به {EXCHANGE_NAME.upper()} ناموفق: {e}")


def fetch_ohlcv(ex, sym, tf='4h', days=730):
    since = ex.parse8601((datetime.now(timezone.utc) - timedelta(days=days)).isoformat())
    data = []
    retries = 0
    while True:
        try:
            ohlcv = ex.fetch_ohlcv(sym, tf, since, limit=1000)
            retries = 0
        except Exception:
            retries += 1
            if retries > MAX_FETCH_RETRIES:
                return None
            time.sleep(1); continue
        if not ohlcv: break
        data.extend(ohlcv)
        since = ohlcv[-1][0] + 1
        if len(ohlcv) < 1000: break
        time.sleep(0.15)
    if not data: return None
    df = pd.DataFrame(data, columns=['ts','open','high','low','close','volume'])
    df['dt'] = pd.to_datetime(df['ts'], unit='ms')
    df.set_index('dt', inplace=True)
    df = df[['open','high','low','close','volume']].astype(float)
    return df[~df.index.duplicated()].sort_index()


def candle_vector(o, h, l, c):
    r = float(h - l)
    if r <= 0 or not np.isfinite(r): return None
    return np.array([(o-l)/r, 1.0, 0.0, (c-l)/r])


def scan_matches(df, pvec, threshold):
    matches = []
    for i in range(len(df)):
        row = df.iloc[i]
        v = candle_vector(row['open'], row['high'], row['low'], row['close'])
        if v is None: continue
        d = float(np.linalg.norm(pvec - v))
        if d <= threshold:
            matches.append({'index': i, 'dt': df.index[i], 'dist': d})
    return matches


def measure_forward(df, matches, forward):
    rows = []
    for m in matches:
        i = m['index']
        if i + forward >= len(df):
            continue
        entry = df['open'].iloc[i + 1]
        f = df.iloc[i + 1 : i + forward + 1]
        rows.append({
            'match_dt': m['dt'],
            'entry_dt': df.index[i + 1],
            'dist': round(m['dist'], 4),
            'entry': entry,
            'end_%': round((f['close'].iloc[-1] - entry) / entry * 100, 2),
            'max_%': round((f['high'].max() - entry) / entry * 100, 2),
            'min_%': round((f['low'].min() - entry) / entry * 100, 2),
        })
    return pd.DataFrame(rows)


# ================== شروع ==================
tg_send_message(
    f"🚀 *شروع بک‌تست*\n"
    f"📅 الگو: `{PATTERN_DATE}`\n"
    f"🪙 نمادها: {len(SYMBOLS)}\n"
    f"⏱️ تایم‌فریم: `{TIMEFRAME}`"
)

ex, ex_name = connect_exchange()

target = pd.to_datetime(PATTERN_DATE).date()
print(f"\n🔍 دریافت داده مرجع {PATTERN_SYMBOL} (روزانه) از {ex_name.upper()} ...")

ref = fetch_ohlcv(ex, PATTERN_SYMBOL, '1d', days=1500)

if ref is None or len(ref) == 0:
    msg = "❌ داده مرجع دریافت نشد."
    print(msg); tg_send_message(msg)
    raise SystemExit

oldest = ref.index[0].date()
newest = ref.index[-1].date()
print(f"📅 بازه‌ی داده: از {oldest} تا {newest}  ({len(ref)} کندل)")

if oldest > target:
    msg = f"❌ LBank فقط تا {oldest} عقب می‌رود."
    print(msg); tg_send_message(msg)
    raise SystemExit

candidates = [r for d, r in ref.iterrows() if d.date() <= target]
ref_row = candidates[-1]
actual_date = ref_row.name.date()

if actual_date != target:
    print(f"⚠️ نزدیک‌ترین کندل به {PATTERN_DATE}: {actual_date}")

row = ref_row
pvec = candle_vector(row['open'], row['high'], row['low'], row['close'])
print(f"\n📌 الگو ({actual_date}):")
print(f"   O={row['open']:.4f}  H={row['high']:.4f}  L={row['low']:.4f}  C={row['close']:.4f}")
print(f"📐 بردار: {pvec.round(3)}\n")

all_trades = {}
skipped = []

print(f"📊 شروع بک‌تست {len(SYMBOLS)} نماد\n")

for idx, sym in enumerate(SYMBOLS, 1):
    print(f"[{idx:02d}/{len(SYMBOLS)}] {sym} ...", end=' ', flush=True)
    t0 = time.time()
    try:
        if HAS_ALARM:
            signal.alarm(PER_SYMBOL_TIMEOUT)
        df = fetch_ohlcv(ex, sym, TIMEFRAME, days=LOOKBACK_DAYS)
        if HAS_ALARM:
            signal.alarm(0)
    except TimeoutError_:
        if HAS_ALARM: signal.alarm(0)
        print("⏱️ timeout"); skipped.append(sym); continue
    except Exception as e:
        if HAS_ALARM: signal.alarm(0)
        print(f"❌ {e.__class__.__name__}"); skipped.append(sym); continue

    if df is None or len(df) < 60:
        print("⏭️ داده ناکافی"); skipped.append(sym); continue

    try:
        matches = scan_matches(df, pvec, SIMILARITY_THRESHOLD)
        trades = measure_forward(df, matches, FORWARD_CANDLES)
        if not trades.empty:
            trades['symbol'] = sym
            all_trades[sym] = trades
            print(f"✅ {len(trades)} تطابق ({time.time()-t0:.1f}s)")
        else:
            print(f"⚪ ۰ تطابق ({time.time()-t0:.1f}s)")
    except Exception as e:
        print(f"❌ {e.__class__.__name__}")
    time.sleep(0.2)


# ================== گزارش ==================
if skipped:
    print(f"\n⏭️ رد شده: {', '.join(skipped)}")

if not all_trades:
    msg = "❌ هیچ تطابقی پیدا نشد."
    print(f"\n{msg}"); tg_send_message(msg)
    raise SystemExit

full = pd.concat(all_trades.values(), ignore_index=True)
full = full.sort_values('entry_dt').reset_index(drop=True)
total = len(full)

wins_df = full[full['end_%'] > 0]['end_%']
loss_df = full[full['end_%'] < 0]['end_%']

win_trades = full[full['end_%'] > 0].copy()
win_trades['drawdown_%'] = -win_trades['min_%']

print("\n" + "="*70)
print("📈 خلاصه‌ی کلی")
print("="*70)
print(f"  تعداد کل معاملات: {total}")
print(f"  نرخ برد:          {len(wins_df)/total*100:.1f}%")
print(f"  میانگین بازده:    {full['end_%'].mean():+.2f}%")
print(f"  میانگین MFE:      {full['max_%'].mean():+.2f}%")
print(f"  میانگین MAE:      {full['min_%'].mean():+.2f}%")


# ================== SL پیشنهادی ==================
if len(win_trades) > 0:
    sl_80 = win_trades['drawdown_%'].quantile(0.80)
    sl_90 = win_trades['drawdown_%'].quantile(0.90)
    sl_95 = win_trades['drawdown_%'].quantile(0.95)
else:
    sl_80 = sl_90 = sl_95 = 5.0


def simulate_tp_sl(df, tp_pct, sl_pct):
    results = []
    for _, r in df.iterrows():
        hit_tp = r['max_%'] >= tp_pct
        hit_sl = r['min_%'] <= -sl_pct
        if hit_tp and hit_sl:
            outcome = -sl_pct
        elif hit_tp:
            outcome = tp_pct
        elif hit_sl:
            outcome = -sl_pct
        else:
            outcome = r['end_%']
        results.append(outcome)

    results = np.array(results)
    wins_arr = results[results > 0]
    losses_arr = results[results < 0]

    n_wins = len(wins_arr); n_loss = len(losses_arr)
    win_rate = n_wins / len(results) * 100 if len(results) else 0
    avg_win = wins_arr.mean() if n_wins > 0 else 0
    avg_loss = abs(losses_arr.mean()) if n_loss > 0 else 0
    expectancy = results.mean()
    rr = avg_win / avg_loss if avg_loss > 0 else 0
    cumulative = np.prod(1 + results / 100)

    return {
        'tp': tp_pct, 'sl': sl_pct,
        'hit_tp_pct': (df['max_%'] >= tp_pct).mean() * 100,
        'hit_sl_pct': (df['min_%'] <= -sl_pct).mean() * 100,
        'win_rate': win_rate,
        'avg_win': avg_win, 'avg_loss': avg_loss,
        'rr': rr, 'expectancy': expectancy,
        'final_capital': cumulative * 100,
    }


SL_FIXED = round(sl_90, 2)
tp_levels = [1, 1.5, 2, 2.5, 3, 4, 5, 6, 8, 10, 12, 15, 20, 25, 30]
sl_options = [2, 3, 4, 5, 6, 8, 10, 12, 15]

tp_results = [simulate_tp_sl(full, tp, SL_FIXED) for tp in tp_levels]
best_tp = max(tp_results, key=lambda x: x['final_capital'])

matrix_vals = {}
for tp in tp_levels:
    for sl in sl_options:
        matrix_vals[(tp, sl)] = simulate_tp_sl(full, tp, sl)

all_combos = list(matrix_vals.values())
best_combo = max(all_combos, key=lambda x: x['expectancy'])
best_capital = max(all_combos, key=lambda x: x['final_capital'])


# ================== 📊 ذخیره‌ی CSV (برای Artifact گیت‌هاب) ==================
full.to_csv(os.path.join(OUTPUT_DIR, f'backtest_results_{PATTERN_DATE}.csv'), index=False)
pd.DataFrame(tp_results).to_csv(os.path.join(OUTPUT_DIR, f'tp_analysis_{PATTERN_DATE}.csv'), index=False)
pd.DataFrame([{**{'tp': tp, 'sl': sl}, **matrix_vals[(tp, sl)]}
              for tp in tp_levels for sl in sl_options]).to_csv(
    os.path.join(OUTPUT_DIR, f'tp_sl_matrix_{PATTERN_DATE}.csv'), index=False)


# ================== 📱 ارسال فقط متن به تلگرام ==================
print("\n📱 ارسال به تلگرام ...")

profitable = best_combo['win_rate'] > 100/(1+best_combo['rr'])
status_emoji = "✅" if profitable else "⚠️"

# ---- پیام ۱: خلاصه ----
msg1 = (
    f"📊 *نتیجه‌ی بک‌تست*\n"
    f"━━━━━━━━━━━━━━━━━━━━\n"
    f"📅 الگو: `{actual_date}`\n"
    f"🪙 نمادها: {len(SYMBOLS)} | رد شده: {len(skipped)}\n"
    f"━━━━━━━━━━━━━━━━━━━━\n"
    f"*📈 خلاصه‌ی کلی:*\n"
    f"• تعداد معاملات: `{total}`\n"
    f"• نرخ برد: `{len(wins_df)/total*100:.1f}%`\n"
    f"• میانگین بازده: `{full['end_%'].mean():+.2f}%`\n"
    f"• میانگین MFE: `{full['max_%'].mean():+.2f}%`\n"
    f"• میانگین MAE: `{full['min_%'].mean():+.2f}%`"
)
tg_send_message(msg1)

# ---- پیام ۲: پیشنهاد نهایی ----
msg2 = (
    f"🎯 *پیشنهاد حد سود و حد ضرر*\n"
    f"━━━━━━━━━━━━━━━━━━━━\n"
    f"• حد سود (TP): `+{best_combo['tp']:.1f}%`\n"
    f"• حد ضرر (SL): `-{best_combo['sl']:.1f}%`\n"
    f"• نسبت R:R: `{best_combo['rr']:.2f} : 1`\n"
    f"• نرخ برد مورد نیاز: `{100/(1+best_combo['rr']):.1f}%`\n"
    f"• نرخ برد واقعی: `{best_combo['win_rate']:.1f}%`\n"
    f"• امید ریاضی: `{best_combo['expectancy']:+.2f}%`\n"
    f"• سرمایه‌ی نهایی: `{best_combo['final_capital']:.2f}`\n"
    f"━━━━━━━━━━━━━━━━━━━━\n"
    f"{status_emoji} {'استراتژی سودآور است' if profitable else 'نیاز به بهینه‌سازی دارد'}"
)
tg_send_message(msg2)

# ---- پیام ۳: SL های پیشنهادی و بهترین TP ----
msg3 = (
    f"📌 *SL پیشنهادی (بر اساس MAE برنده‌ها)*\n"
    f"━━━━━━━━━━━━━━━━━━━━\n"
    f"• SL = `-{sl_80:.2f}%` → حفظ ۸۰٪ برنده‌ها\n"
    f"• SL = `-{sl_90:.2f}%` → حفظ ۹۰٪ برنده‌ها\n"
    f"• SL = `-{sl_95:.2f}%` → حفظ ۹۵٪ برنده‌ها\n"
    f"━━━━━━━━━━━━━━━━━━━━\n"
    f"🏆 بهترین TP (سرمایه): `+{best_tp['tp']:.1f}%` → `{best_tp['final_capital']:.2f}`\n"
    f"🏆 بهترین TP (امید):  `+{best_combo['tp']:.1f}%` | SL: `-{best_combo['sl']:.1f}%`"
)
tg_send_message(msg3)

print("\n✅ همه چیز تمام شد.")
