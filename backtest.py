# ================== ایمپورت‌ها ==================
import matplotlib
matplotlib.use('Agg')

import ccxt, time, warnings, os, requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone

warnings.filterwarnings('ignore')

OUTPUT_DIR = 'outputs'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ================== تلگرام ==================
TELEGRAM_BOT_TOKEN = os.environ.get('TELEGRAM_BOT_TOKEN', '').strip()
TELEGRAM_CHAT_ID   = os.environ.get('TELEGRAM_CHAT_ID', '').strip()
TELEGRAM_ENABLED   = bool(TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID)

TG_API = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}"


def _split_text(text, max_len=3800):
    lines = text.split('\n')
    chunks, cur = [], ''
    for line in lines:
        if len(cur) + len(line) + 1 > max_len:
            if cur: chunks.append(cur)
            cur = line
        else:
            cur = cur + '\n' + line if cur else line
    if cur: chunks.append(cur)
    return chunks


def tg_send_message(text, parse_mode='Markdown'):
    if not TELEGRAM_ENABLED:
        print("ℹ️ تلگرام غیرفعال")
        return False
    url = f"{TG_API}/sendMessage"
    for chunk in _split_text(text):
        try:
            r = requests.post(url, data={
                'chat_id': TELEGRAM_CHAT_ID,
                'text': chunk,
                'parse_mode': parse_mode,
                'disable_web_page_preview': True,
            }, timeout=20)
            if r.status_code != 200:
                # fallback بدون Markdown
                requests.post(url, data={
                    'chat_id': TELEGRAM_CHAT_ID,
                    'text': chunk,
                }, timeout=20)
            time.sleep(0.3)
        except Exception as e:
            print(f"❌ تلگرام: {e}")
    return True


# ================== تنظیمات ==================
SYMBOLS = ['BTC/USDT','ETH/USDT','BNB/USDT','SOL/USDT','XRP/USDT',
           'DOGE/USDT','ADA/USDT','AVAX/USDT','LTC/USDT','UNI/USDT']

TIMEFRAME             = '4h'
LOOKBACK_DAYS         = 730
PATTERN_SYMBOL        = 'ETH/USDT'
PATTERN_TIMEFRAME     = '1d'
PATTERN_LOOKBACK_DAYS = 1500
SIMILARITY_THRESHOLD  = 0.08
FORWARD_CANDLES       = 60
MIN_TRADES            = 30
CAPITAL_PER_TRADE_PCT = 2.0
EXCHANGE_NAME         = 'lbank'
TOP_N                 = 10
TP_LEVELS = [1, 1.5, 2, 2.5, 3, 4, 5, 6, 8, 10, 12, 15, 20, 25, 30]
SL_LEVELS = [2, 3, 4, 5, 6, 8, 10, 12, 15]
MAX_FETCH_RETRIES = 2


# ================== توابع ==================
def connect_exchange():
    ex = getattr(ccxt, EXCHANGE_NAME)({'enableRateLimit': True, 'timeout': 20000})
    ex.load_markets()
    print(f"✅ اتصال به {EXCHANGE_NAME.upper()}")
    return ex


def fetch_ohlcv(ex, sym, tf, days):
    since = ex.parse8601((datetime.now(timezone.utc) - timedelta(days=days)).isoformat())
    data, retries = [], 0
    while True:
        try:
            ohlcv = ex.fetch_ohlcv(sym, tf, since, limit=1000)
            retries = 0
        except Exception:
            retries += 1
            if retries > MAX_FETCH_RETRIES: return None
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
    return df[['open','high','low','close','volume']].astype(float).sort_index()


def build_vectors(df):
    o, h, l, c = df['open'].values, df['high'].values, df['low'].values, df['close'].values
    rng = h - l
    valid = rng > 0
    safe = np.where(valid, rng, 1.0)
    vecs = np.stack([(o-l)/safe, np.ones_like(o), np.zeros_like(o), (c-l)/safe], axis=1)
    return vecs, valid


def simulate(rets, tp, sl):
    capital = 100.0
    tr = np.empty(len(rets))
    pos_pct = CAPITAL_PER_TRADE_PCT / 100.0
    for k, (end_pct, max_pct, min_pct) in enumerate(rets):
        pos = capital * pos_pct
        hit_tp = max_pct >= tp
        hit_sl = min_pct <= -sl
        if hit_tp and hit_sl: pnl_pct = -sl
        elif hit_tp: pnl_pct = tp
        elif hit_sl: pnl_pct = -sl
        else: pnl_pct = end_pct
        prev = capital
        capital += pos * (pnl_pct / 100.0)
        tr[k] = (capital - prev) / prev * 100.0
    wins = tr[tr > 0]; losses = tr[tr < 0]
    wr = len(wins) / len(tr) * 100 if len(tr) else 0
    aw = wins.mean() if len(wins) else 0
    al = abs(losses.mean()) if len(losses) else 0
    rr = aw / al if al > 0 else 0
    return {'capital': capital, 'win_rate': wr, 'rr': rr,
            'avg_win': aw, 'avg_loss': al,
            'expectancy': tr.mean() if len(tr) else 0}


# ================== شروع ==================
tg_send_message(
    f"🔎 *اسکنر الگوهای تاریخی*\n"
    f"جستجوی بهترین ترکیب R:R + Win Rate\n"
    f"🪙 {len(SYMBOLS)} نماد | ⏱️ `{TIMEFRAME}` | افق `{FORWARD_CANDLES}` کندل\n"
    f"📊 حداقل معاملات: `{MIN_TRADES}`"
)

ex = connect_exchange()

# ---------- 1) داده‌ی نمادها ----------
print("\n📥 دریافت داده‌ی نمادها ...")
target_data = {}
for i, sym in enumerate(SYMBOLS, 1):
    print(f"  [{i:02d}/{len(SYMBOLS)}] {sym} ...", end=' ', flush=True)
    df = fetch_ohlcv(ex, sym, TIMEFRAME, days=LOOKBACK_DAYS)
    if df is not None and len(df) > FORWARD_CANDLES + 20:
        target_data[sym] = df
        print(f"✅ {len(df)}")
    else:
        print("⏭️")
    time.sleep(0.2)

if not target_data:
    tg_send_message("❌ داده‌ای دریافت نشد")
    raise SystemExit("no data")

# ---------- 2) بردارها ----------
print("\n⚙️ پیش‌پردازش ...")
all_vecs_list, all_sym_list, all_idx_list = [], [], []
sym_arrays = {}
for sym, df in target_data.items():
    vecs, valid = build_vectors(df)
    idxs = np.where(valid)[0]
    all_vecs_list.append(vecs[idxs])
    all_sym_list.extend([sym] * len(idxs))
    all_idx_list.extend(idxs.tolist())
    sym_arrays[sym] = (df['open'].values, df['high'].values,
                       df['low'].values, df['close'].values)

all_vecs = np.vstack(all_vecs_list)
all_sym  = np.array(all_sym_list)
all_idx  = np.array(all_idx_list)
print(f"  کل بردارها: {len(all_vecs):,}")

# ---------- 3) بازده آینده ----------
print("⏳ بازده‌های آینده ...")
forward_map = {}
for sym, (o, h, l, c) in sym_arrays.items():
    n = len(o)
    arr = np.full((n, 3), np.nan)
    for i in range(n - FORWARD_CANDLES):
        entry = o[i+1]
        if entry <= 0: continue
        fh = h[i+1:i+FORWARD_CANDLES+1].max()
        fl = l[i+1:i+FORWARD_CANDLES+1].min()
        fc = c[i+FORWARD_CANDLES]
        arr[i] = [(fc-entry)/entry*100, (fh-entry)/entry*100, (fl-entry)/entry*100]
    forward_map[sym] = arr

# ---------- 4) داده‌ی مرجع ----------
print(f"\n📥 {PATTERN_SYMBOL} ({PATTERN_TIMEFRAME}) ...")
ref = fetch_ohlcv(ex, PATTERN_SYMBOL, PATTERN_TIMEFRAME, days=PATTERN_LOOKBACK_DAYS)
if ref is None or len(ref) < 30:
    tg_send_message("❌ داده‌ی مرجع ناکافی")
    raise SystemExit("no ref")

ref_vecs, ref_valid = build_vectors(ref)
ref_dates = ref.index
print(f"  {ref_dates[0].date()} → {ref_dates[-1].date()} ({len(ref)})")

# ---------- 5) اسکن ----------
print(f"\n🔎 اسکن {int(ref_valid.sum())} کندل ...\n")
results = []
t_start = time.time()

for pos in range(len(ref_vecs)):
    if not ref_valid[pos]: continue
    vec = ref_vecs[pos]
    dt  = ref_dates[pos]

    dists = np.linalg.norm(all_vecs - vec, axis=1)
    mask = dists <= SIMILARITY_THRESHOLD
    if mask.sum() < MIN_TRADES: continue

    syms = all_sym[mask]; idxs = all_idx[mask]
    rets = []
    for s, i in zip(syms, idxs):
        fr = forward_map[s][i]
        if not np.isnan(fr[0]):
            rets.append(fr)
    if len(rets) < MIN_TRADES: continue
    rets = np.array(rets)

    best = None
    for tp in TP_LEVELS:
        for sl in SL_LEVELS:
            s = simulate(rets, tp, sl)
            if best is None or s['capital'] > best['capital']:
                s['tp'], s['sl'] = tp, sl
                best = s
    if best is None: continue

    ends = rets[:, 0]
    results.append({
        'date': dt.date(), 'n_trades': len(rets),
        'baseline_win_rate': (ends > 0).mean() * 100,
        'baseline_exp': ends.mean(),
        'tp': best['tp'], 'sl': best['sl'],
        'win_rate': best['win_rate'], 'rr': best['rr'],
        'expectancy': best['expectancy'],
        'final_capital': best['capital'],
        'avg_win': best['avg_win'], 'avg_loss': best['avg_loss'],
    })
    if len(results) % 100 == 0:
        print(f"  ... {len(results)} | {time.time()-t_start:.0f}s")

elapsed = time.time() - t_start
print(f"\n✅ {len(results)} الگو در {elapsed:.0f}s")

if not results:
    tg_send_message("❌ الگویی با شرایط کافی پیدا نشد")
    raise SystemExit("no results")

df_res = pd.DataFrame(results).sort_values('final_capital', ascending=False).reset_index(drop=True)
df_res['score_balanced'] = df_res['win_rate'] * df_res['rr']
df_res_balanced = df_res.sort_values('score_balanced', ascending=False).reset_index(drop=True)

# ذخیره‌ی محلی (اختیاری - برای خودت)
out_path = os.path.join(OUTPUT_DIR, 'best_patterns.csv')
df_res.to_csv(out_path, index=False)
print(f"💾 {out_path} (فقط ذخیره‌ی محلی)")

# ================== ارسال به تلگرام ==================
print("\n📱 ارسال به تلگرام ...")
top = df_res.head(TOP_N)

# ---- پیام ۱: خلاصه ----
msg1 = (
    f"🏆 *نتایج اسکن الگوها*\n"
    f"━━━━━━━━━━━━━━━━━━━━\n"
    f"📊 الگوهای اسکن‌شده: `{len(results)}`\n"
    f"🪙 نمادها: `{len(target_data)}`\n"
    f"⏱️ تایم‌فریم: `{TIMEFRAME}` | افق: `{FORWARD_CANDLES}`\n"
    f"📌 حداقل معاملات: `{MIN_TRADES}`\n"
    f"⏱️ زمان اسکن: `{elapsed:.0f}s`\n"
    f"━━━━━━━━━━━━━━━━━━━━\n"
    f"🎯 معیار: *سرمایه‌ی نهایی* (ترکیب WR و R:R)"
)
tg_send_message(msg1)

# ---- پیام‌های ۲-۳: جزئیات top 10 ----
for chunk_start in range(0, TOP_N, 5):
    chunk_text = f"📋 *الگوهای #{chunk_start+1} تا #{min(chunk_start+5, TOP_N)}*\n"
    chunk_text += "━━━━━━━━━━━━━━━━━━━━\n\n"
    for i in range(chunk_start, min(chunk_start+5, TOP_N)):
        r = top.iloc[i]
        chunk_text += (
            f"*#{i+1}*  📅 `{r['date']}`\n"
            f"  ▪️ معاملات: `{int(r['n_trades'])}`\n"
            f"  ▪️ TP/SL: `+{r['tp']}%` / `-{r['sl']}%`\n"
            f"  ▪️ Win Rate: `{r['win_rate']:.1f}%`\n"
            f"  ▪️ R:R: `{r['rr']:.2f} : 1`\n"
            f"  ▪️ Expectancy: `{r['expectancy']:+.2f}%`\n"
            f"  ▪️ سرمایه: `{r['final_capital']:.1f}`\n\n"
        )
    tg_send_message(chunk_text)

# ---- پیام ۴: متعادل‌ها ----
msg4 = "⚖️ *۵ الگوی متعادل (WR × R:R)*\n━━━━━━━━━━━━━━━━━━━━\n\n"
for i in range(min(5, len(df_res_balanced))):
    r = df_res_balanced.iloc[i]
    msg4 += (
        f"*#{i+1}*  📅 `{r['date']}`\n"
        f"  ▪️ WR: `{r['win_rate']:.1f}%`  |  R:R: `{r['rr']:.2f}`\n"
        f"  ▪️ TP/SL: `+{r['tp']}%` / `-{r['sl']}%`\n"
        f"  ▪️ امتیاز: `{r['score_balanced']:.1f}`\n\n"
    )
tg_send_message(msg4)

# ---- پیام ۵: پیشنهاد نهایی ----
best = df_res.iloc[0]
profitable = best['win_rate'] > 100 / (1 + best['rr']) if best['rr'] > 0 else False
status_emoji = "✅" if profitable else "⚠️"

msg5 = (
    f"💡 *پیشنهاد نهایی*\n"
    f"━━━━━━━━━━━━━━━━━━━━\n"
    f"🥇 الگو: `{best['date']}`\n"
    f"• TP: `+{best['tp']}%`\n"
    f"• SL: `-{best['sl']}%`\n"
    f"• Win Rate: `{best['win_rate']:.1f}%`\n"
    f"• R:R: `{best['rr']:.2f} : 1`\n"
    f"• نرخ برد موردنیاز: `{100/(1+best['rr']):.1f}%`\n"
    f"• Expectancy: `{best['expectancy']:+.2f}%`\n"
    f"• سرمایه نهایی: `{best['final_capital']:.1f}`\n"
    f"━━━━━━━━━━━━━━━━━━━━\n"
    f"{status_emoji} {'استراتژی سودآور' if profitable else 'نیاز به بهینه‌سازی'}\n"
    f"⚠️ مراقب Overfitting باش"
)
tg_send_message(msg5)

# ---- پیام پایانی ----
tg_send_message("✅ اسکن تمام شد.")

print("\n✅ تمام شد.")
