import os
import pandas as pd
import numpy as np
import ccxt
from tqdm import tqdm
import time

# ============ Config (env-overridable) ============
TOP_N                = int(os.environ.get('TOP_N', '20'))
FORWARD_CANDLES      = int(os.environ.get('FORWARD_CANDLES', '100'))
OHLCV_LIMIT          = int(os.environ.get('OHLCV_LIMIT', '3000'))
SIMILARITY_THRESHOLD = float(os.environ.get('SIMILARITY_THRESHOLD', '0.5'))

TIMEFRAME            = '4h'
LEVEL_RANGE          = 20
LEVEL_TREND          = 35
DI_LENGTH            = 14
ADX_SMOOTHING        = 14

# --- کندل مرجع ETH 2026-07-02 (روزانه) ---
REF_O, REF_H, REF_L, REF_C = 1609.01, 1720.34, 1595.94, 1698.17

DROP_THRESHOLDS = [2, 3, 4, 5]
RISE_THRESHOLDS = [5, 10, 20, 40, 80]

# ============ Candle Vector ============
def candle_vector(o, h, l, c):
    rng = float(h - l)
    if rng <= 0 or not np.isfinite(rng):
        return None
    return np.array([(o - l) / rng, 1.0, 0.0, (c - l) / rng])

REF_VEC = candle_vector(REF_O, REF_H, REF_L, REF_C)
print(f"📐 بردار کندل ETH 2026-07-02 (روزانه):")
print(f"   O={REF_VEC[0]:.4f} H={REF_VEC[1]:.4f} "
      f"L={REF_VEC[2]:.4f} C={REF_VEC[3]:.4f}")
print(f"⚙️ پارامترها: TOP_N={TOP_N} | FORWARD={FORWARD_CANDLES} | "
      f"LIMIT={OHLCV_LIMIT} | THRESHOLD={SIMILARITY_THRESHOLD}")

# ============ Exchange ============
def init_exchange():
    ex = ccxt.lbank({'options': {'defaultType': 'future'}, 'enableRateLimit': True})
    ex.load_markets()
    return ex

def get_top_symbols(ex, n=20):
    try:
        tickers = ex.fetch_tickers()
    except Exception:
        tickers = {}
    candidates = []
    for sym, m in ex.markets.items():
        if not m.get('swap'):
            continue
        base = (m.get('base') or '').upper()
        if not base or base.isdigit():
            continue
        vol = (tickers.get(sym, {}).get('quoteVolume') or 0)
        candidates.append((base, sym, vol))
    candidates.sort(key=lambda x: x[2], reverse=True)
    seen, result = set(), []
    for base, sym, _ in candidates:
        if base in seen:
            continue
        seen.add(base)
        result.append((base, sym))
        if len(result) >= n:
            break
    return result

def fetch_4h(ex, symbol, limit=OHLCV_LIMIT):
    try:
        raw = ex.fetch_ohlcv(symbol, TIMEFRAME, limit=limit)
        if not raw or len(raw) < 200:
            return None
        df = pd.DataFrame(raw, columns=['ts','open','high','low','close','volume'])
        df['ts'] = pd.to_datetime(df['ts'], unit='ms', utc=True)
        df.set_index('ts', inplace=True)
        df.index = df.index.tz_localize(None)
        now = pd.Timestamp.utcnow().tz_localize(None)
        if df.index[-1] + pd.Timedelta(hours=4) > now:
            df = df.iloc[:-1]
        return df[['open','high','low','close','volume']]
    except Exception:
        return None

# ============ Indicators ============
def adx_di(df, n_di=DI_LENGTH, n_adx=ADX_SMOOTHING):
    H, L, C = df['high'], df['low'], df['close']
    up, dn = H.diff(), -L.diff()
    pdm = pd.Series(np.where((up > dn) & (up > 0), up, 0.0), index=df.index)
    mdm = pd.Series(np.where((dn > up) & (dn > 0), dn, 0.0), index=df.index)
    tr = pd.concat([H - L, (H - C.shift()).abs(),
                    (L - C.shift()).abs()], axis=1).max(axis=1)
    def rma(s, n): return s.ewm(alpha=1/n, adjust=False).mean()
    trr = rma(tr, n_di)
    pdi = 100 * rma(pdm, n_di) / trr
    mdi = 100 * rma(mdm, n_di) / trr
    dx = 100 * (pdi - mdi).abs() / (pdi + mdi).replace(0, np.nan)
    return rma(dx.fillna(0), n_adx), pdi, mdi

def state_machine(adx, pdi, mdi, hr=LEVEL_RANGE, ht=LEVEL_TREND):
    a, p, m = adx.values, pdi.values, mdi.values
    n = len(a)
    cond = np.zeros(n)
    ex = np.zeros(n, dtype=int)
    for i in range(1, n):
        hp, hp_p = a[i] <= hr, a[i-1] <= hr
        up_, up_p = p[i] >= m[i], p[i-1] >= m[i-1]
        dn_, dn_p = m[i] >  p[i], m[i-1] >  p[i-1]
        cross = up_ != up_p
        rising = a[i] > a[i-1]
        e_long_s = (not hp) and up_ and rising and (p[i] >= ht)
        e_shrt_s = (not hp) and dn_ and rising and (m[i] >  ht)
        e_long   = (not hp) and up_ and rising and (not up_p)
        e_short  = (not hp) and dn_ and rising and (not dn_p)
        x_long  = (cross and up_p) or (hp and not hp_p)
        x_short = (cross and dn_p) or (hp and not hp_p)
        prev = cond[i-1]
        x_l_now = (prev > 0) and x_long
        x_s_now = (prev < 0) and x_short
        if x_l_now: ex[i] = 1
        elif x_s_now: ex[i] = -1
        if   prev !=  1   and e_long_s:  cond[i] =  1
        elif prev != -1   and e_shrt_s:  cond[i] = -1
        elif prev !=  0.5 and e_long:    cond[i] =  0.5
        elif prev != -0.5 and e_short:   cond[i] = -0.5
        elif x_l_now or x_s_now:         cond[i] =  0
        else:                            cond[i] = prev
    return (pd.Series(cond, index=adx.index),
            pd.Series(ex, index=adx.index))

def analyze_forward(df, idx, forward=FORWARD_CANDLES):
    entry = df['close'].iloc[idx]
    future = df.iloc[idx+1:idx+1+forward]
    if len(future) < forward:
        return None
    return {
        'entry': float(entry),
        'max_rise_pct': (future['high'].max() / entry - 1) * 100,
        'max_drop_pct': (future['low'].min()  / entry - 1) * 100,
    }

# ============ Main ============
def main():
    print("\n🔌 اتصال به LBank...")
    ex = init_exchange()
    print(f"✅ {len(ex.markets)} بازار")

    print(f"\n📊 انتخاب {TOP_N} ارز برتر...")
    top = get_top_symbols(ex, TOP_N)
    for i, (b, s) in enumerate(top, 1):
        print(f"  {i:2d}. {b}")

    records = []
    total_end_trend = 0
    total_similar   = 0

    for base, sym in tqdm(top, desc="اسکن"):
        try:
            df = fetch_4h(ex, sym)
            if df is None:
                continue
            adx, pdi, mdi = adx_di(df)
            if adx.isna().all():
                continue
            cond, exs = state_machine(adx, pdi, mdi)
            long_exits = np.where(exs.values == 1)[0]
            total_end_trend += len(long_exits)

            n_sim = 0
            for idx in long_exits:
                row = df.iloc[idx]
                v = candle_vector(row['open'], row['high'],
                                  row['low'], row['close'])
                if v is None:
                    continue
                dist = float(np.linalg.norm(REF_VEC - v))
                if dist > SIMILARITY_THRESHOLD:
                    continue

                res = analyze_forward(df, idx, FORWARD_CANDLES)
                if res is None:
                    continue

                records.append({
                    'symbol':       base,
                    'time':         df.index[idx],
                    'entry_price':  res['entry'],
                    'dist':         dist,
                    'max_rise_pct': res['max_rise_pct'],
                    'max_drop_pct': res['max_drop_pct'],
                })
                n_sim += 1
            total_similar += n_sim

            print(f"\n🔍 {base}: {len(long_exits)} End Trend | "
                  f"{n_sim} شبیه کندل مرجع")
            time.sleep(0.4)
        except Exception as e:
            print(f"\n❌ {base}: {e}")
            continue

    if not records:
        print("\n❌ هیچ سیگنالی با شرایط «End Trend + کندل شبیه» یافت نشد")
        pd.DataFrame().to_csv('end_trend_similar_backtest.csv', index=False)
        return

    df_r = pd.DataFrame(records)
    total = len(df_r)

    print("\n" + "=" * 72)
    print("📊 بک‌تست — End Trend + کندل شبیه ETH 2026-07-02")
    print("=" * 72)
    print(f"کل End Trend یافت‌شده:           {total_end_trend}")
    print(f"سیگنال‌های شبیه کندل مرجع:        {total_similar}  "
          f"({total_similar/max(total_end_trend,1)*100:.1f}% کل)")
    print(f"از این تعداد، تحلیل آینده کامل:   {total}")
    print(f"تعداد ارز:                        {df_r['symbol'].nunique()}")
    print(f"افق آینده‌نگری:                   {FORWARD_CANDLES} کندل "
          f"({FORWARD_CANDLES*4/24:.1f} روز)")
    print(f"بازه تاریخی:                      {df_r['time'].min()} → "
          f"{df_r['time'].max()}")
    print(f"آستانه شباهت:                     dist ≤ {SIMILARITY_THRESHOLD}")
    print()

    print("📉 آمار ریزش:")
    for t in DROP_THRESHOLDS:
        hit = int((df_r['max_drop_pct'] <= -t).sum())
        pct = hit / total * 100
        bar = "█" * int(pct / 2)
        print(f"   -{t:2d}%:  {hit:4d}/{total}  ({pct:5.1f}%)  {bar}")

    print("\n📈 آمار رشد:")
    for t in RISE_THRESHOLDS:
        hit = int((df_r['max_rise_pct'] >= t).sum())
        pct = hit / total * 100
        bar = "█" * int(pct / 2)
        print(f"   +{t:2d}%:  {hit:4d}/{total}  ({pct:5.1f}%)  {bar}")

    print("\n📊 توزیع آماری:")
    print(f"   میانگین حداکثر رشد:   {df_r['max_rise_pct'].mean():+7.2f}%")
    print(f"   میانه  حداکثر رشد:    {df_r['max_rise_pct'].median():+7.2f}%")
    print(f"   میانگین حداکثر ریزش:  {df_r['max_drop_pct'].mean():+7.2f}%")
    print(f"   میانه  حداکثر ریزش:   {df_r['max_drop_pct'].median():+7.2f}%")

    print("\n📋 تعداد سیگنال به تفکیک ارز:")
    per_sym = df_r.groupby('symbol').size().sort_values(ascending=False)
    for s, cnt in per_sym.items():
        print(f"   {s:10s}: {cnt:3d}")

    print("\n🎯 ۱۰ سیگنال شبیه‌تر:")
    cols = ['symbol', 'time', 'dist', 'max_rise_pct', 'max_drop_pct']
    print(df_r.nsmallest(10, 'dist')[cols].to_string(index=False))

    df_r.to_csv('end_trend_similar_backtest.csv', index=False)
    print(f"\n💾 ذخیره شد: end_trend_similar_backtest.csv ({total} ردیف)")

if __name__ == "__main__":
    main()
