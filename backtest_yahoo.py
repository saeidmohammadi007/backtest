#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
بک‌تست مستقل (بدون نیاز به scanner_yahoo.py)
رویداد = کندل 4h شبیه کندل مرجع + سیگنال End Trend (ADX/DI)
بعد از هر رویداد ۱۰۰ کندل بعدی: هدف قبل از استاپ؟  (استاپ 2/3/4/5٪ × هدف 10/20/40/80/160٪)
"""
import os
import time
import numpy as np
import pandas as pd
import requests
import yfinance as yf

# --- الگوی مرجع ---
PATTERN_SYMBOL = 'ETH-USD'
PATTERN_DATE   = '2026-07-02'

# --- تنظیمات بک‌تست ---
COINS = ['BTC', 'ETH', 'BNB', 'XRP', 'SOL', 'ADA', 'DOGE', 'TRX', 'AVAX', 'LINK',
         'DOT', 'LTC', 'BCH', 'UNI', 'ATOM', 'XLM', 'ETC', 'NEAR', 'FIL', 'AAVE']
DIST_MAX    = 0.25   # آستانه شباهت کندل
EXIT_FILTER = -1     # -1: خروج از Short | 1: خروج از Long | 0: هر دو
HORIZON     = 100
MIN_GAP     = 6
STOPS       = [2, 3, 4, 5]
TARGETS     = [10, 20, 40, 80, 160]
HIST        = '730d'

# --- ADX ---
ADX_SMOOTHING = 14
DI_LENGTH     = 14
LEVEL_RANGE   = 20
LEVEL_TREND   = 35
TF_HOURS      = 4


def to_4h(df1h):
    df1h = df1h.dropna(subset=['open', 'high', 'low', 'close'])
    if df1h.empty:
        return None
    idx = pd.to_datetime(df1h.index)
    if idx.tz is not None:
        idx = idx.tz_convert('UTC').tz_localize(None)
    df1h = df1h.copy()
    df1h.index = idx
    r = df1h.resample(f'{TF_HOURS}h', origin='start_day').agg(
        {'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last'})
    return r.dropna()


def get_ref():
    df = yf.download(PATTERN_SYMBOL, start='2024-01-01', interval='1d',
                     progress=False, auto_adjust=False)
    if df is None or df.empty:
        return None
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df[['Open', 'High', 'Low', 'Close']].copy()
    df.index = pd.to_datetime(df.index)
    df.columns = ['open', 'high', 'low', 'close']
    return df


def vec(o, h, l, c):
    rng = float(h - l)
    if rng <= 0 or not np.isfinite(rng):
        return None
    return np.array([(o - l) / rng, 1.0, 0.0, (c - l) / rng])


def adx_di(df, n_di=DI_LENGTH, n_adx=ADX_SMOOTHING):
    H, L, C = df['high'], df['low'], df['close']
    up, dn = H.diff(), -L.diff()
    pdm = pd.Series(np.where((up > dn) & (up > 0), up, 0.0), index=df.index)
    mdm = pd.Series(np.where((dn > up) & (dn > 0), dn, 0.0), index=df.index)
    tr = pd.concat([H - L, (H - C.shift()).abs(), (L - C.shift()).abs()], axis=1).max(axis=1)

    def rma(s, n):
        return s.ewm(alpha=1 / n, adjust=False).mean()

    trr = rma(tr, n_di)
    pdi = 100 * rma(pdm, n_di) / trr
    mdi = 100 * rma(mdm, n_di) / trr
    dx = 100 * (pdi - mdi).abs() / (pdi + mdi).replace(0, np.nan)
    return rma(dx.fillna(0), n_adx), pdi, mdi


def exit_signals(adx, pdi, mdi, hr=LEVEL_RANGE, ht=LEVEL_TREND):
    a, p, m = adx.values, pdi.values, mdi.values
    n = len(a)
    cond = np.zeros(n)
    ex = np.zeros(n, dtype=int)
    for i in range(1, n):
        hp, hp_p = a[i] <= hr, a[i - 1] <= hr
        up_, up_p = p[i] >= m[i], p[i - 1] >= m[i - 1]
        dn_, dn_p = m[i] > p[i], m[i - 1] > p[i - 1]
        cross = up_ != up_p
        rising = a[i] > a[i - 1]

        e_long = (not hp) and up_ and rising and (not up_p)
        e_short = (not hp) and dn_ and rising and (not dn_p)
        e_long_s = (not hp) and up_ and rising and (p[i] >= ht)
        e_shrt_s = (not hp) and dn_ and rising and (m[i] > ht)

        x_long = (cross and up_p) or (hp and not hp_p)
        x_short = (cross and dn_p) or (hp and not hp_p)

        prev = cond[i - 1]
        x_l_now = (prev > 0) and x_long
        x_s_now = (prev < 0) and x_short

        if x_l_now:
            ex[i] = 1
        elif x_s_now:
            ex[i] = -1

        if prev != 1 and e_long_s:
            cond[i] = 1
        elif prev != -1 and e_shrt_s:
            cond[i] = -1
        elif prev != 0.5 and e_long:
            cond[i] = 0.5
        elif prev != -0.5 and e_short:
            cond[i] = -0.5
        elif x_l_now or x_s_now:
            cond[i] = 0
        else:
            cond[i] = prev
    return ex


def tg(text):
    tok = os.environ.get('TELEGRAM_BOT_TOKEN')
    cid = os.environ.get('TELEGRAM_CHAT_ID')
    if not tok or not cid:
        print("❌ TELEGRAM env vars not set")
        return
    try:
        r = requests.post(f"https://api.telegram.org/bot{tok}/sendMessage",
                          data={'chat_id': cid, 'text': text[:4000], 'parse_mode': 'HTML'},
                          timeout=15)
        print("✅ ارسال شد" if r.status_code == 200 else f"⚠️ {r.text}")
    except Exception as e:
        print(f"❌ {e}")


def load_4h(base):
    df = yf.download(f"{base}-USD", period=HIST, interval='1h',
                     progress=False, auto_adjust=False)
    if df is None or df.empty:
        return None
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df[['Open', 'High', 'Low', 'Close']].copy()
    df.columns = ['open', 'high', 'low', 'close']
    d4 = to_4h(df)
    return d4 if d4 is not None and len(d4) > 200 else None


def first_hit(mask):
    idx = np.flatnonzero(mask)
    return idx[0] if len(idx) else 10**9


def main():
    ref = get_ref()
    if ref is None:
        raise SystemExit("❌ no ref")
    m = ref.index.date == pd.to_datetime(PATTERN_DATE).date()
    if not m.any():
        raise SystemExit(f"❌ کندل {PATTERN_DATE} موجود نیست")
    c = ref[m].iloc[0]
    pv = vec(c['open'], c['high'], c['low'], c['close'])
    if pv is None:
        raise SystemExit("❌ invalid candle")

    events = []
    for base in COINS:
        try:
            df = load_4h(base)
            if df is None:
                print(f"⚠️ {base}: no data")
                continue
            adx, pdi, mdi = adx_di(df)
            ex = exit_signals(adx, pdi, mdi)
            O, H, L, C = (df[k].values for k in ('open', 'high', 'low', 'close'))
            last_i = -10**9
            for i in range(30, len(df) - HORIZON - 1):
                if ex[i] == 0 or (EXIT_FILTER and ex[i] != EXIT_FILTER):
                    continue
                if i - last_i < MIN_GAP:
                    continue
                v = vec(O[i], H[i], L[i], C[i])
                if v is None or np.linalg.norm(pv - v) > DIST_MAX:
                    continue
                last_i = i
                entry = C[i]
                fh = H[i + 1:i + 1 + HORIZON]
                fl = L[i + 1:i + 1 + HORIZON]
                res = {}
                for s in STOPS:
                    si = first_hit(fl <= entry * (1 - s / 100))
                    for t in TARGETS:
                        ti = first_hit(fh >= entry * (1 + t / 100))
                        res[(s, t)] = ti < si
                events.append({'sym': base, 'ts': df.index[i], 'res': res,
                               'max_up': (fh.max() / entry - 1) * 100,
                               'max_dd': (fl.min() / entry - 1) * 100})
            print(f"✅ {base}: {sum(e['sym'] == base for e in events)} رویداد")
        except Exception as e:
            print(f"⚠️ {base}: {type(e).__name__}: {e}")
        time.sleep(1)

    n = len(events)
    print(f"\nتعداد کل رویدادها: {n}")
    if n == 0:
        tg("❌ بک‌تست: رویدادی پیدا نشد؛ DIST_MAX را بیشتر کنید.")
        return

    tab = pd.DataFrame(
        {f">={t}%": [100 * np.mean([e['res'][(s, t)] for e in events]) for s in STOPS]
         for t in TARGETS},
        index=[f"stop {s}%" for s in STOPS]).round(1)
    print(tab.to_string())

    per = []
    for base in COINS:
        ev = [e for e in events if e['sym'] == base]
        if ev:
            per.append(f"{base}: n={len(ev)}")
    print("\n" + " | ".join(per))

    tg(f"📊 <b>بک‌تست مشابه {PATTERN_SYMBOL} ({PATTERN_DATE})</b>\n"
       f"رویدادها: {n} | افق: {HORIZON} کندل 4h\n"
       f"درصد رسیدن هدف قبل از استاپ:\n<pre>{tab.to_string()}</pre>\n" + " | ".join(per))

    pd.DataFrame([{'sym': e['sym'], 'time': e['ts'], 'max_up%': e['max_up'],
                   'max_dd%': e['max_dd']} for e in events]).to_csv('events.csv', index=False)


if __name__ == '__main__':
    main()
