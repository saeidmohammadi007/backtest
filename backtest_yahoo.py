#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
بک‌تست: همان شرایط scanner_yahoo.py (شباهت کندل + End Trend) روی ۲۰ ارز، در کل تاریخچه‌ی موجود
بعد از هر رویداد، ۱۰۰ کندل 4h بعدی بررسی می‌شود:
  ریزش (استاپ) 2/3/4/5٪  ×  هدف 10/20/40/80/160٪
  موفق = هدف قبل از استاپ لمس شود (اگر در یک کندل هر دو شد، استاپ فرض می‌شود)
اجرا کنار scanner_yahoo.py:   python3 backtest_yahoo.py
"""
import time
import numpy as np
import pandas as pd
import yfinance as yf

import scanner_yahoo as S      # adx_di, state_machine, vec, to_4h, get_ref, PATTERN_*

COINS = ['BTC', 'ETH', 'BNB', 'XRP', 'SOL', 'ADA', 'DOGE', 'TRX', 'AVAX', 'LINK',
         'DOT', 'LTC', 'BCH', 'UNI', 'ATOM', 'XLM', 'ETC', 'NEAR', 'FIL', 'AAVE']

DIST_MAX    = 0.25          # آستانه شباهت کندل (کمتر = سخت‌گیرانه‌تر)
EXIT_FILTER = -1            # -1: خروج از Short (ورود Long) | 1: خروج از Long | 0: هر دو (فقط Long بررسی می‌شود)
HORIZON     = 100           # تعداد کندل بعد از رویداد
MIN_GAP     = 6             # حداقل فاصله بین دو رویداد یک ارز (کندل)
STOPS       = [2, 3, 4, 5]
TARGETS     = [10, 20, 40, 80, 160]
HIST        = '730d'        # سقف داده 1h در یاهو


def load_4h(base):
    df = yf.download(f"{base}-USD", period=HIST, interval='1h',
                     progress=False, auto_adjust=False)
    if df is None or df.empty:
        return None
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df[['Open', 'High', 'Low', 'Close']].copy()
    df.columns = ['open', 'high', 'low', 'close']
    d4 = S.to_4h(df)
    return d4 if d4 is not None and len(d4) > 200 else None


def first_hit(mask):
    idx = np.flatnonzero(mask)
    return idx[0] if len(idx) else 10**9


def main():
    ref = S.get_ref()
    m = ref.index.date == pd.to_datetime(S.PATTERN_DATE).date()
    c = ref[m].iloc[0]
    pv = S.vec(c['open'], c['high'], c['low'], c['close'])

    events = []
    for base in COINS:
        try:
            df = load_4h(base)
            if df is None:
                print(f"⚠️ {base}: no data"); continue
            adx, pdi, mdi = S.adx_di(df)
            _, exs = S.state_machine(adx, pdi, mdi)
            ex = exs.values
            H, L, C = df['high'].values, df['low'].values, df['close'].values
            O = df['open'].values
            last_i = -10**9
            for i in range(30, len(df) - HORIZON - 1):
                if ex[i] == 0 or (EXIT_FILTER and ex[i] != EXIT_FILTER):
                    continue
                if i - last_i < MIN_GAP:
                    continue
                v = S.vec(O[i], H[i], L[i], C[i])
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
                        res[(s, t)] = ti < si          # هدف قبل از استاپ
                events.append({'sym': base, 'ts': df.index[i], 'res': res,
                               'max_up': (fh.max() / entry - 1) * 100,
                               'max_dd': (fl.min() / entry - 1) * 100})
            print(f"✅ {base}: {sum(e['sym'] == base for e in events)} رویداد")
        except Exception as e:
            print(f"⚠️ {base}: {type(e).__name__}: {e}")
        time.sleep(0.5)

    n = len(events)
    print(f"\nتعداد کل رویدادها: {n}")
    if n == 0:
        print("رویدادی پیدا نشد؛ DIST_MAX را بیشتر کنید.")
        S.tg("❌ بک‌تست: رویدادی پیدا نشد؛ DIST_MAX را بیشتر کنید.")
        return

    tab = pd.DataFrame(
        {f"≥{t}%": [100 * np.mean([e['res'][(s, t)] for e in events]) for s in STOPS]
         for t in TARGETS},
        index=[f"استاپ {s}%" for s in STOPS]).round(1)
    print("\nدرصد موفقیت (هدف قبل از استاپ، در ۱۰۰ کندل بعد):")
    print(tab.to_string())
    S.tg(f"📊 <b>بک‌تست مشابه {S.PATTERN_SYMBOL} ({S.PATTERN_DATE})</b>\n"
         f"رویدادها: {n} | افق: {HORIZON} کندل 4h\n"
         f"درصد رسیدن هدف قبل از استاپ:\n<pre>{tab.to_string()}</pre>")

    print("\nبه تفکیک ارز (تعداد رویداد | موفقیت هدف 10٪ با استاپ 5٪):")
    for base in COINS:
        ev = [e for e in events if e['sym'] == base]
        if ev:
            print(f"  {base:5s} n={len(ev):3d}  {100 * np.mean([e['res'][(5, 10)] for e in ev]):5.1f}%")

    pd.DataFrame([{'sym': e['sym'], 'time': e['ts'], 'max_up%': e['max_up'],
                   'max_dd%': e['max_dd']} for e in events]).to_csv('events.csv', index=False)
    print("\n💾 events.csv ذخیره شد")


if __name__ == '__main__':
    main()
