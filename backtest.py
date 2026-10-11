import ccxt
import pandas as pd
import numpy as np
from datetime import datetime
import time
import os
import json

# ============================================================
# 1. تنظیمات اولیه
# ============================================================
EXCHANGE = 'lbank'
TIMEFRAME = '4h'
SYMBOL_BTC = 'BTC/USDT'

# تاریخ و ساعت کندل هدف (UTC)
TARGET_DATE = datetime(2026, 8, 16, 20, 0)

TOP_N_SIMILAR = 20
DROP_THRESHOLD = -1.0
RISE_THRESHOLD_4 = 4.0
RISE_THRESHOLD_5 = 5.0
HORIZON_CANDLES = 6

TOP_10_SYMBOLS = [
    'BTC/USDT', 'ETH/USDT', 'BNB/USDT', 'XRP/USDT', 'SOL/USDT',
    'TRX/USDT', 'DOGE/USDT', 'ADA/USDT', 'LINK/USDT', 'DOT/USDT'
]

# ============================================================
# 2. اتصال به LBank
# ============================================================
exchange = ccxt.lbank({
    'enableRateLimit': True,
    'timeout': 30000,
})

print(f"Exchange: {exchange.name}")
print(f"fetchOHLCV supported: {exchange.has.get('fetchOHLCV')}")


# ============================================================
# 3. دریافت داده‌های تاریخی
# ============================================================
def fetch_ohlcv(exchange, symbol, timeframe, since=None, limit=1000):
    all_ohlcv = []
    max_retries = 5
    retry_delay = 3

    while True:
        retries = 0
        success = False
        ohlcv = []
        while retries < max_retries and not success:
            try:
                ohlcv = exchange.fetch_ohlcv(
                    symbol, timeframe, since=since, limit=min(limit, 1000)
                )
                success = True
            except ccxt.RateLimitExceeded:
                print(f"  Rate limit, retrying in {retry_delay}s...")
                time.sleep(retry_delay)
                retries += 1
            except (ccxt.NetworkError, ccxt.ExchangeError) as e:
                print(f"  Error: {e}, retrying...")
                time.sleep(retry_delay)
                retries += 1

        if not success or not ohlcv:
            break

        all_ohlcv.extend(ohlcv)

        if len(ohlcv) < min(limit, 1000):
            break

        since = ohlcv[-1][0] + 1
        time.sleep(0.15)

    if not all_ohlcv:
        return pd.DataFrame()

    df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    df = df[~df.index.duplicated(keep='first')]
    df.sort_index(inplace=True)
    return df


# ============================================================
# 4. دریافت داده بیت‌کوین
# ============================================================
print("\nFetching BTC data from LBank...")
btc_df = fetch_ohlcv(
    exchange, SYMBOL_BTC, TIMEFRAME,
    since=exchange.parse8601('2020-01-01T00:00:00Z')
)

if btc_df.empty:
    raise Exception("No BTC data received from LBank.")

print(f"BTC candles: {len(btc_df)} | Range: {btc_df.index[0]} -> {btc_df.index[-1]}")

# ============================================================
# 5. کندل هدف
# ============================================================
target_candle = btc_df.loc[:TARGET_DATE].iloc[-1]
print(f"\nTarget candle: {target_candle.name}")
print(f"O:{target_candle['open']:.2f} H:{target_candle['high']:.2f} "
      f"L:{target_candle['low']:.2f} C:{target_candle['close']:.2f}")


# ============================================================
# 6. ویژگی‌های کندل و شباهت
# ============================================================
def candle_features(row):
    body = row['close'] - row['open']
    upper_wick = row['high'] - max(row['open'], row['close'])
    lower_wick = min(row['open'], row['close']) - row['low']
    total_range = row['high'] - row['low']
    if total_range == 0:
        total_range = 1e-9
    return np.array([
        body / total_range,
        upper_wick / total_range,
        lower_wick / total_range,
        (row['close'] - row['open']) / row['open'],
    ])


target_features = candle_features(target_candle)

distances = []
for idx, row in btc_df.iterrows():
    if idx == target_candle.name:
        continue
    features = candle_features(row)
    dist = np.linalg.norm(target_features - features)
    distances.append((idx, dist))

distances.sort(key=lambda x: x[1])
similar_candles = distances[:TOP_N_SIMILAR]

print(f"\nTop {TOP_N_SIMILAR} similar candles found.")


# ============================================================
# 7. تحلیل رفتار پس از کندل
# ============================================================
def analyze_future_movement(df, candle_idx, horizon_candles=6):
    loc = df.index.get_loc(candle_idx)
    if loc + horizon_candles >= len(df):
        return None

    future = df.iloc[loc + 1: loc + 1 + horizon_candles]
    if len(future) == 0:
        return None

    entry_price = df.iloc[loc]['close']
    min_price = future['low'].min()
    max_price = future['high'].max()

    return {
        'entry_price': entry_price,
        'min_return': (min_price - entry_price) / entry_price * 100,
        'max_return': (max_price - entry_price) / entry_price * 100,
    }


# بررسی نمادهای موجود
exchange.load_markets()
available_symbols = [s for s in TOP_10_SYMBOLS if s in exchange.markets]
missing = [s for s in TOP_10_SYMBOLS if s not in exchange.markets]
if missing:
    print(f"Warning: symbols not on LBank: {missing}")

results = {}

for symbol in available_symbols:
    print(f"\nAnalyzing {symbol}...")

    try:
        df = fetch_ohlcv(
            exchange, symbol, TIMEFRAME,
            since=exchange.parse8601('2020-01-01T00:00:00Z')
        )
    except Exception as e:
        print(f"  Error fetching {symbol}: {e}")
        continue

    if df.empty:
        print(f"  No data for {symbol}")
        continue

    outcomes = []
    for candle_idx, _ in similar_candles:
        try:
            closest_idx = df.index.get_indexer([candle_idx], method='nearest')[0]
            if closest_idx == -1:
                continue
            actual_idx = df.index[closest_idx]
            result = analyze_future_movement(df, actual_idx, HORIZON_CANDLES)
            if result:
                outcomes.append(result)
        except Exception:
            continue

    if not outcomes:
        print(f"  Not enough data for {symbol}")
        continue

    total = len(outcomes)
    cond_4 = sum(1 for o in outcomes
                 if o['min_return'] > DROP_THRESHOLD and o['max_return'] > RISE_THRESHOLD_4)
    cond_5 = sum(1 for o in outcomes
                 if o['min_return'] > DROP_THRESHOLD and o['max_return'] > RISE_THRESHOLD_5)

    results[symbol] = {
        'total_similar': total,
        'condition_4_pct': (cond_4 / total) * 100,
        'condition_5_pct': (cond_5 / total) * 100,
        'avg_min_return': float(np.mean([o['min_return'] for o in outcomes])),
        'avg_max_return': float(np.mean([o['max_return'] for o in outcomes])),
    }

    print(f"  Samples: {total} | >4%: {results[symbol]['condition_4_pct']:.1f}% "
          f"| >5%: {results[symbol]['condition_5_pct']:.1f}%")


# ============================================================
# 8. چاپ جدول خلاصه
# ============================================================
print("\n" + "=" * 80)
print("Summary (LBank data)")
print("=" * 80)
print(f"{'Symbol':<12} {'Samples':<10} {'>4%':<12} {'>5%':<12}")
print("-" * 80)

lines = []
lines.append("=" * 80)
lines.append("Crypto Analysis Report - Similar 4h Candles to BTC 2026-08-16 20:00 UTC")
lines.append("=" * 80)
lines.append(f"{'Symbol':<12} {'Samples':<10} {'>4%':<12} {'>5%':<12}")
lines.append("-" * 80)

for symbol, data in results.items():
    row = f"{symbol:<12} {data['total_similar']:<10} " \
          f"{data['condition_4_pct']:<11.1f}% {data['condition_5_pct']:<11.1f}%"
    print(row)
    lines.append(row)

lines.append("=" * 80)
lines.append(f"Drop threshold: >{DROP_THRESHOLD}%  |  Rise thresholds: >{RISE_THRESHOLD_4}% and >{RISE_THRESHOLD_5}%")
lines.append(f"Horizon: {HORIZON_CANDLES} candles ({HORIZON_CANDLES * 4}h)")
lines.append("=" * 80)

# ذخیره در فایل برای artifact
os.makedirs("output", exist_ok=True)
with open("output/report.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(lines))

with open("output/results.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print("\nResults saved to output/report.txt and output/results.json")
