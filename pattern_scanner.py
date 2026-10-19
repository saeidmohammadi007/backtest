import yfinance as yf
import pandas as pd
import numpy as np
from dtw import dtw
import requests
import time
import warnings
warnings.filterwarnings('ignore')

# ---------- تنظیمات ----------
REFERENCE_DATE = "2024-01-26"
WINDOW_SIZE = 6
FORWARD_BARS = 100
MAX_INITIAL_DROP = 0.04
TARGET_GAIN = 0.10

TOP_20_YAHOO = [
    "BTC-USD", "ETH-USD", "BNB-USD", "XRP-USD", "SOL-USD",
    "DOGE-USD", "ADA-USD", "TRX-USD", "AVAX-USD", "LINK-USD",
    "SHIB-USD", "DOT-USD", "BCH-USD", "LTC-USD", "NEAR-USD",
    "UNI-USD", "ICP-USD", "ETC-USD", "XLM-USD", "SUI-USD"
]

TOP_20_NOBITEX = [
    "BTCIRT", "ETHIRT", "BNBIRT", "XRPIRT", "SOLIRT",
    "DOGEIRT", "ADAIRT", "TRXIRT", "AVAXIRT", "LINKIRT",
    "SHIBIRT", "DOTIRT", "BCHIRT", "LTCIRT", "NEARIRT",
    "UNIIRT", "ICPIRT", "ETCIRT", "XLMIRT", "SUIIRT"
]

NOBITEX_URL = "https://apiv2.nobitex.ir/market/udf/history"

def get_reference_candle_from_yahoo(symbol="BTC-USD", date="2024-01-26"):
    """دریافت کندل روزانه از یاهو برای تاریخ مشخص"""
    try:
        df = yf.download(
            symbol,
            start=date,
            end=(pd.Timestamp(date) + pd.Timedelta(days=1)).strftime('%Y-%m-%d'),
            interval="1d",
            progress=False,
            auto_adjust=True
        )
        if df.empty:
            print(f"⚠️ کندل روزانه {date} برای {symbol} در یاهو موجود نیست.")
            return None
        # اگر ستون‌ها MultiIndex بودند، flatten کنیم
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        return df
    except Exception as e:
        print(f"❌ خطا در دریافت کندل مرجع از یاهو: {e}")
        return None

def download_4h_from_nobitex(symbol, start="2020-01-01", end=None):
    """دریافت داده ۴ ساعته از نوبیتکس برای جستجوی الگو"""
    if end is None:
        end = pd.Timestamp.utcnow().strftime('%Y-%m-%d')
    start_ts = int(pd.Timestamp(start, tz='UTC').timestamp())
    end_ts = int(pd.Timestamp(end, tz='UTC').timestamp())
    all_candles = []
    page = 1
    while True:
        params = {
            'symbol': symbol,
            'resolution': '240',
            'from': start_ts,
            'to': end_ts,
            'page': page
        }
        try:
            r = requests.get(NOBITEX_URL, params=params, timeout=30)
            if r.status_code != 200:
                break
            data = r.json()
        except Exception:
            break
        if data.get('s') != 'ok':
            break
        t, o, h, l, c, v = (data.get(k, []) for k in ['t', 'o', 'h', 'l', 'c', 'v'])
        if not t:
            break
        for i in range(len(t)):
            all_candles.append({
                'timestamp': t[i],
                'Open': float(o[i]),
                'High': float(h[i]),
                'Low': float(l[i]),
                'Close': float(c[i]),
                'Volume': float(v[i])
            })
        if len(t) < 500:
            break
        page += 1
        time.sleep(0.5)
    if not all_candles:
        return None
    df = pd.DataFrame(all_candles)
    df['datetime'] = pd.to_datetime(df['timestamp'], unit='s', utc=True)
    return df.set_index('datetime').sort_index()[['Open', 'High', 'Low', 'Close', 'Volume']]

def extract_reference_pattern(df_4h, ref_date, window_size):
    """استخراج الگوی ۶ کندلی ۴ ساعته از داده ۴ ساعته (برای DTW)"""
    ref_dt = pd.Timestamp(ref_date, tz='UTC')
    idx = df_4h.index.get_indexer([ref_dt], method='nearest')[0]
    window = df_4h.iloc[idx:idx + window_size]
    if len(window) < window_size:
        return None
    return window

def find_similar_windows(full_df, ref_window, window_size, top_k=5):
    ref_series = ref_window['Close'].pct_change().dropna().values
    if len(ref_series) < 2:
        return []
    results = []
    closes = full_df['Close'].values
    for i in range(window_size, len(closes) - FORWARD_BARS - 1):
        seg = closes[i - window_size:i]
        seg_series = pd.Series(seg).pct_change().dropna().values
        if len(seg_series) < len(ref_series):
            continue
        try:
            alignment = dtw(ref_series, seg_series, keep_internals=False)
            dist = alignment.normalizedDistance
        except Exception:
            continue
        results.append({'start_idx': i - window_size, 'end_idx': i, 'distance': dist})
    results.sort(key=lambda x: x['distance'])
    return results[:top_k]

def analyze_outcome(full_df, window_info):
    end_idx = window_info['end_idx']
    start_price = full_df['Close'].iloc[end_idx - 1]
    future = full_df.iloc[end_idx:end_idx + FORWARD_BARS]
    if future.empty:
        return None
    initial_drop = (start_price - future['Low'].min()) / start_price
    max_gain = (future['High'].max() - start_price) / start_price
    return {
        'initial_drop_pct': round(initial_drop * 100, 2),
        'max_gain_pct': round(max_gain * 100, 2),
        'valid_drop': initial_drop < MAX_INITIAL_DROP,
        'hit_target': max_gain >= TARGET_GAIN,
        'success': (initial_drop < MAX_INITIAL_DROP) and (max_gain >= TARGET_GAIN)
    }

def main():
    print(f"🔍 اسکن الگوی مشابه تاریخ {REFERENCE_DATE} (تایم‌فریم ۴ ساعته)\n")

    # ۱. دریافت کندل روزانه مرجع مستقیماً از یاهو
    print("📥 دریافت کندل روزانه مرجع از یاهو (BTC-USD)...")
    ref_candle = get_reference_candle_from_yahoo("BTC-USD", REFERENCE_DATE)
    if ref_candle is None:
        print("❌ دریافت کندل مرجع از یاهو ناموفق بود.")
        return
    print(f"✅ کندل روزانه {REFERENCE_DATE} از یاهو دریافت شد:")
    print(ref_candle[['Open', 'High', 'Low', 'Close']].to_string())

    # ۲. دریافت داده ۴ ساعته برای استخراج الگو و جستجو
    print("\n📥 دریافت داده ۴ ساعته BTCIRT از نوبیتکس برای استخراج الگو...")
    btc_4h = download_4h_from_nobitex("BTCIRT")
    if btc_4h is None:
        print("❌ دریافت داده ۴ ساعته ناموفق بود.")
        return

    ref_window = extract_reference_pattern(btc_4h, REFERENCE_DATE, WINDOW_SIZE)
    if ref_window is None:
        print("❌ الگوی ۴ ساعته مرجع استخراج نشد.")
        return
    print(f"✅ الگوی ۴ ساعته مرجع: {len(ref_window)} کندل از {ref_window.index[0]}\n")

    all_results = []
    for i, symbol in enumerate(TOP_20_NOBITEX):
        print(f"--- {symbol} ---")
        df_4h = download_4h_from_nobitex(symbol)
        if df_4h is None or len(df_4h) < WINDOW_SIZE + FORWARD_BARS + 10:
            print("  ⚠️ داده کافی نیست.\n")
            continue
        similar = find_similar_windows(df_4h, ref_window, WINDOW_SIZE, top_k=5)
        if not similar:
            print("  ⚠️ پنجره مشابهی یافت نشد.\n")
            continue
        for win in similar:
            outcome = analyze_outcome(df_4h, win)
            if not outcome:
                continue
            outcome['symbol'] = symbol
            outcome['window_start'] = df_4h.index[win['start_idx']].strftime('%Y-%m-%d %H:%M')
            outcome['dtw_distance'] = round(win['distance'], 4)
            all_results.append(outcome)
            status = "✅" if outcome['success'] else "❌"
            print(f"  {status} {outcome['window_start']} | DTW={outcome['dtw_distance']} | "
                  f"ریزش={outcome['initial_drop_pct']}٪ | رشد={outcome['max_gain_pct']}٪")
        print()

    if all_results:
        df = pd.DataFrame(all_results)
        total = len(df)
        valid = int(df['valid_drop'].sum())
        success = int(df['success'].sum())
        print("=" * 60)
        print(f"کل موارد: {total} | ریزش مجاز: {valid} | موفق: {success}")
        if valid > 0:
            print(f"نرخ موفقیت در موارد با ریزش مجاز: {success/valid*100:.1f}٪")
        df.to_csv("pattern_scan_results.csv", index=False)
        print("💾 نتایج ذخیره شد.")
    else:
        print("❌ نتیجه‌ای یافت نشد.")

if __name__ == "__main__":
    main()
