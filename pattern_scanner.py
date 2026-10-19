
import yfinance as yf
import pandas as pd
import numpy as np
from dtw import dtw
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# ---------- تنظیمات ----------
REFERENCE_DATE = "2024-01-26"
WINDOW_SIZE = 12          # تعداد کندل‌های ۴ ساعته برای الگو (۱۲ کندل = ۲ روز)
FORWARD_BARS = 100        # بررسی ۱۰۰ کندل بعدی
MAX_INITIAL_DROP = 0.04   # حداکثر ریزش اولیه مجاز (۴٪)
TARGET_GAIN = 0.10        # هدف سود (۱۰٪)

# ۲۰ ارز برتر بر اساس مارکت‌کپ (تکراری‌ها و استیبل‌کوین‌ها حذف شده‌اند)
TOP_20_CRYPTOS = [
    "BTC-USD", "ETH-USD", "BNB-USD", "XRP-USD", "SOL-USD",
    "DOGE-USD", "ADA-USD", "TRX-USD", "AVAX-USD", "TON-USD",
    "LINK-USD", "SHIB-USD", "DOT-USD", "BCH-USD", "LTC-USD",
    "NEAR-USD", "MATIC-USD", "UNI-USD", "ICP-USD", "ETC-USD"
]

def download_1h_data(ticker, start="2020-01-01", end="2026-01-01"):
    """دانلود داده ۱ ساعته و تبدیل به ۴ ساعته"""
    try:
        df = yf.download(
            ticker,
            start=start,
            end=end,
            interval="1h",
            progress=False,
            auto_adjust=True
        )
        if df.empty:
            return None

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        ohlc_dict = {
            'Open': 'first',
            'High': 'max',
            'Low': 'min',
            'Close': 'last',
            'Volume': 'sum'
        }
        df_4h = df.resample('4H').agg(ohlc_dict).dropna()
        return df_4h
    except Exception as e:
        print(f"خطا در دانلود {ticker}: {e}")
        return None

def get_window_around_date(df, ref_date, window_size):
    """استخراج پنجره الگو حول تاریخ مرجع (۲۶ ژانویه ۲۰۲۴)"""
    ref_dt = pd.Timestamp(ref_date, tz='UTC')
    if ref_dt not in df.index:
        idx = df.index.get_indexer([ref_dt], method='nearest')[0]
    else:
        idx = df.index.get_loc(ref_dt)

    start_idx = idx - window_size // 2
    end_idx = idx + window_size // 2 + 1

    if start_idx < 0 or end_idx > len(df):
        return None

    window = df.iloc[start_idx:end_idx].copy()
    return window

def find_similar_windows(full_df, ref_window, window_size, top_k=10):
    """
    جستجوی پنجره‌های مشابه با DTW در کل تاریخچه
    از تغییرات درصدی Close استفاده می‌کند تا مقیاس قیمت حذف شود.
    """
    ref_series = ref_window['Close'].pct_change().dropna().values
    if len(ref_series) < window_size - 1:
        return []

    results = []
    closes = full_df['Close'].values

    # حرکت پنجره‌ای روی داده (با گام ۱ کندل)
    for i in range(window_size, len(closes) - FORWARD_BARS - 1):
        segment = closes[i - window_size:i]
        seg_series = pd.Series(segment).pct_change().dropna().values

        if len(seg_series) < len(ref_series):
            continue

        try:
            alignment = dtw(ref_series, seg_series, keep_internals=False)
            distance = alignment.normalizedDistance
        except:
            continue

        results.append({
            'start_idx': i - window_size,
            'end_idx': i,
            'distance': distance
        })

    results.sort(key=lambda x: x['distance'])
    return results[:top_k]

def analyze_outcome(full_df, window_info):
    """
    تحلیل نتیجه پس از پنجره مشابه:
    ۱. آیا ریزش اولیه کمتر از ۴٪ بوده؟
    ۲. آیا در ۱۰۰ کندل بعدی به هدف ۱۰٪ رسیده؟
    """
    end_idx = window_info['end_idx']
    start_price = full_df['Close'].iloc[end_idx - 1]  # قیمت بسته شدن کندل قبل از شروع بررسی

    future_df = full_df.iloc[end_idx:end_idx + FORWARD_BARS]
    if future_df.empty:
        return None

    # محاسبه حداکثر ریزش اولیه (بر اساس Low)
    initial_low = future_df['Low'].min()
    initial_drop_pct = (start_price - initial_low) / start_price

    # محاسبه حداکثر رشد (بر اساس High)
    max_high = future_df['High'].max()
    max_gain_pct = (max_high - start_price) / start_price

    # شرط: ریزش اولیه کمتر از ۴٪ باشد
    valid_drop = initial_drop_pct < MAX_INITIAL_DROP

    # شرط: به هدف ۱۰٪ رسیده باشد
    hit_target = max_gain_pct >= TARGET_GAIN

    return {
        'start_price': round(start_price, 6),
        'initial_drop_pct': round(initial_drop_pct * 100, 2),
        'max_gain_pct': round(max_gain_pct * 100, 2),
        'valid_drop': valid_drop,
        'hit_target': hit_target,
        'success': valid_drop and hit_target
    }

def main():
    print(f"🔍 شروع اسکن الگوی مشابه برای تاریخ {REFERENCE_DATE} در تایم‌فریم ۴ ساعته\n")

    # ۱. دریافت داده مرجع از BTC (به عنوان نماینده بازار)
    print("📥 دانلود داده مرجع BTC-USD برای استخراج الگو...")
    btc_df = download_1h_data("BTC-USD")
    if btc_df is None:
        print("❌ دانلود داده مرجع ناموفق بود.")
        return

    ref_window = get_window_around_date(btc_df, REFERENCE_DATE, WINDOW_SIZE)
    if ref_window is None:
        print("❌ پنجره مرجع پیدا نشد.")
        return

    print(f"✅ پنجره مرجع استخراج شد: {len(ref_window)} کندل ۴ ساعته\n")

    all_results = []

    for ticker in TOP_20_CRYPTOS:
        print(f"--- پردازش {ticker} ---")
        df_4h = download_1h_data(ticker)
        if df_4h is None or len(df_4h) < WINDOW_SIZE + FORWARD_BARS + 10:
            print(f"⚠️ داده کافی برای {ticker} وجود ندارد. رد شد.\n")
            continue

        similar_windows = find_similar_windows(df_4h, ref_window, WINDOW_SIZE, top_k=5)

        if not similar_windows:
            print(f"⚠️ هیچ پنجره مشابهی برای {ticker} پیدا نشد.\n")
            continue

        for win in similar_windows:
            outcome = analyze_outcome(df_4h, win)
            if outcome:
                outcome['ticker'] = ticker
                outcome['window_start'] = df_4h.index[win['start_idx']].strftime('%Y-%m-%d %H:%M')
                outcome['dtw_distance'] = round(win['distance'], 4)
                all_results.append(outcome)
                status = "✅ موفق" if outcome['success'] else "❌ ناموفق"
                print(f"  پنجره {outcome['window_start']} | DTW: {outcome['dtw_distance']} | "
                      f"ریزش اولیه: {outcome['initial_drop_pct']}٪ | "
                      f"حداکثر رشد: {outcome['max_gain_pct']}٪ | {status}")
        print()

    # خلاصه نهایی
    if all_results:
        summary_df = pd.DataFrame(all_results)
        total = len(summary_df)
        valid_drops = summary_df['valid_drop'].sum()
        successes = summary_df['success'].sum()

        print("=" * 60)
        print("📊 خلاصه نتایج")
        print("=" * 60)
        print(f"تعداد کل موارد مشابه بررسی‌شده: {total}")
        print(f"موارد با ریزش اولیه کمتر از ۴٪: {valid_drops} ({valid_drops/total*100:.1f}٪)")
        print(f"مواردی که به هدف ۱۰٪ رسیده‌اند: {successes} ({successes/total*100:.1f}٪ از کل)")
        if valid_drops > 0:
            print(f"نرخ موفقیت در میان موارد با ریزش مجاز: {successes/valid_drops*100:.1f}٪")

        # ذخیره خروجی
        summary_df.to_csv("pattern_scan_results.csv", index=False)
        print("\n💾 نتایج در فایل pattern_scan_results.csv ذخیره شد.")
    else:
        print("❌ هیچ نتیجه‌ای برای تحلیل به دست نیامد.")

if __name__ == "__main__":
    main()
