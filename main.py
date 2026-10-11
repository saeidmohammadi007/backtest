# main.py
# تحلیل سیگنال CCI SMA 14 + MACD با دادهٔ یاهو فایننس و ارسال نتایج به تلگرام

import yfinance as yf
import pandas as pd
import pandas_ta as ta
import numpy as np
import time
import os
import requests
from datetime import datetime

# ==================== تنظیمات ====================
TIMEFRAME = '1d'
LIMIT = 1000
FORWARD_BARS = 100

CCI_LENGTH = 20          # دوره استاندارد CCI
CCI_SMA_LENGTH = 14      # دوره SMA روی CCI

MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9

# ۱۰ ارز برتر (نمادهای یاهو فایننس)
TOP_10_SYMBOLS = [
    'BTC-USD', 'ETH-USD', 'BNB-USD', 'XRP-USD', 'SOL-USD',
    'ADA-USD', 'DOGE-USD', 'AVAX-USD', 'DOT-USD', 'LINK-USD'
]

# ==================== تلگرام ====================
TELEGRAM_BOT_TOKEN = os.environ.get('TELEGRAM_BOT_TOKEN')
TELEGRAM_CHAT_ID = os.environ.get('TELEGRAM_CHAT_ID')

def send_telegram_message(text):
    """ارسال پیام به تلگرام"""
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("  [هشدار] توکن یا chat_id تلگرام تنظیم نشده است.")
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": text,
        "parse_mode": "Markdown"
    }
    try:
        response = requests.post(url, json=payload, timeout=30)
        response.raise_for_status()
        print("  پیام تلگرام با موفقیت ارسال شد.")
    except Exception as e:
        print(f"  خطا در ارسال پیام تلگرام: {e}")

# ==================== دریافت داده از یاهو فایننس ====================
def fetch_ohlcv(symbol, timeframe=TIMEFRAME, limit=LIMIT):
    """
    دریافت داده OHLCV از یاهو فایننس با استفاده از yfinance.
    ستون‌ها به صورت open, high, low, close, volume بازگردانده می‌شوند.
    """
    print(f"  در حال دریافت داده {symbol} از یاهو فایننس...")
    try:
        # دریافت دادهٔ روزانه
        df = yf.download(
            symbol,
            period='max',          # تمام تاریخ موجود
            interval=timeframe,    # '1d'
            progress=False,
            auto_adjust=True,      # اصلاح خودکار قیمت‌ها
            threads=False
        )

        if df is None or df.empty:
            print(f"  دادهٔ خالی برای {symbol}")
            return None

        # اطمینان از تک‌سطحی بودن ستون‌ها (yfinance ممکن است MultiIndex بدهد)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        # یکسان‌سازی نام ستون‌ها با حروف کوچک
        df.columns = [c.lower() for c in df.columns]

        # حذف ستون‌های اضافی در صورت وجود
        cols_to_keep = ['open', 'high', 'low', 'close', 'volume']
        df = df[[c for c in cols_to_keep if c in df.columns]]

        # حذف ردیف‌های NaN
        df = df.dropna()

        # محدود کردن به آخرین limit کندل
        df = df.tail(limit)

        return df

    except Exception as e:
        print(f"  دریافت {symbol} با خطا مواجه شد: {e}")
        return None

# ==================== محاسبه اندیکاتورها ====================
def compute_indicators(df):
    """محاسبه CCI(20)، SMA(14) روی CCI و MACD"""
    df = df.copy()

    # CCI(20)
    df['cci'] = ta.cci(df['high'], df['low'], df['close'], length=CCI_LENGTH)

    # SMA(14) روی CCI
    df['cci_sma'] = df['cci'].rolling(window=CCI_SMA_LENGTH).mean()

    # MACD
    macd_df = ta.macd(df['close'], fast=MACD_FAST, slow=MACD_SLOW, signal=MACD_SIGNAL)
    df['macd'] = macd_df[f'MACD_{MACD_FAST}_{MACD_SLOW}_{MACD_SIGNAL}']
    df['macd_signal'] = macd_df[f'MACDs_{MACD_FAST}_{MACD_SLOW}_{MACD_SIGNAL}']

    return df

# ==================== تشخیص کراس‌ها ====================
def detect_crossovers(df):
    """
    شرایط سیگنال:
    ۱. SMA(14) روی CCI، رو به بالا خط CCI(20) را کراس کند.
    ۲. در بازهٔ ۰ تا ۲ کندل بعد، MACD هم رو به بالا خط سیگنال را کراس کند.
    """
    df = df.copy()

    # کراس رو به بالای SMA(14) از روی CCI(20)
    df['cci_sma_cross_up'] = (
        (df['cci_sma'] > df['cci']) &
        (df['cci_sma'].shift(1) <= df['cci'].shift(1))
    )

    # کراس رو به بالای MACD (طلایی)
    df['macd_cross_up'] = (
        (df['macd'] > df['macd_signal']) &
        (df['macd'].shift(1) <= df['macd_signal'].shift(1))
    )

    # ترکیب شرط
    signal = pd.Series(False, index=df.index)
    cci_cross_indices = df.index[df['cci_sma_cross_up']]

    for idx in cci_cross_indices:
        pos = df.index.get_loc(idx)
        end_pos = min(pos + 3, len(df))
        if df['macd_cross_up'].iloc[pos:end_pos].any():
            signal.iloc[pos] = True

    df['signal'] = signal
    return df

# ==================== تحلیل بازدهی آینده ====================
def analyze_forward_returns(df, symbol):
    """برای هر سیگنال، بازدهی ۱۰۰ کندل بعد را محاسبه می‌کند."""
    results = []
    signal_indices = df.index[df['signal']]

    for idx in signal_indices:
        pos = df.index.get_loc(idx)

        if pos + FORWARD_BARS >= len(df):
            continue

        entry_price = df['close'].iloc[pos]
        future_prices = df['close'].iloc[pos+1 : pos+1+FORWARD_BARS]

        if len(future_prices) < FORWARD_BARS:
            continue

        forward_return = (future_prices.iloc[-1] / entry_price - 1) * 100
        max_return = (future_prices.max() / entry_price - 1) * 100
        min_return = (future_prices.min() / entry_price - 1) * 100

        results.append({
            'symbol': symbol,
            'signal_date': idx,
            'entry_price': entry_price,
            'forward_return_pct': forward_return,
            'max_return_pct': max_return,
            'min_return_pct': min_return,
            'win': forward_return > 0
        })

    return results

# ==================== فرمت‌دهی پیام تلگرام ====================
def format_telegram_message(summary_df, results_df):
    """ساخت پیام متنی برای تلگرام"""
    lines = []
    lines.append("📊 *نتایج تحلیل CCI SMA 14 + MACD (دادهٔ یاهو فایننس)*")
    lines.append(f"🗓 تاریخ اجرا: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    lines.append("")

    lines.append("🔹 *خلاصه به تفکیک ارز:*")
    for _, row in summary_df.iterrows():
        if row['signals'] == 0:
            continue
        lines.append(
            f"*{row['symbol']}* | سیگنال: {int(row['signals'])} | "
            f"میانگین بازدهی: {row['avg_return_100d']:.2f}% | "
            f"نرخ برد: {row['win_rate']:.1f}%"
        )
    lines.append("")

    if len(results_df) > 0:
        all_returns = results_df['forward_return_pct']
        lines.append("📈 *آمار کلی:*")
        lines.append(f"تعداد کل سیگنال‌ها: {len(results_df)}")
        lines.append(f"میانگین بازدهی ۱۰۰ روزه: {all_returns.mean():.2f}%")
        lines.append(f"نرخ برد: {(all_returns > 0).mean() * 100:.1f}%")
        lines.append(f"بیشترین بازدهی: {all_returns.max():.2f}%")
        lines.append(f"بیشترین ضرر: {all_returns.min():.2f}%")

        lines.append("")
        lines.append("🕐 *آخرین سیگنال‌ها:*")
        last_signals = results_df.tail(5)
        for _, row in last_signals.iterrows():
            date_str = row['signal_date'].strftime('%Y-%m-%d')
            lines.append(
                f"• {row['symbol']} | {date_str} | "
                f"ورود: {row['entry_price']:.2f} | "
                f"بازدهی: {row['forward_return_pct']:.2f}%"
            )

    return "\n".join(lines)

# ==================== اجرای اصلی ====================
def main():
    all_results = []
    summary = []

    print("=" * 60)
    print("تحلیل سیگنال: کراس SMA(14) روی CCI(20) + کراس MACD")
    print(f"منبع داده: یاهو فایننس  |  تایم‌فریم: {TIMEFRAME}  |  کندل‌های آینده: {FORWARD_BARS}")
    print("=" * 60)

    for symbol in TOP_10_SYMBOLS:
        print(f"\nپردازش {symbol}...")

        df = fetch_ohlcv(symbol)
        if df is None or len(df) < 200:
            print(f"  داده کافی نیست، {symbol} رد شد.")
            continue

        df = compute_indicators(df)
        df = detect_crossovers(df)

        signal_count = df['signal'].sum()
        print(f"  تعداد سیگنال‌ها: {signal_count}")

        if signal_count == 0:
            summary.append({
                'symbol': symbol,
                'signals': 0,
                'avg_return_100d': np.nan,
                'win_rate': np.nan
            })
            continue

        results = analyze_forward_returns(df, symbol)
        all_results.extend(results)

        if results:
            returns = [r['forward_return_pct'] for r in results]
            wins = [r['win'] for r in results]

            summary.append({
                'symbol': symbol,
                'signals': len(results),
                'avg_return_100d': np.mean(returns),
                'win_rate': np.mean(wins) * 100
            })

        time.sleep(1)  # جلوگیری از محدودیت نرخ یاهو فایننس

    # ==================== نمایش در لاگ ====================
    summary_df = pd.DataFrame(summary)
    results_df = pd.DataFrame(all_results)

    if len(summary_df) > 0:
        print("\n" + "=" * 60)
        print("خلاصه نتایج (به تفکیک ارز)")
        print("=" * 60)
        print(summary_df.round(2).to_string(index=False))

    # ==================== ارسال به تلگرام ====================
    if len(results_df) > 0:
        results_df['signal_date'] = pd.to_datetime(results_df['signal_date'])
        message = format_telegram_message(summary_df, results_df)
        send_telegram_message(message)
    else:
        send_telegram_message("📊 تحلیل انجام شد اما هیچ سیگنالی یافت نشد.")

    print("\nپایان اجرا.")

if __name__ == '__main__':
    main()
