# main.py
# تحلیل سیگنال CCI SMA 14 + MACD با دادهٔ یاهو فایننس
# همراه با ثبت موقعیت (کندل) سقف و کف در ۱۰۰ کندل بعدی

import yfinance as yf
import pandas as pd
import numpy as np
import time
import os
import requests
from datetime import datetime

# ==================== تنظیمات ====================
TIMEFRAME = '1d'
LIMIT = 1000
FORWARD_BARS = 100

CCI_LENGTH = 20
CCI_SMA_LENGTH = 14

MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9

TOP_10_SYMBOLS = [
    'BTC-USD', 'ETH-USD', 'BNB-USD', 'XRP-USD', 'SOL-USD',
    'ADA-USD', 'DOGE-USD', 'AVAX-USD', 'DOT-USD', 'LINK-USD'
]

# ==================== تلگرام ====================
TELEGRAM_BOT_TOKEN = os.environ.get('TELEGRAM_BOT_TOKEN')
TELEGRAM_CHAT_ID = os.environ.get('TELEGRAM_CHAT_ID')

def send_telegram_message(text):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("  [هشدار] توکن یا chat_id تلگرام تنظیم نشده است.")
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": text, "parse_mode": "Markdown"}
    try:
        response = requests.post(url, json=payload, timeout=30)
        response.raise_for_status()
        print("  پیام تلگرام با موفقیت ارسال شد.")
    except Exception as e:
        print(f"  خطا در ارسال پیام تلگرام: {e}")

# ==================== دریافت داده ====================
def fetch_ohlcv(symbol, timeframe=TIMEFRAME, limit=LIMIT):
    print(f"  در حال دریافت داده {symbol} از یاهو فایننس...")
    try:
        df = yf.download(symbol, period='max', interval=timeframe,
                         progress=False, auto_adjust=True, threads=False)
        if df is None or df.empty:
            print(f"  دادهٔ خالی برای {symbol}")
            return None
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df.columns = [c.lower() for c in df.columns]
        cols = ['open', 'high', 'low', 'close', 'volume']
        df = df[[c for c in cols if c in df.columns]]
        df = df.dropna().tail(limit)
        return df
    except Exception as e:
        print(f"  دریافت {symbol} با خطا مواجه شد: {e}")
        return None

# ==================== محاسبهٔ دستی اندیکاتورها ====================
def calculate_cci(high, low, close, length=20):
    tp = (high + low + close) / 3
    sma_tp = tp.rolling(window=length).mean()

    def mean_deviation(x):
        return np.mean(np.abs(x - x.mean()))

    mad = tp.rolling(window=length).apply(mean_deviation, raw=True)
    cci = (tp - sma_tp) / (0.015 * mad)
    return cci

def calculate_macd(close, fast=12, slow=26, signal=9):
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    return macd_line, signal_line

def compute_indicators(df):
    df = df.copy()
    df['cci'] = calculate_cci(df['high'], df['low'], df['close'], length=CCI_LENGTH)
    df['cci_sma'] = df['cci'].rolling(window=CCI_SMA_LENGTH).mean()
    macd_line, signal_line = calculate_macd(df['close'], fast=MACD_FAST, slow=MACD_SLOW, signal=MACD_SIGNAL)
    df['macd'] = macd_line
    df['macd_signal'] = signal_line
    return df

# ==================== تشخیص کراس‌ها ====================
def detect_crossovers(df):
    df = df.copy()

    df['cci_sma_cross_up'] = (
        (df['cci_sma'] > df['cci']) &
        (df['cci_sma'].shift(1) <= df['cci'].shift(1))
    )
    df['macd_cross_up'] = (
        (df['macd'] > df['macd_signal']) &
        (df['macd'].shift(1) <= df['macd_signal'].shift(1))
    )

    signal = pd.Series(False, index=df.index)
    for idx in df.index[df['cci_sma_cross_up']]:
        pos = df.index.get_loc(idx)
        end_pos = min(pos + 3, len(df))
        if df['macd_cross_up'].iloc[pos:end_pos].any():
            signal.iloc[pos] = True

    df['signal'] = signal
    return df

# ==================== تحلیل بازدهی آینده ====================
def analyze_forward_returns(df, symbol):
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

        peak_candle = int(future_prices.values.argmax() + 1)
        trough_candle = int(future_prices.values.argmin() + 1)

        ret_30 = (future_prices.iloc[29] / entry_price - 1) * 100 if len(future_prices) >= 30 else np.nan
        ret_50 = (future_prices.iloc[49] / entry_price - 1) * 100 if len(future_prices) >= 50 else np.nan
        ret_60 = (future_prices.iloc[59] / entry_price - 1) * 100 if len(future_prices) >= 60 else np.nan

        results.append({
            'symbol': symbol,
            'signal_date': idx,
            'entry_price': entry_price,
            'forward_return_pct': forward_return,
            'max_return_pct': max_return,
            'max_return_candle': peak_candle,
            'min_return_pct': min_return,
            'min_return_candle': trough_candle,
            'ret_30_pct': ret_30,
            'ret_50_pct': ret_50,
            'ret_60_pct': ret_60,
            'win': forward_return > 0
        })

    return results

# ==================== فرمت‌دهی پیام تلگرام ====================
def format_telegram_message(summary_df, results_df):
    lines = []
    lines.append("📊 *نتایج تحلیل CCI SMA 14 + MACD (یاهو فایننس)*")
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
        lines.append("📈 *آمار کلی (کندل ۱۰۰):*")
        lines.append(f"تعداد کل سیگنال‌ها: {len(results_df)}")
        lines.append(f"میانگین بازدهی ۱۰۰ روزه: {all_returns.mean():.2f}%")
        lines.append(f"نرخ برد: {(all_returns > 0).mean() * 100:.1f}%")
        lines.append(f"بیشترین بازدهی: {all_returns.max():.2f}%")
        lines.append(f"بیشترین ضرر: {all_returns.min():.2f}%")

        lines.append("")
        lines.append("⏱ *کِی سقف/کف می‌زند؟*")
        lines.append(f"میانگین کندل سقف: {results_df['max_return_candle'].mean():.1f}")
        lines.append(f"میانه کندل سقف: {results_df['max_return_candle'].median():.0f}")
        lines.append(f"میانگین کندل کف: {results_df['min_return_candle'].mean():.1f}")
        lines.append(f"سقف قبل از کندل ۵۰: {(results_df['max_return_candle'] <= 50).mean() * 100:.1f}٪ موارد")
        lines.append(f"میانگین بازدهی در سقف: {results_df['max_return_pct'].mean():.2f}%")

        lines.append("")
        lines.append("📊 *مقایسهٔ بازدهی در زمان‌های مختلف:*")
        lines.append(f"کندل ۳۰: {results_df['ret_30_pct'].mean():.2f}%")
        lines.append(f"کندل ۵۰: {results_df['ret_50_pct'].mean():.2f}%")
        lines.append(f"کندل ۶۰: {results_df['ret_60_pct'].mean():.2f}%")
        lines.append(f"کندل ۱۰۰: {results_df['forward_return_pct'].mean():.2f}%")

        lines.append("")
        lines.append("🏆 *۵ سیگنال آخر:*")
        for _, row in results_df.tail(5).iterrows():
            date_str = row['signal_date'].strftime('%Y-%m-%d')
            lines.append(
                f"• {row['symbol']} | {date_str} | "
                f"ورود: {row['entry_price']:.2f} | "
                f"۱۰۰ کندل: {row['forward_return_pct']:.2f}% | "
                f"سقف: {row['max_return_pct']:.2f}% (کندل {int(row['max_return_candle'])})"
            )

    return "\n".join(lines)

# ==================== اجرای اصلی ====================
def main():
    all_results = []
    summary = []

    print("=" * 60)
    print("تحلیل سیگنال: کراس SMA(14) روی CCI(20) + کراس MACD")
    print(f"منبع داده: یاهو فایننس  |  کندل‌های آینده: {FORWARD_BARS}")
    print("=" * 60)

    for symbol in TOP_10_SYMBOLS:
        print(f"\nپردازش {symbol}...")

        df = fetch_ohlcv(symbol)
        if df is None or len(df) < 200:
            print(f"  داده کافی نیست، {symbol} رد شد.")
            continue

        df = compute_indicators(df)
        df = detect_crossovers(df)

        signal_count = int(df['signal'].sum())
        print(f"  تعداد سیگنال‌ها: {signal_count}")

        if signal_count == 0:
            summary.append({'symbol': symbol, 'signals': 0,
                            'avg_return_100d': np.nan, 'win_rate': np.nan})
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

        time.sleep(1)

    summary_df = pd.DataFrame(summary)
    results_df = pd.DataFrame(all_results)

    if len(summary_df) > 0:
        print("\n" + "=" * 60)
        print("خلاصه نتایج (به تفکیک ارز)")
        print("=" * 60)
        print(summary_df.round(2).to_string(index=False))

    if len(results_df) > 0:
        print("\n" + "=" * 60)
        print("آمار زمان سقف و کف")
        print("=" * 60)
        print(f"  میانگین کندل سقف: {results_df['max_return_candle'].mean():.1f}")
        print(f"  میانه کندل سقف:   {results_df['max_return_candle'].median():.0f}")
        print(f"  میانگین کندل کف:  {results_df['min_return_candle'].mean():.1f}")
        print(f"  سقف قبل از کندل ۵۰: {(results_df['max_return_candle'] <= 50).mean() * 100:.1f}٪")
        print("\n  مقایسهٔ بازدهی میانگین در زمان‌های مختلف:")
        print(f"    کندل ۳۰:  {results_df['ret_30_pct'].mean():.2f}%")
        print(f"    کندل ۵۰:  {results_df['ret_50_pct'].mean():.2f}%")
        print(f"    کندل ۶۰:  {results_df['ret_60_pct'].mean():.2f}%")
        print(f"    کندل ۱۰۰: {results_df['forward_return_pct'].mean():.2f}%")

    if len(results_df) > 0:
        results_df['signal_date'] = pd.to_datetime(results_df['signal_date'])
        message = format_telegram_message(summary_df, results_df)
        send_telegram_message(message)
    else:
        send_telegram_message("📊 تحلیل انجام شد اما هیچ سیگنالی یافت نشد.")

    print("\nپایان اجرا.")

if __name__ == '__main__':
    main()
