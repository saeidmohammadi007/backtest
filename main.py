# main.py
# تحلیل MAE/MFE برای تعیین حد ضرر و حد سود بهینه

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

# سطوح مختلف برای شبیه‌سازی
SL_LEVELS = [3, 5, 8, 10, 15, 20, 25, 30]      # حد ضرر (درصد)
TP_LEVELS = [5, 10, 15, 20, 30, 40, 50, 75]    # حد سود (درصد)

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
    print(f"  در حال دریافت داده {symbol}...")
    try:
        df = yf.download(symbol, period='max', interval=timeframe,
                         progress=False, auto_adjust=True, threads=False)
        if df is None or df.empty:
            return None
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df.columns = [c.lower() for c in df.columns]
        cols = ['open', 'high', 'low', 'close', 'volume']
        df = df[[c for c in cols if c in df.columns]]
        return df.dropna().tail(limit)
    except Exception as e:
        print(f"  خطا در دریافت {symbol}: {e}")
        return None

# ==================== محاسبهٔ اندیکاتورها ====================
def calculate_cci(high, low, close, length=20):
    tp = (high + low + close) / 3
    sma_tp = tp.rolling(window=length).mean()

    def mean_deviation(x):
        return np.mean(np.abs(x - x.mean()))

    mad = tp.rolling(window=length).apply(mean_deviation, raw=True)
    return (tp - sma_tp) / (0.015 * mad)

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

# ==================== تشخیص سیگنال ====================
def detect_crossovers(df):
    df = df.copy()

    df['cci_cross_up'] = (
        (df['cci'] > df['cci_sma']) &
        (df['cci'].shift(1) <= df['cci_sma'].shift(1))
    )
    df['macd_cross_up'] = (
        (df['macd'] > df['macd_signal']) &
        (df['macd'].shift(1) <= df['macd_signal'].shift(1))
    )

    signal = pd.Series(False, index=df.index)
    for idx in df.index[df['cci_cross_up']]:
        pos = df.index.get_loc(idx)
        end_pos = min(pos + 3, len(df))
        if df['macd_cross_up'].iloc[pos:end_pos].any():
            signal.iloc[pos] = True

    df['signal'] = signal
    return df

# ==================== تحلیل MAE/MFE ====================
def analyze_mae_mfe(df, symbol):
    """
    برای هر سیگنال:
    - MFE (سقف سود): بیشترین سود قبل از هر چیز
    - MAE قبل از سقف: بیشترین ریزش قبل از رسیدن به سقف
    - drawdown بعد از سقف: ریزش از سقف تا پایان
    """
    results = []
    for idx in df.index[df['signal']]:
        pos = df.index.get_loc(idx)
        if pos + FORWARD_BARS >= len(df):
            continue

        entry = df['close'].iloc[pos]
        future = df['close'].iloc[pos+1 : pos+1+FORWARD_BARS]
        if len(future) < FORWARD_BARS:
            continue

        # سقف و موقعیت آن
        peak_pos = int(future.values.argmax())
        peak_price = future.iloc[peak_pos]
        peak_return = (peak_price / entry - 1) * 100
        peak_candle = peak_pos + 1  # 1-indexed

        # کف کلی در کل ۱۰۰ کندل
        trough_pos = int(future.values.argmin())
        trough_return = (future.iloc[trough_pos] / entry - 1) * 100
        trough_candle = trough_pos + 1

        # بازدهی قبل از سقف (از ورود تا سقف)
        pre_peak = future.iloc[:peak_pos + 1]
        pre_peak_min = (pre_peak.min() / entry - 1) * 100  # MAE قبل از سقف

        # آیا قبل از سقف، سیگنال زیر صفر رفته؟
        went_negative = pre_peak_min < 0

        # بازدهی نهایی در کندل ۱۰۰
        final_return = (future.iloc[-1] / entry - 1) * 100

        results.append({
            'symbol': symbol,
            'entry_price': entry,
            'final_return_pct': final_return,
            'peak_return_pct': peak_return,
            'peak_candle': peak_candle,
            'trough_return_pct': trough_return,
            'trough_candle': trough_candle,
            'mae_before_peak_pct': pre_peak_min,
            'went_negative': went_negative,
            'win_at_100': final_return > 0
        })

    return results

# ==================== شبیه‌سازی SL/TP ====================
def simulate_sl_tp(df, symbol, sl_levels, tp_levels):
    """
    برای هر ترکیب SL/TP، شبیه‌سازی می‌کند که در ۱۰۰ کندل بعد چه اتفاقی می‌افتد.
    اولین‌باری که هر کدام لمس شود، خروج در همان سطح رخ می‌دهد.
    اگر هیچ‌کدام لمس نشد، خروج در کندل ۱۰۰ با قیمت واقعی.
    """
    results = []
    for idx in df.index[df['signal']]:
        pos = df.index.get_loc(idx)
        if pos + FORWARD_BARS >= len(df):
            continue

        entry = df['close'].iloc[pos]
        future = df['close'].iloc[pos+1 : pos+1+FORWARD_BARS]
        if len(future) < FORWARD_BARS:
            continue

        # محاسبه سطوح قیمتی
        for sl in sl_levels:
            for tp in tp_levels:
                sl_price = entry * (1 - sl / 100)
                tp_price = entry * (1 + tp / 100)

                exit_reason = 'timeout'
                exit_return = (future.iloc[-1] / entry - 1) * 100

                # پیمایش کندل به کندل
                # (از low و high استفاده می‌کنیم نه close)
                future_high = df['high'].iloc[pos+1 : pos+1+FORWARD_BARS]
                future_low = df['low'].iloc[pos+1 : pos+1+FORWARD_BARS]

                for j in range(len(future)):
                    hit_sl = future_low.iloc[j] <= sl_price
                    hit_tp = future_high.iloc[j] >= tp_price

                    # اگر هر دو در یک کندل لمس شوند، فرض می‌کنیم SL اول خورده
                    if hit_sl and hit_tp:
                        exit_reason = 'sl_first'
                        exit_return = -sl
                        break
                    elif hit_sl:
                        exit_reason = 'sl'
                        exit_return = -sl
                        break
                    elif hit_tp:
                        exit_reason = 'tp'
                        exit_return = tp
                        break

                results.append({
                    'symbol': symbol,
                    'sl': sl,
                    'tp': tp,
                    'exit_reason': exit_reason,
                    'return_pct': exit_return
                })

    return results

# ==================== گزارش MAE/MFE ====================
def build_mae_mfe_report(results_df):
    lines = []
    lines.append("📉 *تحلیل MAE/MFE (برای تعیین SL/TP)*")
    lines.append("")

    total = len(results_df)
    winners = results_df[results_df['win_at_100']]
    losers = results_df[~results_df['win_at_100']]

    lines.append(f"کل سیگنال‌ها: {total}")
    lines.append(f"برنده (کندل ۱۰۰): {len(winners)} ({len(winners)/total*100:.1f}%)")
    lines.append(f"بازنده: {len(losers)} ({len(losers)/total*100:.1f}%)")
    lines.append("")

    # === تحلیل برنده‌ها ===
    lines.append("✅ *برنده‌ها چه ریزشی قبل از سقف داشتند؟*")
    w = winners['mae_before_peak_pct']
    lines.append(f"میانگین MAE: {w.mean():.2f}%")
    lines.append(f"میانه MAE: {w.median():.2f}%")

    for level in [3, 5, 8, 10, 15, 20]:
        pct = (w >= -level).mean() * 100
        lines.append(f"  بدون ریزش بیش از -{level}%: {pct:.1f}%")
    lines.append("")

    # === سقف برنده‌ها ===
    lines.append("🎯 *برنده‌ها چه سودی رسیدند؟*")
    p = winners['peak_return_pct']
    lines.append(f"میانگین سقف: {p.mean():.2f}%")
    lines.append(f"میانه سقف: {p.median():.2f}%")

    for level in [5, 10, 15, 20, 30, 50]:
        pct = (p >= level).mean() * 100
        lines.append(f"  سقف بالای +{level}%: {pct:.1f}%")
    lines.append("")

    # === تحلیل بازنده‌ها ===
    lines.append("❌ *بازنده‌ها چه سودی قبل از ریزش داشتند؟*")
    l = losers['peak_return_pct']
    lines.append(f"میانگین سقف بازنده‌ها: {l.mean():.2f}%")
    lines.append(f"میانه سقف بازنده‌ها: {l.median():.2f}%")

    for level in [3, 5, 10, 15, 20]:
        pct = (l >= level).mean() * 100
        lines.append(f"  سقف بالای +{level}%: {pct:.1f}%")

    return "\n".join(lines)

# ==================== گزارش SL/TP ====================
def build_sl_tp_report(sim_df):
    """
    برای هر ترکیب SL/TP، میانگین بازدهی، نرخ برد و تعداد معاملات را محاسبه می‌کند.
    """
    lines = []
    lines.append("")
    lines.append("🧮 *بهترین ترکیب‌های SL/TP (بر اساس میانگین بازدهی):*")
    lines.append("")

    # گروه‌بندی بر اساس SL و TP
    grouped = sim_df.groupby(['sl', 'tp']).agg(
        avg_return=('return_pct', 'mean'),
        win_rate=('return_pct', lambda x: (x > 0).mean() * 100),
        count=('return_pct', 'size')
    ).reset_index()

    # مرتب‌سازی بر اساس میانگین بازدهی
    grouped = grouped.sort_values('avg_return', ascending=False)

    # بهترین ۱۰ ترکیب
    top = grouped.head(10)
    lines.append("*۱۰ ترکیب برتر:*")
    for _, row in top.iterrows():
        lines.append(
            f"SL=-{int(row['sl'])}% / TP=+{int(row['tp'])}% | "
            f"بازدهی: {row['avg_return']:.2f}% | "
            f"برد: {row['win_rate']:.1f}%"
        )

    lines.append("")
    lines.append("*بدترین ۵ ترکیب:*")
    bottom = grouped.tail(5)
    for _, row in bottom.iterrows():
        lines.append(
            f"SL=-{int(row['sl'])}% / TP=+{int(row['tp'])}% | "
            f"بازدهی: {row['avg_return']:.2f}% | "
            f"برد: {row['win_rate']:.1f}%"
        )

    return "\n".join(lines)

# ==================== اجرای اصلی ====================
def main():
    all_mae_mfe = []
    all_sim = []

    print("=" * 60)
    print("تحلیل MAE/MFE و شبیه‌سازی SL/TP")
    print("=" * 60)

    for symbol in TOP_10_SYMBOLS:
        print(f"\nپردازش {symbol}...")

        df = fetch_ohlcv(symbol)
        if df is None or len(df) < 200:
            continue

        df = compute_indicators(df)
        df = detect_crossovers(df)

        signal_count = int(df['signal'].sum())
        print(f"  تعداد سیگنال: {signal_count}")
        if signal_count == 0:
            continue

        mae_mfe = analyze_mae_mfe(df, symbol)
        all_mae_mfe.extend(mae_mfe)

        sim = simulate_sl_tp(df, symbol, SL_LEVELS, TP_LEVELS)
        all_sim.extend(sim)

        time.sleep(1)

    mae_df = pd.DataFrame(all_mae_mfe)
    sim_df = pd.DataFrame(all_sim)

    if len(mae_df) == 0:
        send_telegram_message("هیچ سیگنالی یافت نشد.")
        return

    # نمایش در لاگ
    print("\n" + "=" * 60)
    print("تحلیل MAE/MFE")
    print("=" * 60)
    print(mae_df[['symbol', 'peak_return_pct', 'mae_before_peak_pct',
                  'peak_candle', 'final_return_pct', 'win_at_100']].round(2).to_string(index=False))

    # ارسال به تلگرام
    message = "📊 *تحلیل MAE/MFE و SL/TP*\n"
    message += f"🗓 {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n"
    message += build_mae_mfe_report(mae_df)
    message += "\n"
    message += build_sl_tp_report(sim_df)

    send_telegram_message(message)

    # ذخیره CSV برای بررسی بیشتر
    mae_df.to_csv('mae_mfe.csv', index=False)
    sim_df.to_csv('sl_tp_simulation.csv', index=False)

    print("\nپایان اجرا.")

if __name__ == '__main__':
    main()
