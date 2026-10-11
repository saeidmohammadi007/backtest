# main.py
# نسخهٔ دقیق: تفکیک کامل TP hit، SL hit، timeout، و سطوح سود

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
MACD_FAST, MACD_SLOW, MACD_SIGNAL = 12, 26, 9

TOP_10_SYMBOLS = [
    'BTC-USD', 'ETH-USD', 'BNB-USD', 'XRP-USD', 'SOL-USD',
    'ADA-USD', 'DOGE-USD', 'AVAX-USD', 'DOT-USD', 'LINK-USD'
]

SL_LEVELS = [3, 5, 8, 10, 15, 20, 25, 30]
TP_LEVELS = [5, 10, 15, 20, 30, 40, 50, 65, 75, 100]

PROFIT_THRESHOLDS = [10, 15, 20, 30, 40, 50, 65, 75, 100, 150, 200, 300]

# ==================== تلگرام ====================
TELEGRAM_BOT_TOKEN = os.environ.get('TELEGRAM_BOT_TOKEN')
TELEGRAM_CHAT_ID = os.environ.get('TELEGRAM_CHAT_ID')

def send_telegram_message(text):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("  [هشدار] توکن یا chat_id تنظیم نشده.")
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    try:
        r = requests.post(url, json={
            "chat_id": TELEGRAM_CHAT_ID,
            "text": text,
            "parse_mode": "Markdown"
        }, timeout=30)
        r.raise_for_status()
        print("  پیام تلگرام ارسال شد.")
    except Exception as e:
        print(f"  خطا در ارسال: {e}")

# ==================== داده ====================
def fetch_ohlcv(symbol):
    print(f"  دریافت {symbol}...")
    try:
        df = yf.download(symbol, period='max', interval=TIMEFRAME,
                         progress=False, auto_adjust=True, threads=False)
        if df is None or df.empty:
            return None
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df.columns = [c.lower() for c in df.columns]
        df = df[['open', 'high', 'low', 'close', 'volume']]
        return df.dropna().tail(LIMIT)
    except Exception as e:
        print(f"  خطا: {e}")
        return None

# ==================== اندیکاتورها ====================
def calc_cci(high, low, close, length=20):
    tp = (high + low + close) / 3
    sma_tp = tp.rolling(window=length).mean()
    mad = tp.rolling(window=length).apply(
        lambda x: np.mean(np.abs(x - x.mean())), raw=True)
    return (tp - sma_tp) / (0.015 * mad)

def calc_macd(close, fast=12, slow=26, signal=9):
    ema_f = close.ewm(span=fast, adjust=False).mean()
    ema_s = close.ewm(span=slow, adjust=False).mean()
    macd = ema_f - ema_s
    return macd, macd.ewm(span=signal, adjust=False).mean()

def compute_indicators(df):
    df = df.copy()
    df['cci'] = calc_cci(df['high'], df['low'], df['close'], CCI_LENGTH)
    df['cci_sma'] = df['cci'].rolling(window=CCI_SMA_LENGTH).mean()
    m, s = calc_macd(df['close'], MACD_FAST, MACD_SLOW, MACD_SIGNAL)
    df['macd'], df['macd_signal'] = m, s
    return df

# ==================== سیگنال ====================
def detect_signals(df):
    df = df.copy()
    df['cci_cross'] = (df['cci'] > df['cci_sma']) & (df['cci'].shift(1) <= df['cci_sma'].shift(1))
    df['macd_cross'] = (df['macd'] > df['macd_signal']) & (df['macd'].shift(1) <= df['macd_signal'].shift(1))

    signal = pd.Series(False, index=df.index)
    for idx in df.index[df['cci_cross']]:
        pos = df.index.get_loc(idx)
        end = min(pos + 3, len(df))
        if df['macd_cross'].iloc[pos:end].any():
            signal.iloc[pos] = True
    df['signal'] = signal
    return df

# ==================== تحلیل ====================
def analyze_all_signals(df, symbol):
    """
    برای هر سیگنال سه چیز محاسبه می‌کند:
    1. سقف سود (بر اساس high) و کندل آن
    2. کف (بر اساس low) و کندل آن
    3. MAE قبل از سقف (بر اساس low)
    """
    out = []
    for idx in df.index[df['signal']]:
        pos = df.index.get_loc(idx)
        if pos + FORWARD_BARS >= len(df):
            continue

        entry = df['close'].iloc[pos]
        f_close = df['close'].iloc[pos+1 : pos+1+FORWARD_BARS]
        f_high = df['high'].iloc[pos+1 : pos+1+FORWARD_BARS]
        f_low = df['low'].iloc[pos+1 : pos+1+FORWARD_BARS]

        if len(f_close) < FORWARD_BARS:
            continue

        # سقف (بر اساس high)
        peak_pos = int(f_high.values.argmax())
        peak_ret = (f_high.iloc[peak_pos] / entry - 1) * 100
        peak_candle = peak_pos + 1

        # کف کل (بر اساس low)
        trough_pos = int(f_low.values.argmin())
        trough_ret = (f_low.iloc[trough_pos] / entry - 1) * 100
        trough_candle = trough_pos + 1

        # MAE قبل از سقف
        pre_peak_low = f_low.iloc[:peak_pos + 1]
        mae_before_peak = (pre_peak_low.min() / entry - 1) * 100

        # بازدهی کندل ۱۰۰
        final_ret = (f_close.iloc[-1] / entry - 1) * 100

        out.append({
            'symbol': symbol,
            'signal_date': idx,
            'entry': entry,
            'final_return': final_ret,
            'peak_return': peak_ret,
            'peak_candle': peak_candle,
            'trough_return': trough_ret,
            'trough_candle': trough_candle,
            'mae_before_peak': mae_before_peak,
            'win_at_100': final_ret > 0
        })
    return out

def simulate_sl_tp(symbol, df, sl, tp):
    """
    برای یک ترکیب SL/TP خاص، همهٔ سیگنال‌ها را شبیه‌سازی می‌کند.
    خروجی: لیستی از dict شامل exit_reason و return_pct
    """
    out = []
    for idx in df.index[df['signal']]:
        pos = df.index.get_loc(idx)
        if pos + FORWARD_BARS >= len(df):
            continue

        entry = df['close'].iloc[pos]
        f_high = df['high'].iloc[pos+1 : pos+1+FORWARD_BARS]
        f_low = df['low'].iloc[pos+1 : pos+1+FORWARD_BARS]
        f_close = df['close'].iloc[pos+1 : pos+1+FORWARD_BARS]

        if len(f_close) < FORWARD_BARS:
            continue

        sl_price = entry * (1 - sl / 100)
        tp_price = entry * (1 + tp / 100)

        exit_reason = 'timeout'
        exit_return = (f_close.iloc[-1] / entry - 1) * 100

        for j in range(len(f_close)):
            hit_sl = f_low.iloc[j] <= sl_price
            hit_tp = f_high.iloc[j] >= tp_price

            if hit_sl and hit_tp:
                # محافظه‌کارانه: SL اول
                exit_reason = 'sl'
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

        out.append({
            'symbol': symbol,
            'exit_reason': exit_reason,
            'return_pct': exit_return
        })
    return out

# ==================== گزارش‌ها ====================
def report_basic(results_df):
    total = len(results_df)
    winners = results_df[results_df['win_at_100']]
    losers = results_df[~results_df['win_at_100']]

    lines = [
        "📉 *آمار پایه*",
        "",
        f"کل سیگنال: {total}",
        f"سودآور در کندل ۱۰۰: {len(winners)} ({len(winners)/total*100:.1f}%)",
        f"زیان‌ده در کندل ۱۰۰: {len(losers)} ({len(losers)/total*100:.1f}%)",
        "",
        f"میانگین سقف (high): {results_df['peak_return'].mean():.2f}%",
        f"میانه سقف: {results_df['peak_return'].median():.2f}%",
        f"میانگین MAE: {results_df['mae_before_peak'].mean():.2f}%",
        f"میانه MAE: {results_df['mae_before_peak'].median():.2f}%",
    ]
    return "\n".join(lines)

def report_profit_levels(results_df, thresholds):
    """
    تعداد دقیق سیگنال‌هایی که سقفشان به هر سطح رسیده.
    این عدد دقیق است — نه تخمین.
    """
    total = len(results_df)
    lines = [
        "",
        "🎯 *تعداد دقیق سیگنال‌هایی که به هر سطح سود رسیدند*",
        "",
        "| سطح | تعداد | درصد |",
        "|------|-------|------|",
    ]
    for th in thresholds:
        c = (results_df['peak_return'] >= th).sum()
        lines.append(f"| +{th}% | {c} | {c/total*100:.1f}% |")
    return "\n".join(lines)

def report_sl_tp_matrix(all_sim_df):
    """
    ماتریس کامل SL/TP با تفکیک دقیق:
    - tp_hits: تعداد واقعی رسیدن به TP
    - sl_hits: تعداد خوردن SL
    - timeout: تعداد رسیدن به پایان ۱۰۰ کندل
    - avg: بازدهی میانگین
    """
    lines = [
        "",
        "🧮 *ماتریس کامل SL/TP (تفکیک دقیق)*",
        "",
        "برای هر ترکیب:",
        "- `TP` = تعداد واقعی که به هدف سود رسید",
        "- `SL` = تعداد که حد ضرر خورد",
        "- `TO` = تعداد که در کندل ۱۰۰ بسته شد",
        "- `میانگین` = بازدهی میانگین",
        "",
    ]

    grouped = all_sim_df.groupby(['sl', 'tp'])

    # ساخت جدول
    rows = []
    for (sl, tp), group in grouped:
        total = len(group)
        tp_hits = (group['exit_reason'] == 'tp').sum()
        sl_hits = (group['exit_reason'] == 'sl').sum()
        to_hits = (group['exit_reason'] == 'timeout').sum()
        avg = group['return_pct'].mean()
        profitable = (group['return_pct'] > 0).sum()
        rows.append({
            'SL': sl, 'TP': tp, 'total': total,
            'tp_hits': tp_hits, 'sl_hits': sl_hits, 'to_hits': to_hits,
            'avg': avg, 'profitable': profitable,
            'profitable_pct': profitable / total * 100
        })

    df = pd.DataFrame(rows).sort_values('avg', ascending=False)

    lines.append("*۱۰ ترکیب با بالاترین میانگین بازدهی:*")
    lines.append("")
    lines.append("`SL/TP | TP-hit | SL-hit | TO | میانگین | سودآور`")
    for _, r in df.head(10).iterrows():
        lines.append(
            f"`-{int(r['SL'])}/+{int(r['TP'])}` | "
            f"{int(r['tp_hits'])} | {int(r['sl_hits'])} | {int(r['to_hits'])} | "
            f"{r['avg']:.2f}% | {r['profitable_pct']:.1f}%"
        )

    lines.append("")
    lines.append("*۵ ترکیب با بالاترین نرخ TP-hit واقعی:*")
    lines.append("")
    df_by_tp = df.sort_values('tp_hits', ascending=False)
    for _, r in df_by_tp.head(5).iterrows():
        lines.append(
            f"`-{int(r['SL'])}/+{int(r['TP'])}` → "
            f"{int(r['tp_hits'])} TP | {int(r['sl_hits'])} SL | "
            f"میانگین {r['avg']:.2f}%"
        )

    return "\n".join(lines)

def report_mae_groups(results_df):
    lines = [
        "",
        "🔬 *تفکیک بر اساس میزان ریزش (MAE قبل از سقف)*",
        "",
        "فقط سیگنال‌هایی که در کندل ۱۰۰ سودآور بودند.",
        "",
    ]

    winners = results_df[results_df['win_at_100']].copy()
    if len(winners) == 0:
        return "\n".join(lines)

    def bucket(x):
        if x >= -3: return "۰ تا -۳٪"
        if x >= -5: return "-۳٪ تا -۵٪"
        if x >= -10: return "-۵٪ تا -۱۰٪"
        if x >= -15: return "-۱۰٪ تا -۱۵٪"
        return "بیش از -۱۵٪"

    winners['bucket'] = winners['mae_before_peak'].apply(bucket)
    order = ["۰ تا -۳٪", "-۳٪ تا -۵٪", "-۵٪ تا -۱۰٪", "-۱۰٪ تا -۱۵٪", "بیش از -۱۵٪"]

    for b in order:
        sub = winners[winners['bucket'] == b]
        if len(sub) == 0:
            continue
        p = sub['peak_return']
        lines.append(f"▫️ *{b}* — {len(sub)} سیگنال")
        lines.append(f"   کمترین: +{p.min():.1f}%")
        lines.append(f"   میانه: +{p.median():.1f}%")
        lines.append(f"   بیشترین: +{p.max():.1f}%")
        lines.append(f"   تعداد بالای +65٪: {(p >= 65).sum()}")
        lines.append("")

    return "\n".join(lines)

# ==================== اجرا ====================
def main():
    all_signals = []
    all_sims = []

    print("=" * 60)
    print("شروع تحلیل")
    print("=" * 60)

    for symbol in TOP_10_SYMBOLS:
        df = fetch_ohlcv(symbol)
        if df is None or len(df) < 200:
            continue
        df = compute_indicators(df)
        df = detect_signals(df)

        sig_count = int(df['signal'].sum())
        print(f"  {symbol}: {sig_count} سیگنال")
        if sig_count == 0:
            continue

        # تحلیل پایه
        all_signals.extend(analyze_all_signals(df, symbol))

        # شبیه‌سازی همهٔ ترکیب‌های SL/TP
        for sl in SL_LEVELS:
            for tp in TP_LEVELS:
                all_sims.extend(simulate_sl_tp(symbol, df, sl, tp))

        time.sleep(1)

    signals_df = pd.DataFrame(all_signals)
    sims_df = pd.DataFrame(all_sims)

    if len(signals_df) == 0:
        send_telegram_message("هیچ سیگنالی یافت نشد.")
        return

    print(f"\nمجموع سیگنال‌ها: {len(signals_df)}")

    # ==================== پیام تلگرام ====================
    msg = "📊 *تحلیل دقیق CCI>SMA + MACD*\n"
    msg += f"🗓 {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n"
    msg += report_basic(signals_df)
    msg += report_profit_levels(signals_df, PROFIT_THRESHOLDS)
    msg += report_mae_groups(signals_df)
    msg += report_sl_tp_matrix(sims_df)

    send_telegram_message(msg)

    # ذخیره CSV
    signals_df.to_csv('signals.csv', index=False)
    sims_df.to_csv('sims.csv', index=False)

    print("\nپایان.")

if __name__ == '__main__':
    main()
