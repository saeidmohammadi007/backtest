# main.py — نسخهٔ نهایی: پنجره MACD = ۲ کندل، همه اعداد واقعی
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

COMBOS = [
    (5, 10), (5, 15), (5, 20), (5, 30),
    (8, 15), (8, 20), (8, 30),
    (10, 15), (10, 20), (10, 30), (10, 50),
    (15, 15), (15, 20), (15, 30), (15, 50), (15, 75), (15, 100),
    (20, 30), (20, 50), (20, 75), (20, 100),
]

PROFIT_LEVELS = [5, 10, 15, 20, 30, 40, 50, 65, 75, 100, 150, 200, 300]

# ==================== تلگرام ====================
TELEGRAM_BOT_TOKEN = os.environ.get('TELEGRAM_BOT_TOKEN')
TELEGRAM_CHAT_ID = os.environ.get('TELEGRAM_CHAT_ID')

def send_telegram(text):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("[هشدار] توکن یا chat_id تنظیم نشده.")
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    try:
        for i in range(0, len(text), 4000):
            r = requests.post(url, json={
                "chat_id": TELEGRAM_CHAT_ID,
                "text": text[i:i+4000],
                "parse_mode": "Markdown"
            }, timeout=30)
            r.raise_for_status()
        print("  پیام تلگرام ارسال شد.")
    except Exception as e:
        print(f"  خطا: {e}")

# ==================== داده ====================
def fetch(symbol):
    print(f"  دریافت {symbol}...")
    try:
        df = yf.download(symbol, period='max', interval=TIMEFRAME,
                         progress=False, auto_adjust=True, threads=False)
        if df is None or df.empty:
            return None
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df.columns = [c.lower() for c in df.columns]
        return df[['open','high','low','close','volume']].dropna().tail(LIMIT)
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
    ef = close.ewm(span=fast, adjust=False).mean()
    es = close.ewm(span=slow, adjust=False).mean()
    m = ef - es
    return m, m.ewm(span=signal, adjust=False).mean()

def add_indicators(df):
    df = df.copy()
    df['cci'] = calc_cci(df['high'], df['low'], df['close'], CCI_LENGTH)
    df['cci_sma'] = df['cci'].rolling(window=CCI_SMA_LENGTH).mean()
    m, s = calc_macd(df['close'], MACD_FAST, MACD_SLOW, MACD_SIGNAL)
    df['macd'], df['macd_signal'] = m, s
    return df

# ==================== سیگنال ====================
def add_signal(df):
    df = df.copy()
    cci_up = (df['cci'] > df['cci_sma']) & (df['cci'].shift(1) <= df['cci_sma'].shift(1))
    macd_up = (df['macd'] > df['macd_signal']) & (df['macd'].shift(1) <= df['macd_signal'].shift(1))

    sig = pd.Series(False, index=df.index)
    for idx in df.index[cci_up]:
        pos = df.index.get_loc(idx)
        # پنجره ۲ کندل: کندل فعلی + ۲ کندل بعد
        end = min(pos + 3, len(df))
        if macd_up.iloc[pos:end].any():
            sig.iloc[pos] = True
    df['signal'] = sig
    return df

# ==================== شبیه‌سازی ====================
def simulate_combo(symbols_data, sl, tp):
    tp_hits = 0
    sl_hits = 0
    timeouts = 0
    returns = []

    for symbol, df in symbols_data.items():
        for idx in df.index[df['signal']]:
            pos = df.index.get_loc(idx)
            if pos + FORWARD_BARS >= len(df):
                continue
            entry = df['close'].iloc[pos]
            highs = df['high'].iloc[pos+1 : pos+1+FORWARD_BARS]
            lows  = df['low'].iloc[pos+1 : pos+1+FORWARD_BARS]
            closes = df['close'].iloc[pos+1 : pos+1+FORWARD_BARS]
            if len(closes) < FORWARD_BARS:
                continue

            sl_price = entry * (1 - sl/100)
            tp_price = entry * (1 + tp/100)
            exit_type = 'timeout'
            exit_ret = (closes.iloc[-1] / entry - 1) * 100

            for j in range(len(closes)):
                hit_sl = lows.iloc[j] <= sl_price
                hit_tp = highs.iloc[j] >= tp_price
                if hit_sl and hit_tp:
                    exit_type = 'sl'; exit_ret = -sl; break
                elif hit_sl:
                    exit_type = 'sl'; exit_ret = -sl; break
                elif hit_tp:
                    exit_type = 'tp'; exit_ret = tp; break

            if exit_type == 'tp': tp_hits += 1
            elif exit_type == 'sl': sl_hits += 1
            else: timeouts += 1
            returns.append(exit_ret)

    return {
        'sl': sl, 'tp': tp,
        'tp_hits': tp_hits, 'sl_hits': sl_hits, 'timeouts': timeouts,
        'avg_return': np.mean(returns) if returns else 0,
        'net_return': np.sum(returns),
        'total': len(returns)
    }

# ==================== گزارش‌ها ====================
def report_profit_levels(signals_df, total):
    lines = [
        "🎯 *تعداد واقعی رسیدن به هر سطح سود*",
        "",
        "| سطح | تعداد | درصد |",
        "|------|-------|------|",
    ]
    for lvl in PROFIT_LEVELS:
        c = (signals_df['peak_high_pct'] >= lvl).sum()
        lines.append(f"| +{lvl}% | {c} | {c/total*100:.1f}% |")
    return "\n".join(lines)

def report_combos(combo_data):
    lines = [
        "🧮 *گزارش کامل ترکیب‌های SL/TP*",
        "",
        "`TP` = رسیدن واقعی به هدف سود",
        "`SL` = خوردن حد ضرر",
        "`TO` = بسته در کندل ۱۰۰",
        "`وین‌ریت` = TP ÷ کل سیگنال‌ها",
        "`بازدهی` = میانگین همهٔ معاملات",
        "`سود خالص` = مجموع بازدهی",
        "",
        "*همه ترکیب‌ها (مرتب بر اساس بازدهی):*",
        "",
        "`SL/TP | TP | SL | TO | وین‌ریت | بازدهی | سود خالص`",
    ]
    combo_data.sort(key=lambda x: x['avg_return'], reverse=True)
    for c in combo_data:
        wr = c['tp_hits'] / c['total'] * 100 if c['total'] else 0
        lines.append(
            f"`-{c['sl']}/+{c['tp']}` | "
            f"{c['tp_hits']} | {c['sl_hits']} | {c['timeouts']} | "
            f"{wr:.1f}% | "
            f"{c['avg_return']:.2f}% | "
            f"{c['net_return']:.0f}%"
        )
    return "\n".join(lines)

def report_best(combo_data):
    lines = ["", "🥇 *بهترین ترکیب‌ها:*", ""]

    b = max(combo_data, key=lambda x: x['avg_return'])
    wr = b['tp_hits'] / b['total'] * 100 if b['total'] else 0
    lines.append(f"▪️ بیشترین بازدهی: `-{b['sl']}/+{b['tp']}`")
    lines.append(f"   بازدهی: {b['avg_return']:.2f}% | وین‌ریت: {wr:.1f}%")
    lines.append(f"   TP: {b['tp_hits']} | SL: {b['sl_hits']} | TO: {b['timeouts']}")
    lines.append("")

    b2 = max(combo_data, key=lambda x: x['tp_hits'] / x['total'] if x['total'] else 0)
    wr2 = b2['tp_hits'] / b2['total'] * 100 if b2['total'] else 0
    lines.append(f"▪️ بالاترین وین‌ریت: `-{b2['sl']}/+{b2['tp']}`")
    lines.append(f"   وین‌ریت: {wr2:.1f}% | بازدهی: {b2['avg_return']:.2f}%")
    lines.append(f"   TP: {b2['tp_hits']} | SL: {b2['sl_hits']} | TO: {b2['timeouts']}")
    lines.append("")

    b3 = max(combo_data, key=lambda x: x['net_return'])
    wr3 = b3['tp_hits'] / b3['total'] * 100 if b3['total'] else 0
    lines.append(f"▪️ بیشترین سود خالص: `-{b3['sl']}/+{b3['tp']}`")
    lines.append(f"   سود خالص: {b3['net_return']:.0f}% | بازدهی: {b3['avg_return']:.2f}%")
    lines.append(f"   وین‌ریت: {wr3:.1f}%")

    return "\n".join(lines)

# ==================== اجرا ====================
def main():
    all_signals = []
    symbols_data = {}

    print("=" * 60)
    print("شروع تحلیل — پنجره MACD = ۲ کندل")
    print("=" * 60)

    for symbol in TOP_10_SYMBOLS:
        df = fetch(symbol)
        if df is None or len(df) < 200:
            continue
        df = add_indicators(df)
        df = add_signal(df)
        df.index.name = 'date'

        cnt = int(df['signal'].sum())
        print(f"  {symbol}: {cnt} سیگنال")
        if cnt == 0:
            continue

        symbols_data[symbol] = df

        for idx in df.index[df['signal']]:
            pos = df.index.get_loc(idx)
            if pos + FORWARD_BARS >= len(df):
                continue
            entry = df['close'].iloc[pos]
            highs = df['high'].iloc[pos+1 : pos+1+FORWARD_BARS]
            closes = df['close'].iloc[pos+1 : pos+1+FORWARD_BARS]
            if len(closes) < FORWARD_BARS:
                continue
            peak_pos = int(highs.values.argmax())
            all_signals.append({
                'symbol': symbol,
                'date': idx,
                'entry': entry,
                'peak_high_pct': (highs.iloc[peak_pos] / entry - 1) * 100,
                'peak_candle': peak_pos + 1,
                'final_return': (closes.iloc[-1] / entry - 1) * 100,
            })

        time.sleep(1)

    signals_df = pd.DataFrame(all_signals)
    total = len(signals_df)
    print(f"\nمجموع سیگنال: {total}")

    if total == 0:
        send_telegram("هیچ سیگنالی یافت نشد.")
        return

    print("\nشبیه‌سازی ترکیب‌ها...")
    combo_data = []
    for sl, tp in COMBOS:
        r = simulate_combo(symbols_data, sl, tp)
        combo_data.append(r)
        print(f"  -{sl}/+{tp}: TP={r['tp_hits']} SL={r['sl_hits']} "
              f"TO={r['timeouts']} | بازدهی {r['avg_return']:.2f}%")

    msg = "📊 *تحلیل نهایی CCI>SMA + MACD*\n"
    msg += f"🗓 {datetime.now().strftime('%Y-%m-%d %H:%M')}\n"
    msg += f"پنجره MACD: *۲ کندل*\n\n"
    msg += f"کل سیگنال: *{total}*\n\n"
    msg += report_profit_levels(signals_df, total)
    msg += "\n\n"
    msg += report_combos(combo_data)
    msg += "\n"
    msg += report_best(combo_data)

    send_telegram(msg)

    signals_df.to_csv('signals.csv', index=False)
    pd.DataFrame(combo_data).to_csv('combos.csv', index=False)
    print("\nپایان.")

if __name__ == '__main__':
    main()
