import ccxt
import numpy as np
import pandas as pd
import time
from datetime import datetime, timezone
from collections import defaultdict

# ========== پیکربندی ==========
TIMEFRAME = '4h'
LIMIT = 1000  # تعداد کندل برای بک‌تست (بیشتر = دقت بالاتر)
SYMBOL_LIMIT = 100  # تعداد ارز برای بک‌تست

# پارامترهای ADX (مطابق کد اصلی)
DI_LEN = 14
ADX_LEN = 14
HL_RANGE = 20
HL_TREND = 35

# پارامترهای الیگیتور (مطابق کد اصلی)
JAW_P, JAW_S = 13, 8
TEETH_P, TEETH_S = 8, 5
LIPS_P, LIPS_S = 5, 3

# پارامترهای بک‌تست
TARGET_PCT = 0.10        # هدف ۱۰ درصد
MAX_DRAWDOWN_PCT = 0.04  # حداکثر ریزش مجاز ۴ درصد
MAX_HOLD_CANDLES = 100   # بررسی تا ۱۰۰ کندل بعد (≈ ۱۷ روز در تایم‌فریم ۴ ساعته)

# ========== توابع کمکی ==========
def smma(arr, length):
    result = np.full_like(arr, np.nan, dtype=float)
    if len(arr) < length:
        return result
    result[length - 1] = np.mean(arr[:length])
    for i in range(length, len(arr)):
        result[i] = (result[i - 1] * (length - 1) + arr[i]) / length
    return result

def rma(arr, length):
    return smma(arr, length)

# ========== محاسبه ADX ==========
def calc_adx_arrays(ohlcv):
    high = np.array([c[2] for c in ohlcv], dtype=float)
    low = np.array([c[3] for c in ohlcv], dtype=float)
    close = np.array([c[4] for c in ohlcv], dtype=float)

    up = np.diff(high, prepend=high[0])
    down = -np.diff(low, prepend=low[0])
    plus_dm = np.where((up > down) & (up > 0), up, 0.0)
    minus_dm = np.where((down > up) & (down > 0), down, 0.0)
    plus_dm[0] = 0.0
    minus_dm[0] = 0.0

    tr = np.maximum.reduce([
        high - low,
        np.abs(high - np.roll(close, 1)),
        np.abs(low - np.roll(close, 1))
    ])
    tr[0] = high[0] - low[0]

    tr_rma = rma(tr, DI_LEN)
    plus_rma = rma(plus_dm, DI_LEN)
    minus_rma = rma(minus_dm, DI_LEN)

    with np.errstate(divide='ignore', invalid='ignore'):
        di_plus = 100.0 * plus_rma / tr_rma
        di_minus = 100.0 * minus_rma / tr_rma
        di_plus = np.nan_to_num(di_plus, nan=0.0)
        di_minus = np.nan_to_num(di_minus, nan=0.0)

    di_sum = di_plus + di_minus
    di_sum[di_sum == 0] = 1.0
    dx = 100.0 * np.abs(di_plus - di_minus) / di_sum
    adx = rma(dx, ADX_LEN)
    return adx, di_plus, di_minus

# ========== محاسبه الیگیتور ==========
def calc_alligator_arrays(ohlcv):
    high = np.array([c[2] for c in ohlcv], dtype=float)
    low = np.array([c[3] for c in ohlcv], dtype=float)
    median = (high + low) / 2.0

    jaw_raw = smma(median, JAW_P)
    teeth_raw = smma(median, TEETH_P)
    lips_raw = smma(median, LIPS_P)

    def shift_forward(arr, shift):
        out = np.full_like(arr, np.nan, dtype=float)
        if shift < len(arr):
            out[shift:] = arr[:-shift]
        return out

    jaw = shift_forward(jaw_raw, JAW_S)
    teeth = shift_forward(teeth_raw, TEETH_S)
    lips = shift_forward(lips_raw, LIPS_S)
    return jaw, teeth, lips

# ========== بررسی شرط سیگنال ==========
def check_signal_condition(adx, di_plus, di_minus, jaw, teeth, lips, idx):
    """بررسی معتبر بودن سیگنال در ایندکس idx (ADX + الیگیتور)"""
    if idx < 1 or idx >= len(adx):
        return False

    # شرط ADX (مطابق تابع candle_ok در کد اصلی)
    if np.isnan(adx[idx]) or np.isnan(di_plus[idx]) or np.isnan(di_minus[idx]):
        return False
    if np.isnan(adx[idx - 1]):
        return False

    sig = adx[idx]
    sig_prev = adx[idx - 1]
    di_up = di_plus[idx] >= di_minus[idx]
    di_up_up = di_plus[idx] >= HL_TREND
    hl_range = sig <= HL_RANGE
    sig_up = sig > sig_prev

    adx_ok = (not hl_range) and di_up and sig_up and di_up_up
    if not adx_ok:
        return False

    # شرط الیگیتور: چیدمان آبی (Jaw) > سبز (Lips) > قرمز (Teeth)
    if np.isnan(jaw[idx]) or np.isnan(teeth[idx]) or np.isnan(lips[idx]):
        return False
    alligator_ok = bool(jaw[idx] > lips[idx] > teeth[idx])

    return alligator_ok

# ========== بک‌تست یک ارز ==========
def backtest_symbol(ohlcv, symbol):
    """بک‌تست سیگنال‌ها روی یک ارز و برگرداندن لیست نتایج"""
    if len(ohlcv) < 100:
        return []

    adx, di_plus, di_minus = calc_adx_arrays(ohlcv)
    jaw, teeth, lips = calc_alligator_arrays(ohlcv)

    results = []

    # از کندل 50 به بعد بررسی می‌کنیم تا اندیکاتورها warmup شده باشند
    for entry_idx in range(50, len(ohlcv) - 1):
        if not check_signal_condition(adx, di_plus, di_minus, jaw, teeth, lips, entry_idx):
            continue

        # رد کردن سیگنال‌هایی که داده کافی برای بررسی کامل ۱۰۰ کندل بعدی ندارند
        if entry_idx + 1 + MAX_HOLD_CANDLES > len(ohlcv):
            continue

        # قیمت ورود: open کندل بعدی (سیگنال در close کندل جاری صادر می‌شود)
        entry_price = ohlcv[entry_idx + 1][1]
        if entry_price <= 0:
            continue

        target_price = entry_price * (1 + TARGET_PCT)
        stop_price = entry_price * (1 - MAX_DRAWDOWN_PCT)

        success = False
        max_favorable = 0.0
        max_adverse = 0.0
        exit_idx = entry_idx + 1
        exit_reason = 'timeout'

        for i in range(entry_idx + 1, entry_idx + 1 + MAX_HOLD_CANDLES):
            high = ohlcv[i][2]
            low = ohlcv[i][3]

            favorable = (high - entry_price) / entry_price
            adverse = (low - entry_price) / entry_price
            max_favorable = max(max_favorable, favorable)
            max_adverse = min(max_adverse, adverse)

            # اولویت با حد ضرر (محافظه‌کارانه)
            if low <= stop_price:
                exit_idx = i
                success = False
                exit_reason = 'stop_loss'
                break

            if high >= target_price:
                exit_idx = i
                success = True
                exit_reason = 'target'
                break

        signal_time = datetime.fromtimestamp(ohlcv[entry_idx][0] / 1000, tz=timezone.utc)

        results.append({
            'symbol': symbol,
            'signal_time': signal_time,
            'entry_price': entry_price,
            'success': success,
            'exit_reason': exit_reason,
            'max_favorable': max_favorable,
            'max_adverse': max_adverse,
            'hold_candles': exit_idx - entry_idx
        })

    return results

# ========== اجرای اصلی ==========
def run_backtest():
    print("=" * 70)
    print("بک‌تست سیگنال الیگیتور + ADX در تایم‌فریم ۴ ساعته")
    print(f"هدف: {TARGET_PCT*100:.0f}% | حداکثر ریزش مجاز: {MAX_DRAWDOWN_PCT*100:.0f}%")
    print(f"بررسی تا {MAX_HOLD_CANDLES} کندل بعد (≈ {MAX_HOLD_CANDLES*4/24:.0f} روز)")
    print(f"داده تاریخی: {LIMIT} کندل (≈ {LIMIT*4/24:.0f} روز)")
    print("=" * 70)

    exchange = ccxt.binanceusdm({'enableRateLimit': True})
    exchange.load_markets()

    symbols = [
        sym for sym, m in exchange.markets.items()
        if m.get('swap') and m.get('linear') and m.get('quote') == 'USDT'
        and m.get('active', True)
    ][:SYMBOL_LIMIT]

    print(f"تعداد ارز برای بک‌تست: {len(symbols)}")
    print()

    all_results = []
    failed_symbols = []

    for i, symbol in enumerate(symbols, 1):
        try:
            print(f"\r[{i}/{len(symbols)}] دریافت داده {symbol}...", end='', flush=True)
            ohlcv = exchange.fetch_ohlcv(symbol, TIMEFRAME, limit=LIMIT)

            if len(ohlcv) < 100:
                failed_symbols.append(symbol)
                continue

            results = backtest_symbol(ohlcv, symbol)
            all_results.extend(results)

            time.sleep(exchange.rateLimit / 1000)

        except Exception as e:
            failed_symbols.append(symbol)

    print("\n")

    if not all_results:
        print("❌ هیچ سیگنالی در داده‌های تاریخی پیدا نشد.")
        return

    df = pd.DataFrame(all_results)

    # ========== آمار کلی ==========
    total_signals = len(df)
    successful = int(df['success'].sum())
    failed = total_signals - successful
    success_rate = (successful / total_signals) * 100

    target_hits = int((df['exit_reason'] == 'target').sum())
    stop_hits = int((df['exit_reason'] == 'stop_loss').sum())
    timeouts = int((df['exit_reason'] == 'timeout').sum())

    print("=" * 70)
    print("📊 نتایج کلی بک‌تست")
    print("=" * 70)
    print(f"تعداد کل سیگنال‌ها: {total_signals}")
    print(f"سیگنال‌های موفق (رسیدن به هدف {TARGET_PCT*100:.0f}% بدون ریزش >{MAX_DRAWDOWN_PCT*100:.0f}%): {successful}")
    print(f"سیگنال‌های ناموفق: {failed}")
    print(f"نرخ موفقیت: {success_rate:.2f}%")
    print()
    print(f"برخورد با هدف (۱۰٪): {target_hits} ({target_hits/total_signals*100:.1f}%)")
    print(f"برخورد با حد ضرر (۴٪): {stop_hits} ({stop_hits/total_signals*100:.1f}%)")
    print(f"تایم‌اوت (بدون رسیدن به هدف/ضرر): {timeouts} ({timeouts/total_signals*100:.1f}%)")
    print()

    # ========== توزیع ساعات موفق ==========
    successful_df = df[df['success'] == True].copy()
    if len(successful_df) > 0:
        successful_df['hour_utc'] = successful_df['signal_time'].dt.hour
        hour_counts = successful_df['hour_utc'].value_counts().sort_index()

        print("=" * 70)
        print("⏰ توزیع ساعات UTC سیگنال‌های موفق")
        print("=" * 70)
        print(f"{'ساعت (UTC)':<15} {'تعداد موفق':<15} {'درصد از کل موفق':<20}")
        print("-" * 50)
        for hour in range(24):
            count = int(hour_counts.get(hour, 0))
            if count > 0:
                pct = (count / successful) * 100
                bar = '█' * int(pct / 2)
                print(f"{hour:02d}:00{'':<9} {count:<15} {pct:>5.1f}%  {bar}")
        print()

        # بهترین ساعت
        best_hour = int(hour_counts.idxmax())
        best_count = int(hour_counts.max())
        print(f"🏆 بهترین ساعت UTC: {best_hour:02d}:00 با {best_count} سیگنال موفق")
        print()

    # ========== توزیع ساعات همه سیگنال‌ها (برای مقایسه) ==========
    df['hour_utc'] = df['signal_time'].dt.hour
    all_hour_counts = df['hour_utc'].value_counts().sort_index()

    print("=" * 70)
    print("📅 توزیع ساعات UTC همه سیگنال‌ها (موفق + ناموفق)")
    print("=" * 70)
    print(f"{'ساعت (UTC)':<15} {'کل سیگنال':<15} {'موفق':<10} {'نرخ موفقیت':<15}")
    print("-" * 60)
    for hour in range(24):
        total_h = int(all_hour_counts.get(hour, 0))
        if total_h > 0:
            succ_h = int(hour_counts.get(hour, 0)) if len(successful_df) > 0 else 0
            rate_h = (succ_h / total_h) * 100 if total_h > 0 else 0
            print(f"{hour:02d}:00{'':<9} {total_h:<15} {succ_h:<10} {rate_h:>5.1f}%")
    print()

    # ========== آمار حداکثر سود/ضرر ==========
    print("=" * 70)
    print("📈 آمار حرکت قیمت پس از سیگنال")
    print("=" * 70)
    print(f"میانگین حداکثر سود شناور: {df['max_favorable'].mean()*100:.2f}%")
    print(f"میانگین حداکثر ضرر شناور: {df['max_adverse'].mean()*100:.2f}%")
    print(f"میانگین مدت نگهداری (کندل): {df['hold_candles'].mean():.1f} "
          f"(≈ {df['hold_candles'].mean()*4/24:.1f} روز)")
    print()

    # ========== بهترین نمادها ==========
    print("=" * 70)
    print("🏅 ۱۰ نماد با بیشترین سیگنال موفق")
    print("=" * 70)
    sym_stats = df.groupby('symbol').agg(
        total=('success', 'size'),
        wins=('success', 'sum')
    )
    sym_stats['rate'] = (sym_stats['wins'] / sym_stats['total'] * 100).round(1)
    top_syms = sym_stats.sort_values(['wins', 'rate'], ascending=False).head(10)
    print(f"{'نماد':<22} {'کل':<8} {'موفق':<8} {'نرخ':<10}")
    print("-" * 50)
    for sym, row in top_syms.iterrows():
        print(f"{sym:<22} {int(row['total']):<8} {int(row['wins']):<8} {row['rate']:>5.1f}%")
    print()

    # ========== گزارش نمادهای خطادار ==========
    if failed_symbols:
        print(f"⚠️ {len(failed_symbols)} نماد بررسی نشدند (خطا یا داده ناکافی).")
        print(f"   نمونه: {', '.join(failed_symbols[:5])}...")

if __name__ == '__main__':
    run_backtest()
