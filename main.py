import requests
import pandas as pd
import numpy as np
import time

# ================= تنظیمات =================
BASE_URL = "https://api.lbkex.com"
TIMEFRAME = "day1"
KLINE_SIZE = 1000
TOP_N = 100
ADX_LEN = 14
DI_LEN = 14
HL_RANGE = 20
HL_TREND = 35
FORWARD_BARS = 100
MIN_BARS = 150
DRAWDOWN_LEVELS = [0.02, 0.03, 0.04, 0.05]
TARGET_LEVELS = [0.10, 0.20, 0.40, 0.80, 1.00]


# ================= توابع کمکی =================
def rma(series, length):
    alpha = 1.0 / length
    return series.ewm(alpha=alpha, adjust=False, min_periods=length).mean()


def fetch_all_usdt_pairs():
    """دریافت همه جفت‌ارزهای USDT از LBank"""
    url = f"{BASE_URL}/v1/currencyPairs.do"
    resp = requests.get(url, timeout=15)
    data = resp.json()

    if isinstance(data, list):
        pairs = [p for p in data if isinstance(p, str) and p.endswith("_usdt")]
    elif isinstance(data, dict):
        raw = data.get("data", data.get("result", []))
        if isinstance(raw, list):
            pairs = [p for p in raw if isinstance(p, str) and p.endswith("_usdt")]
        else:
            raise Exception(f"خروجی جفت‌ارزها قابل تفسیر نیست: {data}")
    else:
        raise Exception(f"نوع خروجی ناشناخته: {type(data)}")

    print(f"تعداد جفت‌ارزهای USDT: {len(pairs)}")
    return pairs


def fetch_tickers(pairs):
    """دریافت تیکر ۲۴ ساعته برای مرتب‌سازی بر اساس حجم معاملات"""
    url = f"{BASE_URL}/v2/ticker/24hr.do"
    params = {"symbol": "all"}
    resp = requests.get(url, params=params, timeout=15)
    data = resp.json()
    if data.get("result") != "true":
        raise Exception(f"دریافت تیکر ناموفق بود: {data}")

    ticker_map = {}
    for item in data["data"]:
        sym = item.get("symbol", "").lower()
        if sym.endswith("_usdt"):
            ticker_map[sym] = {
                "turnover": float(item.get("turnover", 0)),
                "vol": float(item.get("vol", 0)),
            }

    pairs_set = set(pairs)
    ranked = sorted(
        [(sym, info) for sym, info in ticker_map.items() if sym in pairs_set],
        key=lambda x: x[1]["turnover"],
        reverse=True,
    )
    return [r[0] for r in ranked[:TOP_N]]


def fetch_klines(symbol, timeframe="day1", size=KLINE_SIZE):
    """دریافت کندل‌ها (پارامتر time = زمان شروع)"""
    url = f"{BASE_URL}/v2/kline.do"
    step = 86400  # day1
    params = {
        "symbol": symbol,
        "size": size,
        "type": timeframe,
        "time": int(time.time()) - size * step,
    }
    resp = requests.get(url, params=params, timeout=20)
    data = resp.json()
    if data.get("result") != "true":
        print(f"  [هشدار] دریافت کندل {symbol} ناموفق بود: {data.get('msg', data)}")
        return None

    rows = data["data"]
    if not rows:
        return None

    df = pd.DataFrame(rows).iloc[:, :6]
    df.columns = ["timestamp", "open", "high", "low", "close", "volume"]
    ts = df["timestamp"].astype(float)
    unit = "ms" if ts.iloc[0] > 1e11 else "s"
    df["timestamp"] = pd.to_datetime(ts, unit=unit, utc=True)
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna().sort_values("timestamp").reset_index(drop=True)
    return df.iloc[:-1].reset_index(drop=True)  # حذف کندل ناقص آخر


# ================= منطق ADX =================
def compute_adx_gu5(df, di_len=14, sig_len=14):
    high = df["high"]
    low = df["low"]
    close = df["close"]

    up = high.diff()
    down = -low.diff()

    plus_dm = np.where((up > down) & (up > 0), up, 0.0)
    minus_dm = np.where((down > up) & (down > 0), down, 0.0)

    plus_dm = pd.Series(plus_dm, index=df.index)
    minus_dm = pd.Series(minus_dm, index=df.index)
    plus_dm.iloc[0] = np.nan
    minus_dm.iloc[0] = np.nan

    tr = pd.concat(
        [high - low, (high - close.shift()).abs(), (low - close.shift()).abs()], axis=1
    ).max(axis=1)
    true_range = rma(tr, di_len)

    plus = 100 * rma(plus_dm, di_len) / true_range
    minus = 100 * rma(minus_dm, di_len) / true_range
    plus = plus.ffill()
    minus = minus.ffill()

    sum_dm = plus + minus
    sum_dm = sum_dm.replace(0, 1)
    sig = 100 * rma((plus - minus).abs() / sum_dm, sig_len)
    sig = sig.ffill()

    df["sig"] = sig
    df["diPlus"] = plus
    df["diMinus"] = minus

    hlRange = sig <= HL_RANGE
    diUp = plus >= minus
    diUpUp = plus >= HL_TREND
    diDn = minus > plus
    diDnDn = minus > HL_TREND
    crossDi = ((plus > minus) & (plus.shift(1) <= minus.shift(1))) | (
        (plus < minus) & (plus.shift(1) >= minus.shift(1))
    )
    sigUp = sig > sig.shift(1)

    condition = np.zeros(len(df))
    condition[0] = 0.0

    for i in range(1, len(df)):
        prev = condition[i - 1]
        if not hlRange.iloc[i] and diUp.iloc[i] and sigUp.iloc[i] and diUpUp.iloc[i]:
            condition[i] = 1
        elif not hlRange.iloc[i] and diDn.iloc[i] and sigUp.iloc[i] and diDnDn.iloc[i]:
            condition[i] = -1
        elif not hlRange.iloc[i] and diUp.iloc[i] and sigUp.iloc[i] and not diUp.iloc[i - 1]:
            condition[i] = 0.5
        elif not hlRange.iloc[i] and diDn.iloc[i] and sigUp.iloc[i] and not diDn.iloc[i - 1]:
            condition[i] = -0.5
        elif (crossDi.iloc[i] and diUp.iloc[i - 1]) or (hlRange.iloc[i] and not hlRange.iloc[i - 1]):
            condition[i] = 0
        elif (crossDi.iloc[i] and diDn.iloc[i - 1]) or (hlRange.iloc[i] and not hlRange.iloc[i - 1]):
            condition[i] = 0
        else:
            condition[i] = prev

    df["condition"] = condition
    df["hlRange"] = hlRange.values

    df["longX"] = (df["condition"].shift(1).isin([0.5, 1.0])) & (df["condition"] == 0)
    df["shortX"] = (df["condition"].shift(1).isin([-0.5, -1.0])) & (df["condition"] == 0)
    df["longEStr"] = (df["condition"].shift(1) != 1) & (df["condition"] == 1)
    df["shortEStr"] = (df["condition"].shift(1) != -1) & (df["condition"] == -1)

    return df


# ================= تشخیص الگو =================
def detect_pattern(df):
    """× ← ناحیه زرد ← مثلث سبز بزرگ"""
    signal_cols = ["longX", "shortX", "longEStr", "shortEStr"]
    yellow = df["hlRange"]

    matches = []
    for idx in df.index[df["longEStr"]]:
        found = False
        for back in range(10, 61):
            prev_idx = idx - back
            if prev_idx < 0:
                break
            if df.loc[prev_idx, "longX"] or df.loc[prev_idx, "shortX"]:
                between_signals = df.loc[prev_idx + 1:idx - 1, signal_cols].any(axis=1)
                if between_signals.any():
                    break
                between_yellow = yellow.loc[prev_idx + 1:idx - 1]
                if not between_yellow.any():
                    break
                found = True
                break
        if found:
            matches.append(idx)
    return matches


def evaluate_forward(df, entry_idx, drawdown_levels, target_levels, forward_bars=100):
    entry_price = df.loc[entry_idx, "close"]
    end_idx = min(entry_idx + forward_bars, len(df) - 1)
    future_high = df.loc[entry_idx + 1:end_idx, "high"].values
    future_low = df.loc[entry_idx + 1:end_idx, "low"].values

    if len(future_high) == 0:
        return None

    results = {}
    for dd in drawdown_levels:
        dd_price = entry_price * (1 - dd)
        touch_indices = np.where(future_low <= dd_price)[0]
        if len(touch_indices) == 0:
            results[dd] = {t: False for t in target_levels}
            results[dd]["_touched"] = False
            continue

        touch_idx = touch_indices[0]
        results[dd] = {"_touched": True}
        remaining_high = future_high[touch_idx:]
        for t in target_levels:
            target_price = entry_price * (1 + t)
            results[dd][t] = bool(np.any(remaining_high >= target_price))
    return results


# ================= اجرای اصلی =================
def main():
    print("=" * 60)
    print("اسکریپت آمار الگوی ADX در LBank")
    print(f"تایم‌فریم: {TIMEFRAME} | {TOP_N} جفت‌ارز USDT برتر")
    print("شرط: × ← ناحیه زرد ← مثلث سبز بزرگ")
    print("=" * 60)

    all_pairs = fetch_all_usdt_pairs()
    time.sleep(0.5)

    top_pairs = fetch_tickers(all_pairs)
    print(f"{len(top_pairs)} جفت‌ارز برتر از نظر حجم: {top_pairs[:5]} ...")
    time.sleep(0.5)

    all_matches = []

    for i, symbol in enumerate(top_pairs):
        print(f"[{i + 1}/{len(top_pairs)}] در حال پردازش {symbol} ...", end=" ")

        df = fetch_klines(symbol, TIMEFRAME, KLINE_SIZE)
        if df is None or len(df) < MIN_BARS:
            print("داده کافی نیست، رد شد")
            time.sleep(0.2)
            continue

        try:
            df = compute_adx_gu5(df, DI_LEN, ADX_LEN)
        except Exception as e:
            print(f"خطا در محاسبه ADX: {e}")
            time.sleep(0.2)
            continue

        matches = detect_pattern(df)

        if not matches:
            print(f"الگویی پیدا نشد ({len(df)} کندل)")
            time.sleep(0.2)
            continue

        print(f"{len(matches)} الگو پیدا شد")

        for idx in matches:
            res = evaluate_forward(df, idx, DRAWDOWN_LEVELS, TARGET_LEVELS, FORWARD_BARS)
            if res:
                all_matches.append({
                    "symbol": symbol,
                    "timestamp": df.loc[idx, "timestamp"],
                    "entry_price": df.loc[idx, "close"],
                    "results": res,
                })
        time.sleep(0.3)

    print("\n" + "=" * 60)
    print("نتایج آماری")
    print("=" * 60)

    if not all_matches:
        print("هیچ الگوی منطبقی پیدا نشد.")
        return

    print(f"تعداد کل الگوهای پیدا شده: {len(all_matches)}")
    print("هر ستون: درصد موفقیت (تعداد موفق / تعداد دفعاتی که افت لمس شد)\n")

    header = f"{'افت':<8}" + "".join([f"{'← ' + str(int(t * 100)) + '%':<16}" for t in TARGET_LEVELS])
    print(header)
    print("-" * 90)

    for dd in DRAWDOWN_LEVELS:
        row = f"{int(dd * 100)}%{'':<5}"
        touched_count = sum(1 for m in all_matches if m["results"].get(dd, {}).get("_touched"))
        for t in TARGET_LEVELS:
            hit_count = 0
            for m in all_matches:
                r = m["results"].get(dd)
                if r and r.get("_touched") and r.get(t):
                    hit_count += 1
            pct = (hit_count / touched_count * 100) if touched_count > 0 else 0
            row += f"{pct:.1f}%({hit_count}/{touched_count})".ljust(18)
        print(row)

    print(f"\nتعداد کل الگوها: {len(all_matches)}")
    for dd in DRAWDOWN_LEVELS:
        touched = sum(1 for m in all_matches if m["results"].get(dd, {}).get("_touched"))
        print(f"  افت {int(dd * 100)}% در {touched} مورد از {len(all_matches)} لمس شد")

    print("\nجزئیات نمونه‌ها (۲۰ مورد اول):")
    for m in all_matches[:20]:
        print(f"  {m['symbol']} | تاریخ: {m['timestamp']} | قیمت ورود: {m['entry_price']}")
        for dd in DRAWDOWN_LEVELS:
            r = m["results"].get(dd, {})
            if r.get("_touched"):
                hits = [str(int(t * 100)) + "%" for t in TARGET_LEVELS if r.get(t)]
                print(f"    افت {int(dd * 100)}% ← اهداف رسیده: {hits if hits else 'هیچ'}")
            else:
                print(f"    افت {int(dd * 100)}% ← لمس نشد")


if __name__ == "__main__":
    main()
