import time
import requests
import pandas as pd
import numpy as np

# ================== تنظیمات ==================
TOP_N = 20
LOOKAHEAD = 100

TARGETS_UP = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
DRAWDOWN_LEVELS = [2, 3, 4, 5, 6, 7, 8, 9, 10]


# ================== لیست ۲۰ ارز برتر ==================
def get_top_coins():
    url = "https://api.coingecko.com/api/v3/coins/markets"
    params = {
        "vs_currency": "usd",
        "order": "market_cap_desc",
        "per_page": TOP_N,
        "page": 1,
        "sparkline": "false",
    }
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    data = r.json()
    return [(c["symbol"].upper() + "USDT", c["symbol"].upper()) for c in data]


# ================== OHLC ۴ ساعته از Binance ==================
def get_ohlc_4h(symbol, limit=500):
    url = "https://api.binance.com/api/v3/klines"
    params = {"symbol": symbol, "interval": "4h", "limit": limit}
    r = requests.get(url, params=params, timeout=30)
    if r.status_code != 200:
        return None
    data = r.json()
    if not data or len(data) < 50:
        return None
    df = pd.DataFrame(data, columns=[
        "timestamp", "open", "high", "low", "close", "volume",
        "close_time", "quote_volume", "trades",
        "taker_base", "taker_quote", "ignore",
    ])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = df[col].astype(float)
    return df[["timestamp", "open", "high", "low", "close", "volume"]]


# ================== ADX ==================
def calc_adx(df, di_len=14, adx_len=14):
    high, low, close = df["high"], df["low"], df["close"]
    up = high.diff()
    down = -low.diff()
    plus_dm = np.where((up > down) & (up > 0), up, 0.0)
    minus_dm = np.where((down > up) & (down > 0), down, 0.0)
    tr = pd.concat(
        [high - low, (high - close.shift()).abs(), (low - close.shift()).abs()],
        axis=1,
    ).max(axis=1)

    def rma(s, n):
        return s.ewm(alpha=1.0 / n, adjust=False).mean()

    atr = rma(tr, di_len)
    plus_di = 100 * rma(pd.Series(plus_dm, index=df.index), di_len) / atr
    minus_di = 100 * rma(pd.Series(minus_dm, index=df.index), di_len) / atr
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    adx = rma(dx, adx_len)

    df["plus_di"] = plus_di
    df["minus_di"] = minus_di
    df["adx"] = adx
    return df


# ================== Alligator ==================
def calc_alligator(df):
    median = (df["high"] + df["low"]) / 2

    def smma(s, n):
        return s.ewm(alpha=1.0 / n, adjust=False).mean()

    df["alligator_jaw"] = smma(median, 13).shift(8)
    df["alligator_teeth"] = smma(median, 8).shift(5)
    df["alligator_lips"] = smma(median, 5).shift(3)
    return df


# ================== سیگنال ==================
def detect_signals(df, hl_trend=35, hl_range=20):
    df["signal"] = False
    for i in range(1, len(df)):
        adx_ok = (
            df.loc[i, "adx"] > hl_trend
            and df.loc[i, "adx"] > df.loc[i - 1, "adx"]
            and df.loc[i, "plus_di"] > df.loc[i, "minus_di"]
            and df.loc[i, "plus_di"] >= hl_trend
        )
        j, t, l = (
            df.loc[i, "alligator_jaw"],
            df.loc[i, "alligator_teeth"],
            df.loc[i, "alligator_lips"],
        )
        alligator_ok = pd.notna(j) and pd.notna(t) and pd.notna(l) and j > t > l
        range_ok = df.loc[i, "adx"] > hl_range
        df.loc[i, "signal"] = bool(adx_ok and alligator_ok and range_ok)
    return df


# ================== تحلیل ماتریس ==================
def analyze_signals(df, lookahead=LOOKAHEAD):
    signals = df.index[df["signal"]].tolist()
    hit_matrix = {t: {d: 0 for d in DRAWDOWN_LEVELS} for t in TARGETS_UP}
    reached = {t: 0 for t in TARGETS_UP}
    total = 0
    for sig_idx in signals:
        future = df.loc[sig_idx + 1: sig_idx + lookahead]
        if len(future) == 0:
            continue
        total += 1
        entry = df.loc[sig_idx, "close"]
        if entry <= 0:
            continue
        for t in TARGETS_UP:
            target = entry * (1 + t / 100.0)
            mask = (future["high"] >= target).values
            if not mask.any():
                continue
            hit_pos = int(mask.argmax())
            reached[t] += 1
            seg = future.iloc[: hit_pos + 1]
            dd = (entry - seg["low"].min()) / entry * 100
            for d in DRAWDOWN_LEVELS:
                if dd < d:
                    hit_matrix[t][d] += 1
    return {"total_signals": total, "reached": reached, "matrix": hit_matrix}


# ================== جدول متنی ==================
def build_matrix_table(res):
    total = res["total_signals"]
    if total == 0:
        return "  (بدون سیگنال)"
    header = "هدف/ریزش |" + "|".join(f"<{d}٪".rjust(5) for d in DRAWDOWN_LEVELS)
    lines = [header, "-" * len(header)]
    for t in TARGETS_UP:
        row = [f"{t:>3}٪     |"]
        for d in DRAWDOWN_LEVELS:
            pct = res["matrix"][t][d] / total * 100
            row.append(f"{pct:>4.0f}٪")
        lines.append("|".join(row))
    lines.append("")
    lines.append(
        "رسیده فقط: "
        + " | ".join(f"{t}٪→{res['reached'][t] / total * 100:.0f}٪" for t in TARGETS_UP)
    )
    return "\n".join(lines)


# ================== گزارش نهایی ==================
def build_report(coin_results):
    total_all = sum(r["total_signals"] for r in coin_results.values())
    if total_all == 0:
        return "📊 هیچ سیگنالی با شرایط مشخص‌شده در تایم‌فریم ۴ ساعته یافت نشد."

    lines = []
    lines.append("=" * 70)
    lines.append(f"گزارش ADX + Alligator | TF=4h | {len(coin_results)} ارز | سیگنال کل: {total_all}")
    lines.append("=" * 70)
    lines.append("راهنما: هر خانه = درصدی از کل سیگنال‌ها که هم به هدف رسیدند")
    lines.append("و هم قبل از رسیدن، ریزششان زیر حد ستون بود.")
    lines.append("")

    gm = {t: {d: 0 for d in DRAWDOWN_LEVELS} for t in TARGETS_UP}
    gr = {t: 0 for t in TARGETS_UP}

    for sym, res in coin_results.items():
        if res["total_signals"] == 0:
            continue
        lines.append(f"---- {sym} | {res['total_signals']} سیگنال ----")
        lines.append(build_matrix_table(res))
        lines.append("")
        for t in TARGETS_UP:
            gr[t] += res["reached"][t]
            for d in DRAWDOWN_LEVELS:
                gm[t][d] += res["matrix"][t][d]

    lines.append("=" * 70)
    lines.append("خلاصه کل ۲۰ ارز")
    lines.append("=" * 70)
    lines.append(build_matrix_table({
        "total_signals": total_all, "reached": gr, "matrix": gm,
    }))
    return "\n".join(lines)


# ================== main ==================
def main():
    print("دریافت لیست ۲۰ ارز برتر از CoinGecko...")
    coins = get_top_coins()
    print(f"تعداد ارز: {len(coins)}\n")

    empty = lambda: {
        "total_signals": 0,
        "reached": {t: 0 for t in TARGETS_UP},
        "matrix": {t: {d: 0 for d in DRAWDOWN_LEVELS} for t in TARGETS_UP},
    }

    coin_results = {}
    for symbol_binance, display in coins:
        print(f"بررسی {display} ({symbol_binance})...")
        try:
            df = get_ohlc_4h(symbol_binance)
            if df is None or len(df) < 50:
                print(f"  ⚠️ داده کافی برای {display} نبود (یا در Binance نیست).\n")
                coin_results[display] = empty()
                time.sleep(0.3)
                continue
            df = calc_adx(df)
            df = calc_alligator(df)
            df = detect_signals(df)
            res = analyze_signals(df)
            coin_results[display] = res
            print(f"  ✅ سیگنال‌ها: {res['total_signals']}\n")
        except Exception as e:
            print(f"  ❌ خطا برای {display}: {e}\n")
            coin_results[display] = empty()
        time.sleep(0.3)

    report = build_report(coin_results)

    # چاپ گزارش در لاگ Actions
    print("\n\n")
    print(report)

    # ذخیره برای artifact (اختیاری)
    with open("report.txt", "w", encoding="utf-8") as f:
        f.write(report)


if __name__ == "__main__":
    main()
