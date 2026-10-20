import ccxt
import pandas as pd
import numpy as np
import requests

# ================== تنظیمات ==================
TOP_N = 20
LOOKAHEAD = 100
TARGETS_UP = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
DRAWDOWN_LEVELS = [2, 3, 4, 5, 6, 7, 8, 9, 10]
BULLISH_ALLIGATOR = True      # True: lips > teeth > jaw (صعودی)
ONLY_NEW_SIGNALS = True       # فقط اولین کندلِ شروع سیگنال
HL_TREND = 35
HL_RANGE = 20

STABLECOINS = {
    "USDT", "USDC", "USDS", "DAI", "BUSD", "TUSD", "FDUSD",
    "USDD", "USDE", "PYUSD", "FRAX", "LUSD", "GUSD", "USTC",
    "USD1", "RLUSD", "EURC", "EURT", "XAUT", "PAXG",
    # wrapped / staked
    "WBTC", "STETH", "WSTETH", "WETH", "WEETH", "WBETH", "CBBTC",
    "BSC-USD", "SUSDE", "BUIDL", "USDTB", "USDF",
}

EXCHANGE_ID = 'lbank'
exchange = getattr(ccxt, EXCHANGE_ID)({
    'enableRateLimit': True,
    'options': {'defaultType': 'spot'},
})


# ================== لیست ارزها ==================
def get_top_coins():
    url = "https://api.coingecko.com/api/v3/coins/markets"
    params = {
        "vs_currency": "usd", "order": "market_cap_desc",
        "per_page": TOP_N * 3, "page": 1, "sparkline": "false",
    }
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    coins = []
    for c in r.json():
        sym = c["symbol"].upper()
        if sym in STABLECOINS:
            continue
        coins.append(sym)
        if len(coins) >= TOP_N:
            break
    return coins


# ================== تبدیل ۱ ساعته → ۴ ساعته ==================
def resample_1h_to_4h(df_1h):
    if df_1h is None or len(df_1h) < 4:
        return None
    df = df_1h.copy()
    df["bucket"] = df["timestamp"].dt.floor("4h")

    counts = df.groupby("bucket").size()
    complete = counts[counts == 4].index
    df = df[df["bucket"].isin(complete)]
    if len(df) < 4:
        return None

    agg = df.groupby("bucket").agg(
        open=("open", "first"),
        high=("high", "max"),
        low=("low", "min"),
        close=("close", "last"),
        volume=("volume", "sum"),
    ).reset_index().rename(columns={"bucket": "timestamp"})

    # bucketهای ناقص قبلاً حذف شده‌اند، پس کندل جاری (ناقص) خودبه‌خود نیست
    return agg[["timestamp", "open", "high", "low", "close", "volume"]]


def _to_df(ohlcv):
    df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = df[col].astype(float)
    return df


def get_ohlc_1h(symbol_display, limit=2000):
    symbol = f"{symbol_display}/USDT"
    if symbol not in exchange.markets:
        return None
    try:
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe='1h', limit=limit)
    except Exception as e:
        print(f"  ❌ خطا در دریافت {symbol}: {e}")
        return None
    if not ohlcv or len(ohlcv) < 50:
        return None
    return _to_df(ohlcv)


def get_ohlc_4h(symbol_display, limit_4h=1000):
    symbol = f"{symbol_display}/USDT"
    if symbol not in exchange.markets:
        return None, "نماد در LBank نیست"

    try:
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe='4h', limit=limit_4h)
        if ohlcv and len(ohlcv) >= 50:
            df = _to_df(ohlcv)
            # کندل آخر ممکن است هنوز بسته نشده باشد
            return df.iloc[:-1].reset_index(drop=True), "4h مستقیم"
    except Exception:
        pass

    df_1h = get_ohlc_1h(symbol_display, limit=2000)
    if df_1h is None:
        return None, "1h هم نشد"
    df_4h = resample_1h_to_4h(df_1h)
    if df_4h is None or len(df_4h) < 50:
        n = len(df_4h) if df_4h is not None else 0
        return None, f"تبدیل 1h→4h فقط {n} کندل داد"
    return df_4h, f"1h→4h ({len(df_1h)} کندل 1h)"


# ================== ADX ==================
def calc_adx(df, di_len=14, adx_len=14):
    high, low, close = df["high"], df["low"], df["close"]
    up = high.diff()
    down = -low.diff()
    plus_dm = np.where((up > down) & (up > 0), up, 0.0)
    minus_dm = np.where((down > up) & (down > 0), down, 0.0)
    tr = pd.concat([high - low,
                    (high - close.shift()).abs(),
                    (low - close.shift()).abs()], axis=1).max(axis=1)

    def rma(s, n):
        return s.ewm(alpha=1.0 / n, adjust=False).mean()

    atr = rma(tr, di_len)
    plus_di = 100 * rma(pd.Series(plus_dm, index=df.index), di_len) / atr
    minus_di = 100 * rma(pd.Series(minus_dm, index=df.index), di_len) / atr
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    df["plus_di"] = plus_di
    df["minus_di"] = minus_di
    df["adx"] = rma(dx, adx_len)
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


# ================== سیگنال (وکتوری) ==================
def detect_signals(df, hl_trend=HL_TREND, hl_range=HL_RANGE):
    adx_ok = (
        (df["adx"] > hl_trend)
        & (df["adx"] > df["adx"].shift(1))
        & (df["plus_di"] > df["minus_di"])
        & (df["plus_di"] >= hl_trend)
    )
    j, t, l = df["alligator_jaw"], df["alligator_teeth"], df["alligator_lips"]
    if BULLISH_ALLIGATOR:
        alligator_ok = (l > t) & (t > j)
    else:
        alligator_ok = (j > t) & (t > l)
    range_ok = df["adx"] > hl_range

    raw = (adx_ok & alligator_ok & range_ok).fillna(False)
    if ONLY_NEW_SIGNALS:
        raw = raw & ~raw.shift(1, fill_value=False)
    df["signal"] = raw
    return df


# ================== تحلیل ==================
def analyze_signals(df, lookahead=LOOKAHEAD):
    signals = df.index[df["signal"]].tolist()
    hit_matrix = {t: {d: 0 for d in DRAWDOWN_LEVELS} for t in TARGETS_UP}
    reached = {t: 0 for t in TARGETS_UP}
    total = 0
    for sig_idx in signals:
        # فقط سیگنال‌هایی که lookahead کندل کامل بعدشان هست
        if sig_idx + lookahead >= len(df):
            continue
        entry = df.loc[sig_idx, "close"]
        if entry <= 0:
            continue
        future = df.loc[sig_idx + 1: sig_idx + lookahead]
        total += 1
        for t in TARGETS_UP:
            target = entry * (1 + t / 100.0)
            mask = (future["high"] >= target).values
            if not mask.any():
                continue
            hit_pos = int(mask.argmax())
            reached[t] += 1
            seg = future.iloc[:hit_pos + 1]
            dd = (entry - seg["low"].min()) / entry * 100
            for d in DRAWDOWN_LEVELS:
                if dd < d:
                    hit_matrix[t][d] += 1
    return {"total_signals": total, "reached": reached, "matrix": hit_matrix}


# ================== جدول ==================
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
    lines.append("رسیده فقط: " + " | ".join(
        f"{t}٪→{res['reached'][t] / total * 100:.0f}٪" for t in TARGETS_UP))
    return "\n".join(lines)


def build_report(coin_results, source_info):
    total_all = sum(r["total_signals"] for r in coin_results.values())
    if total_all == 0:
        return "📊 هیچ سیگنالی یافت نشد."
    lines = ["=" * 70,
             f"گزارش ADX + Alligator | TF=4h | {EXCHANGE_ID.upper()} | سیگنال کل: {total_all}",
             "=" * 70, ""]
    gm = {t: {d: 0 for d in DRAWDOWN_LEVELS} for t in TARGETS_UP}
    gr = {t: 0 for t in TARGETS_UP}
    for sym, res in coin_results.items():
        if res["total_signals"] == 0:
            continue
        src = source_info.get(sym, "?")
        lines.append(f"---- {sym} | {res['total_signals']} سیگنال | منبع: {src} ----")
        lines.append(build_matrix_table(res))
        lines.append("")
        for t in TARGETS_UP:
            gr[t] += res["reached"][t]
            for d in DRAWDOWN_LEVELS:
                gm[t][d] += res["matrix"][t][d]
    lines.append("=" * 70)
    lines.append("خلاصه کل")
    lines.append("=" * 70)
    lines.append(build_matrix_table({"total_signals": total_all, "reached": gr, "matrix": gm}))
    return "\n".join(lines)


# ================== main ==================
def empty_result():
    return {
        "total_signals": 0,
        "reached": {t: 0 for t in TARGETS_UP},
        "matrix": {t: {d: 0 for d in DRAWDOWN_LEVELS} for t in TARGETS_UP},
    }


def main():
    print(f"بارگذاری بازارهای {EXCHANGE_ID.upper()}...")
    try:
        exchange.load_markets()
        print(f"✅ {len(exchange.markets)} بازار بارگذاری شد.\n")
    except Exception as e:
        print(f"❌ خطا: {e}")
        return

    coins = get_top_coins()
    print(f"ارزها: {', '.join(coins)}\n")

    missing = [s for s in coins if f"{s}/USDT" not in exchange.markets]
    if missing:
        print(f"⚠️ این‌ها در LBank نیستند: {', '.join(missing)}\n")

    coin_results, source_info = {}, {}

    for sym in coins:
        print(f"بررسی {sym}...")
        try:
            df, src = get_ohlc_4h(sym)
            if df is None or len(df) < 50:
                print(f"  ⚠️ {src}\n")
                coin_results[sym] = empty_result()
                source_info[sym] = src
                continue
            source_info[sym] = f"{src} ({len(df)} کندل 4h)"
            df = calc_adx(df)
            df = calc_alligator(df)
            df = detect_signals(df)
            res = analyze_signals(df)
            coin_results[sym] = res
            print(f"  ✅ {res['total_signals']} سیگنال | منبع: {source_info[sym]}\n")
        except Exception as e:
            print(f"  ❌ {e}\n")
            coin_results[sym] = empty_result()
            source_info[sym] = "خطا"

    report = build_report(coin_results, source_info)
    print("\n\n" + report)
    with open("report.txt", "w", encoding="utf-8") as f:
        f.write(report)


if __name__ == "__main__":
    main()
