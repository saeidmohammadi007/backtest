import os
import time
import requests
import pandas as pd
import numpy as np
import asyncio
from telegram import Bot

# ================== تنظیمات ==================
TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN")
CHAT_ID = os.environ.get("CHAT_ID")

TOP_N = 20
LOOKAHEAD = 100

TARGETS_UP = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
DRAWDOWN_LEVELS = [2, 3, 4, 5, 6, 7, 8, 9, 10]

# ================== دریافت ۲۰ ارز برتر ==================
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
    return [(c["id"], c["symbol"].upper()) for c in data]


# ================== دریافت OHLC ۴ ساعته ==================
def get_ohlc_4h(coin_id, days=90):
    """
    CoinGecko OHLC برای 3-30 روز، کندل 4 ساعته برمی‌گرداند.
    برای پوشش کافی از 90 روز استفاده می‌کنیم؛ اگر API بازه دیگری داد،
    همان را می‌پذیریم (فقط باید حداقل ~50 کندل داشته باشیم).
    """
    url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/ohlc"
    params = {"vs_currency": "usd", "days": days}
    r = requests.get(url, params=params, timeout=30)
    if r.status_code != 200:
        return None
    data = r.json()
    if not data:
        return None
    df = pd.DataFrame(data, columns=["timestamp", "open", "high", "low", "close"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


# ================== ADX (مطابق Gu5) ==================
def calc_adx(df, di_len=14, adx_len=14):
    high = df["high"]
    low = df["low"]
    close = df["close"]

    up = high.diff()
    down = -low.diff()

    plus_dm = np.where((up > down) & (up > 0), up, 0.0)
    minus_dm = np.where((down > up) & (down > 0), down, 0.0)

    tr1 = high - low
    tr2 = (high - close.shift()).abs()
    tr3 = (low - close.shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    def rma(series, length):
        alpha = 1.0 / length
        return series.ewm(alpha=alpha, adjust=False).mean()

    atr = rma(tr, di_len)
    plus_di = 100 * rma(pd.Series(plus_dm, index=df.index), di_len) / atr
    minus_di = 100 * rma(pd.Series(minus_dm, index=df.index), di_len) / atr

    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    adx = rma(dx, adx_len)

    df["plus_di"] = plus_di
    df["minus_di"] = minus_di
    df["adx"] = adx
    return df


# ================== Williams Alligator ==================
def calc_alligator(df):
    median = (df["high"] + df["low"]) / 2

    def smma(series, length):
        alpha = 1.0 / length
        return series.ewm(alpha=alpha, adjust=False).mean()

    jaw_raw = smma(median, 13)
    df["alligator_jaw"] = jaw_raw.shift(8)        # آبی

    teeth_raw = smma(median, 8)
    df["alligator_teeth"] = teeth_raw.shift(5)    # سبز

    lips_raw = smma(median, 5)
    df["alligator_lips"] = lips_raw.shift(3)      # قرمز

    return df


# ================== تشخیص سیگنال (مثلث سبز بزرگ + ترتیب الیگیتور) ==================
def detect_signals(df, hl_trend=35, hl_range=20):
    """
    شرط سیگنال:
    1) ADX > 35 و ADX > ADX[1] و +DI > -DI و +DI >= 35  → مثلث سبز بزرگ
    2) Alligator: Jaw (آبی) > Teeth (سبز) > Lips (قرمز)
    3) ADX > 20 (range filter)
    """
    df["signal"] = False

    for i in range(1, len(df)):
        adx_ok = (
            df.loc[i, "adx"] > hl_trend
            and df.loc[i, "adx"] > df.loc[i - 1, "adx"]
            and df.loc[i, "plus_di"] > df.loc[i, "minus_di"]
            and df.loc[i, "plus_di"] >= hl_trend
        )

        j = df.loc[i, "alligator_jaw"]
        t = df.loc[i, "alligator_teeth"]
        l = df.loc[i, "alligator_lips"]
        alligator_ok = (
            pd.notna(j) and pd.notna(t) and pd.notna(l)
            and j > t > l
        )

        range_ok = df.loc[i, "adx"] > hl_range

        df.loc[i, "signal"] = bool(adx_ok and alligator_ok and range_ok)

    return df


# ================== تحلیل ماتریس ریزش × هدف ==================
def analyze_signals(df, lookahead=LOOKAHEAD):
    """
    برای هر سیگنال و هر هدف:
      - آیا قیمت به هدف رسید؟
      - اگر رسید، حداکثر ریزش قبل از رسیدن چقدر بود؟
    خروجی:
      total_signals : تعداد کل سیگنال‌ها
      reached[t]    : تعداد سیگنال‌هایی که به هدف t رسیدند (بدون قید ریزش)
      matrix[t][d]  : تعداد سیگنال‌هایی که به هدف t رسیدند و ریزش قبل از هدف < d بود
    """
    signals = df.index[df["signal"]].tolist()

    hit_matrix = {t: {d: 0 for d in DRAWDOWN_LEVELS} for t in TARGETS_UP}
    reached_matrix = {t: 0 for t in TARGETS_UP}
    total_signals = 0

    for sig_idx in signals:
        future = df.loc[sig_idx + 1: sig_idx + lookahead]
        if len(future) == 0:
            continue
        total_signals += 1
        entry = df.loc[sig_idx, "close"]
        if entry <= 0:
            continue

        for t in TARGETS_UP:
            target_price = entry * (1 + t / 100.0)
            hit_mask = (future["high"] >= target_price).values

            if not hit_mask.any():
                continue

            hit_pos = int(hit_mask.argmax())
            reached_matrix[t] += 1

            segment = future.iloc[: hit_pos + 1]
            min_low = segment["low"].min()
            drawdown_pct = (entry - min_low) / entry * 100

            for d in DRAWDOWN_LEVELS:
                if drawdown_pct < d:
                    hit_matrix[t][d] += 1

    return {
        "total_signals": total_signals,
        "reached": reached_matrix,
        "matrix": hit_matrix,
    }


# ================== ساخت جدول متنی ==================
def build_matrix_table(res):
    total = res["total_signals"]
    if total == 0:
        return "  (بدون سیگنال)"

    header = "هدف/ریزش  |" + "|".join(f"<{d}٪".rjust(5) for d in DRAWDOWN_LEVELS)
    lines = [header, "-" * len(header)]

    for t in TARGETS_UP:
        row = [f"{t:>3}٪      |"]
        for d in DRAWDOWN_LEVELS:
            cnt = res["matrix"][t][d]
            pct = cnt / total * 100
            row.append(f"{pct:>4.0f}٪")
        lines.append("|".join(row))

    # خط «رسیده صرفاً»
    reached_line = "رسیده فقط: " + " | ".join(
        f"{t}٪→{res['reached'][t] / total * 100:.0f}٪" for t in TARGETS_UP
    )
    lines.append("")
    lines.append(reached_line)
    return "\n".join(lines)


# ================== پیام فارسی نهایی ==================
def build_fa_message(coin_results):
    total_all = sum(r["total_signals"] for r in coin_results.values())

    if total_all == 0:
        return "📊 در ۲۰ ارز برتر، هیچ سیگنالی با شرایط مشخص‌شده در تایم‌فریم ۴ ساعته یافت نشد."

    lines = []
    lines.append("📊 *گزارش ADX + Alligator — ماتریس ریسک/بازده*")
    lines.append(f"تایم‌فریم ۴ ساعته | {len(coin_results)} ارز | مجموع سیگنال: {total_all}")
    lines.append("")
    lines.append("📖 *راهنما:*")
    lines.append("هر خانه = درصدی از کل سیگنال‌ها که هم به آن هدف رسیدند")
    lines.append("و هم قبل از رسیدن به هدف، ریزششان کمتر از حد ستون بود.")
    lines.append("مثال: خانه (هدف ۱۰٪، ستون <4٪) = ۶۰٪ یعنی ۶۰٪ سیگنال‌ها")
    lines.append("با ریزش کمتر از ۴٪ به هدف ۱۰٪ رسیدند.")
    lines.append("")

    global_matrix = {t: {d: 0 for d in DRAWDOWN_LEVELS} for t in TARGETS_UP}
    global_reached = {t: 0 for t in TARGETS_UP}

    for sym, res in coin_results.items():
        if res["total_signals"] == 0:
            continue
        lines.append(f"🔹 *{sym}* — {res['total_signals']} سیگنال")
        lines.append(build_matrix_table(res))
        lines.append("")

        for t in TARGETS_UP:
            global_reached[t] += res["reached"][t]
            for d in DRAWDOWN_LEVELS:
                global_matrix[t][d] += res["matrix"][t][d]

    lines.append("📌 *خلاصه کل ۲۰ ارز*")
    global_res = {
        "total_signals": total_all,
        "reached": global_reached,
        "matrix": global_matrix,
    }
    lines.append(build_matrix_table(global_res))

    return "\n".join(lines)


# ================== ارسال به تلگرام ==================
async def send_telegram(message):
    bot = Bot(token=TELEGRAM_TOKEN)
    max_len = 3800
    if len(message) <= max_len:
        await bot.send_message(chat_id=CHAT_ID, text=message, parse_mode="Markdown")
    else:
        for i in range(0, len(message), max_len):
            chunk = message[i: i + max_len]
            await bot.send_message(chat_id=CHAT_ID, text=chunk, parse_mode="Markdown")


# ================== main ==================
def main():
    print("دریافت لیست ۲۰ ارز برتر...")
    coins = get_top_coins()
    print(f"تعداد ارز: {len(coins)}")

    empty_result = lambda: {
        "total_signals": 0,
        "reached": {t: 0 for t in TARGETS_UP},
        "matrix": {t: {d: 0 for d in DRAWDOWN_LEVELS} for t in TARGETS_UP},
    }

    coin_results = {}

    for coin_id, symbol in coins:
        print(f"بررسی {symbol} ({coin_id})...")
        try:
            df = get_ohlc_4h(coin_id)
            if df is None or len(df) < 50:
                print(f"  داده کافی برای {symbol} وجود ندارد.")
                coin_results[symbol] = empty_result()
                time.sleep(2)  # جلوگیری از rate limit
                continue

            df = calc_adx(df)
            df = calc_alligator(df)
            df = detect_signals(df)
            res = analyze_signals(df)
            coin_results[symbol] = res
            print(f"  سیگنال‌ها: {res['total_signals']}")
        except Exception as e:
            print(f"  خطا برای {symbol}: {e}")
            coin_results[symbol] = empty_result()

        time.sleep(2)  # CoinGecko rate limit

    message = build_fa_message(coin_results)
    print("ارسال به تلگرام...")
    asyncio.run(send_telegram(message))
    print("انجام شد.")


if __name__ == "__main__":
    main()
