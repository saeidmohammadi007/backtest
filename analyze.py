import os
import time
import numpy as np
import pandas as pd
import requests
from datetime import datetime, timedelta

# ================== تنظیمات ==================
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

SYMBOL = "BTCUSDT"
TARGET_DATE = "2024-11-05"
DATA_START_DATE = "2019-01-01"
MIN_GAP_DAYS = 30
TOP_N_SIMILAR = 20
FUTURE_HORIZONS = [1, 3, 7, 14, 30, 60, 90]

SCENARIOS = [
    {"name": "ریزش زیر ۱٪ و رشد بالای ۱۰٪", "drawdown": -0.01, "up_side": 0.10},
    {"name": "ریزش زیر ۱٪ و رشد بالای ۲۰٪", "drawdown": -0.01, "up_side": 0.20},
    {"name": "ریزش زیر ۲٪ و رشد بالای ۱۰٪", "drawdown": -0.02, "up_side": 0.10},
    {"name": "ریزش زیر ۲٪ و رشد بالای ۲۰٪", "drawdown": -0.02, "up_side": 0.20},
    {"name": "ریزش زیر ۳٪ و رشد بالای ۱۰٪", "drawdown": -0.03, "up_side": 0.10},
    {"name": "ریزش زیر ۳٪ و رشد بالای ۲۰٪", "drawdown": -0.03, "up_side": 0.20},
    {"name": "ریزش زیر ۴٪ و رشد بالای ۲۵٪", "drawdown": -0.04, "up_side": 0.25},
    {"name": "ریزش زیر ۵٪ و رشد بالای ۱۵٪", "drawdown": -0.05, "up_side": 0.15},
]


# ================== دریافت داده از Binance ==================
def fetch_data(symbol="BTCUSDT", start_date="2019-01-01", end_date=None):
    """دریافت داده روزانه از Binance API"""
    if end_date is None:
        end_date = datetime.now().strftime("%Y-%m-%d")

    start_ts = int(datetime.strptime(start_date, "%Y-%m-%d").timestamp() * 1000)
    end_ts = int(datetime.strptime(end_date, "%Y-%m-%d").timestamp() * 1000)

    url = "https://api.binance.com/api/v3/klines"
    all_data = []
    current = start_ts

    print(f"دریافت داده {symbol} از {start_date} تا {end_date} از Binance...")

    while current < end_ts:
        params = {
            "symbol": symbol,
            "interval": "1d",
            "startTime": current,
            "limit": 1000,
        }
        try:
            r = requests.get(url, params=params, timeout=30)
            r.raise_for_status()
            data = r.json()
        except Exception as e:
            print(f"خطا در دریافت داده: {e}")
            break

        if not data:
            break
        all_data.extend(data)
        current = data[-1][0] + 86400000
        time.sleep(0.2)

    if not all_data:
        raise ValueError("داده‌ای از Binance دریافت نشد.")

    df = pd.DataFrame(all_data, columns=[
        "date", "open", "high", "low", "close", "volume",
        "close_time", "qav", "trades", "tbbav", "tbqav", "ignore",
    ])
    df["date"] = pd.to_datetime(df["date"], unit="ms")
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = df[col].astype(float)

    df = df[["date", "open", "high", "low", "close", "volume"]]
    df = df.sort_values("date").reset_index(drop=True)

    print(f"تعداد کل کندل‌ها از Binance: {len(df)}")
    return df


# ================== محاسبه فاصله اقلیدسی ==================
def candle_vector(row):
    """تبدیل کندل به بردار نرمال‌شده"""
    o, h, l, c = row["open"], row["high"], row["low"], row["close"]
    body = c - o
    range_total = h - l
    if range_total == 0:
        range_total = 1e-9
    return np.array([
        (o - c) / range_total,
        (h - c) / range_total,
        (l - c) / range_total,
        body / range_total,
        (c - o) / o,
    ])


def euclidean_distance(v1, v2):
    return float(np.sqrt(np.sum((v1 - v2) ** 2)))


def find_similar_candles(df, target_date, top_n=20, min_gap_days=30):
    """پیدا کردن کندل‌های مشابه با فاصله اقلیدسی"""
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])

    target_row = df[df["date"] == pd.Timestamp(target_date)]
    if target_row.empty:
        raise ValueError(f"کندل تاریخ {target_date} پیدا نشد.")

    target_vec = candle_vector(target_row.iloc[0])
    target_price = float(target_row.iloc[0]["close"])

    distances = []
    for _, row in df.iterrows():
        if row["date"] == pd.Timestamp(target_date):
            continue
        time_gap = abs((row["date"] - pd.Timestamp(target_date)).days)
        if time_gap < min_gap_days:
            continue

        vec = candle_vector(row)
        dist = euclidean_distance(target_vec, vec)
        distances.append({
            "date": row["date"],
            "distance": dist,
            "open": row["open"],
            "high": row["high"],
            "low": row["low"],
            "close": row["close"],
        })

    if not distances:
        raise ValueError("هیچ کندل مشابهی پیدا نشد.")

    dist_df = pd.DataFrame(distances).sort_values("distance").reset_index(drop=True)

    selected = []
    last_date = None
    for _, row in dist_df.iterrows():
        if last_date is None or abs((row["date"] - last_date).days) >= min_gap_days:
            selected.append(row)
            last_date = row["date"]
        if len(selected) >= top_n:
            break

    result_df = pd.DataFrame(selected)
    print(f"تعداد کندل مشابه پیدا شده: {len(result_df)}")
    print(f"میانگین فاصله اقلیدسی: {result_df['distance'].mean():.4f}")

    return result_df, target_price


# ================== محاسبه احتمالات ==================
def calculate_outcomes(df, similar_df):
    """محاسبه درصد احتمال تحقق سناریوها"""
    results = []

    for horizon in FUTURE_HORIZONS:
        outcomes = []
        for _, row in similar_df.iterrows():
            match_date = row["date"]
            future = df[df["date"] > match_date].head(horizon)
            if len(future) < horizon:
                continue

            entry_price = row["close"]
            future_close = future["close"].values

            max_drawdown = (future_close.min() - entry_price) / entry_price
            max_runup = (future_close.max() - entry_price) / entry_price

            outcomes.append({
                "max_drawdown": max_drawdown,
                "max_runup": max_runup,
                "final_return": (future_close[-1] - entry_price) / entry_price,
            })

        if not outcomes:
            continue

        out_df = pd.DataFrame(outcomes)
        total = len(out_df)

        horizon_result = {
            "horizon": horizon,
            "total_cases": total,
            "avg_max_drawdown": out_df["max_drawdown"].mean(),
            "avg_max_runup": out_df["max_runup"].mean(),
            "avg_final_return": out_df["final_return"].mean(),
        }

        for scenario in SCENARIOS:
            dd = scenario["drawdown"]
            up = scenario["up_side"]
            condition = (out_df["max_drawdown"] >= dd) & (out_df["max_runup"] >= up)
            horizon_result[f"prob_{scenario['name']}"] = condition.mean() * 100

        results.append(horizon_result)

    return pd.DataFrame(results)


# ================== ارسال به تلگرام ==================
def send_telegram_message(token, chat_id, message):
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {
        "chat_id": chat_id,
        "text": message,
        "parse_mode": "HTML",
        "disable_web_page_preview": True,
    }
    try:
        response = requests.post(url, json=payload, timeout=30)
        response.raise_for_status()
        print("پیام با موفقیت به تلگرام ارسال شد.")
        return True
    except Exception as e:
        print(f"خطا در ارسال پیام تلگرام: {e}")
        return False


def send_long_message(token, chat_id, message, max_len=3800):
    """ارسال پیام طولانی به صورت تکه‌تکه"""
    if len(message) <= max_len:
        return send_telegram_message(token, chat_id, message)

    parts = []
    current = ""
    for line in message.split("\n"):
        if len(current) + len(line) + 1 > max_len:
            parts.append(current)
            current = line
        else:
            current = current + "\n" + line if current else line
    if current:
        parts.append(current)

    for i, part in enumerate(parts):
        header = f"<b>({i+1}/{len(parts)})</b>\n" if len(parts) > 1 else ""
        send_telegram_message(token, chat_id, header + part)
        time.sleep(1)


def format_results(similar_df, target_price, prob_df):
    lines = []
    lines.append("📊 <b>تحلیل شباهت کندل BTC/USDT</b>")
    lines.append(f"📅 کندل مرجع: <b>{TARGET_DATE}</b>")
    lines.append(f"💰 قیمت بسته شدن: <b>${target_price:,.2f}</b>")
    lines.append(f"🔍 تعداد کندل مشابه: <b>{len(similar_df)}</b>")
    lines.append(f"📏 میانگین فاصله اقلیدسی: <b>{similar_df['distance'].mean():.4f}</b>")
    lines.append(f"📚 منبع داده: <b>Binance (از ۲۰۱۹)</b>")
    lines.append("")

    lines.append("🕯️ <b>۱۰ کندل مشابه اول:</b>")
    for _, row in similar_df.head(10).iterrows():
        date_str = row["date"].strftime("%Y-%m-%d")
        lines.append(
            f"  • {date_str} | فاصله: {row['distance']:.3f} | بسته: ${row['close']:,.0f}"
        )
    lines.append("")

    lines.append("📈 <b>نتایج آماری بر اساس کندل‌های مشابه:</b>")
    lines.append("")

    for _, row in prob_df.iterrows():
        horizon = int(row["horizon"])
        total = int(row["total_cases"])
        avg_dd = row["avg_max_drawdown"] * 100
        avg_up = row["avg_max_runup"] * 100

        lines.append(f"⏳ <b>{horizon} روز آینده</b> (نمونه: {total})")
        lines.append(f"  📉 میانگین حداکثر افت: {avg_dd:.1f}%")
        lines.append(f"  📈 میانگین حداکثر رشد: {avg_up:.1f}%")
        for scenario in SCENARIOS:
            prob = row.get(f"prob_{scenario['name']}", 0)
            lines.append(f"  ▪️ {scenario['name']}: <b>{prob:.1f}%</b>")
        lines.append("")

    lines.append("⚠️ <i>این تحلیل صرفاً جنبه آماری دارد و توصیه سرمایه‌گذاری نیست.</i>")
    return "\n".join(lines)


# ================== اجرای اصلی ==================
def main():
    print("شروع تحلیل...")

    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("خطا: TELEGRAM_BOT_TOKEN و TELEGRAM_CHAT_ID باید تنظیم شوند.")
        return

    end_date = (datetime.strptime(TARGET_DATE, "%Y-%m-%d") + timedelta(days=120)).strftime("%Y-%m-%d")
    today = datetime.now().strftime("%Y-%m-%d")
    if end_date > today:
        end_date = today

    df = fetch_data(SYMBOL, DATA_START_DATE, end_date)

    df = df[df["date"] >= pd.Timestamp(DATA_START_DATE)].reset_index(drop=True)
    print(f"داده‌های فیلترشده: {len(df)} کندل")

    similar_df, target_price = find_similar_candles(
        df, TARGET_DATE, top_n=TOP_N_SIMILAR, min_gap_days=MIN_GAP_DAYS
    )

    prob_df = calculate_outcomes(df, similar_df)

    message = format_results(similar_df, target_price, prob_df)
    print("\n" + "=" * 50)
    print(message)
    print("=" * 50)

    send_long_message(TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, message)


if __name__ == "__main__":
    main()
