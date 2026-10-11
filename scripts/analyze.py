import os
import time
import numpy as np
import pandas as pd
import yfinance as yf
import requests
from datetime import datetime, timedelta

# ================== تنظیمات ==================
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

# ۱۰ ارز برتر
COINS = [
    {"symbol": "BTC-USD",  "name": "Bitcoin"},
    {"symbol": "ETH-USD",  "name": "Ethereum"},
    {"symbol": "BNB-USD",  "name": "Binance Coin"},
    {"symbol": "SOL-USD",  "name": "Solana"},
    {"symbol": "XRP-USD",  "name": "XRP"},
    {"symbol": "ADA-USD",  "name": "Cardano"},
    {"symbol": "DOGE-USD", "name": "Dogecoin"},
    {"symbol": "TRX-USD",  "name": "TRON"},
    {"symbol": "DOT-USD",  "name": "Polkadot"},
    {"symbol": "MATIC-USD", "name": "Polygon"},
]

DATA_START_DATE = "2019-01-01"
DATA_END_DATE   = "2024-12-31"
TARGET_DATE     = "2024-11-05"
MIN_GAP_DAYS    = 30
TOP_N_SIMILAR   = 15
FUTURE_HORIZONS = [1, 3, 7, 14, 30, 60, 90]

SCENARIOS = [
    {"name": "ریزش زیر ۱٪ و رشد بالای ۱۰٪", "drawdown": -0.01, "up_side": 0.10},
    {"name": "ریزش زیر ۱٪ و رشد بالای ۲۰٪", "drawdown": -0.01, "up_side": 0.20},
    {"name": "ریزش زیر ۲٪ و رشد بالای ۱۰٪", "drawdown": -0.02, "up_side": 0.10},
    {"name": "ریزش زیر ۲٪ و رشد بالای ۲۰٪", "drawdown": -0.02, "up_side": 0.20},
    {"name": "ریزش زیر ۳٪ و رشد بالای ۱۰٪", "drawdown": -0.03, "up_side": 0.10},
    {"name": "ریزش زیر ۳٪ و رشد بالای ۲۰٪", "drawdown": -0.03, "up_side": 0.20},
]


# ================== دریافت داده از Yahoo Finance ==================
def fetch_data(symbol, start_date, end_date):
    """دریافت داده روزانه از Yahoo Finance"""
    print(f"  دریافت {symbol} از {start_date} تا {end_date}...")
    df = yf.download(symbol, start=start_date, end=end_date,
                     interval="1d", auto_adjust=False, progress=False)

    if df.empty:
        print(f"  ⚠️ داده‌ای برای {symbol} دریافت نشد.")
        return None

    # اگر ستون‌ها MultiIndex هستند، تخت کن
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # lowercase کردن ستون‌ها (قبل از reset_index)
    df.columns = [str(c).lower() for c in df.columns]

    # reset_index و سپس lowercase کردن دوباره (چون index قبلاً Date بوده)
    df = df.reset_index()
    df.columns = [str(c).lower() for c in df.columns]

    # حذف adj close اگر هست
    if "adj close" in df.columns:
        df = df.drop(columns=["adj close"])

    # بررسی وجود ستون date
    if "date" not in df.columns:
        print(f"  ❌ ستون date پیدا نشد. ستون‌ها: {list(df.columns)}")
        return None

    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
    df = df.sort_values("date").reset_index(drop=True)

    # فقط ۲۰۱۹ تا ۲۰۲۴
    df = df[(df["date"] >= pd.Timestamp(start_date)) &
            (df["date"] <= pd.Timestamp(end_date))].reset_index(drop=True)

    print(f"  ✅ {len(df)} کندل دریافت شد.")
    return df


# ================== محاسبه فاصله اقلیدسی ==================
def candle_vector(row):
    o, h, l, c = row["open"], row["high"], row["low"], row["close"]
    body = c - o
    rng = h - l if (h - l) != 0 else 1e-9
    return np.array([
        (o - c) / rng,
        (h - c) / rng,
        (l - c) / rng,
        body / rng,
        (c - o) / o if o != 0 else 0,
    ])


def euclidean_distance(v1, v2):
    return float(np.sqrt(np.sum((v1 - v2) ** 2)))


def find_similar_candles(df, target_date, top_n=15, min_gap_days=30):
    df = df.copy()
    target_row = df[df["date"] == pd.Timestamp(target_date)]
    if target_row.empty:
        return None, None

    target_vec = candle_vector(target_row.iloc[0])
    target_price = float(target_row.iloc[0]["close"])

    distances = []
    for _, row in df.iterrows():
        if row["date"] == pd.Timestamp(target_date):
            continue
        gap = abs((row["date"] - pd.Timestamp(target_date)).days)
        if gap < min_gap_days:
            continue
        vec = candle_vector(row)
        distances.append({
            "date": row["date"],
            "distance": euclidean_distance(target_vec, vec),
            "close": row["close"],
        })

    if not distances:
        return None, None

    dist_df = pd.DataFrame(distances).sort_values("distance").reset_index(drop=True)

    selected, last_date = [], None
    for _, row in dist_df.iterrows():
        if last_date is None or abs((row["date"] - last_date).days) >= min_gap_days:
            selected.append(row)
            last_date = row["date"]
        if len(selected) >= top_n:
            break

    return pd.DataFrame(selected), target_price


# ================== محاسبه احتمالات ==================
def calculate_outcomes(df, similar_df):
    results = []
    for horizon in FUTURE_HORIZONS:
        outcomes = []
        for _, row in similar_df.iterrows():
            future = df[df["date"] > row["date"]].head(horizon)
            if len(future) < horizon:
                continue
            entry = row["close"]
            fc = future["close"].values
            outcomes.append({
                "max_drawdown": (fc.min() - entry) / entry,
                "max_runup": (fc.max() - entry) / entry,
                "final_return": (fc[-1] - entry) / entry,
            })
        if not outcomes:
            continue
        out_df = pd.DataFrame(outcomes)
        res = {
            "horizon": horizon,
            "total": len(out_df),
            "avg_dd": out_df["max_drawdown"].mean(),
            "avg_up": out_df["max_runup"].mean(),
        }
        for s in SCENARIOS:
            cond = (out_df["max_drawdown"] >= s["drawdown"]) & \
                   (out_df["max_runup"] >= s["up_side"])
            res[f"p_{s['name']}"] = cond.mean() * 100
        results.append(res)
    return pd.DataFrame(results)


# ================== ارسال تلگرام ==================
def send_telegram(token, chat_id, text):
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    try:
        r = requests.post(url, json={
            "chat_id": chat_id, "text": text,
            "parse_mode": "HTML", "disable_web_page_preview": True,
        }, timeout=30)
        r.raise_for_status()
        return True
    except Exception as e:
        print(f"خطا تلگرام: {e}")
        return False


def send_long(token, chat_id, text, max_len=3800):
    if len(text) <= max_len:
        return send_telegram(token, chat_id, text)
    parts, cur = [], ""
    for line in text.split("\n"):
        if len(cur) + len(line) + 1 > max_len:
            parts.append(cur)
            cur = line
        else:
            cur = cur + "\n" + line if cur else line
    if cur:
        parts.append(cur)
    for i, p in enumerate(parts):
        hdr = f"<b>({i+1}/{len(parts)})</b>\n" if len(parts) > 1 else ""
        send_telegram(token, chat_id, hdr + p)
        time.sleep(1)


# ================== قالب‌بندی نتایج ==================
def format_coin_result(coin_name, symbol, similar_df, target_price, prob_df):
    lines = []
    lines.append(f"🪙 <b>{coin_name} ({symbol})</b>")
    lines.append(f"💰 قیمت بسته شدن مرجع: <b>${target_price:,.4f}</b>")
    lines.append(f"🔍 تعداد کندل مشابه: <b>{len(similar_df)}</b>")
    lines.append(f"📏 میانگین فاصله: <b>{similar_df['distance'].mean():.4f}</b>")
    lines.append("")

    lines.append("🕯️ <b>۵ کندل مشابه اول:</b>")
    for _, r in similar_df.head(5).iterrows():
        lines.append(f"  • {r['date'].strftime('%Y-%m-%d')} | فاصله: {r['distance']:.3f} | بسته: ${r['close']:,.2f}")
    lines.append("")

    for _, row in prob_df.iterrows():
        h = int(row["horizon"])
        lines.append(f"⏳ <b>{h} روز آینده</b> (نمونه: {int(row['total'])})")
        lines.append(f"  📉 میانگین حداکثر افت: {row['avg_dd']*100:.1f}%")
        lines.append(f"  📈 میانگین حداکثر رشد: {row['avg_up']*100:.1f}%")
        for s in SCENARIOS:
            p = row.get(f"p_{s['name']}", 0)
            lines.append(f"  ▪️ {s['name']}: <b>{p:.1f}%</b>")
        lines.append("")
    return "\n".join(lines)


# ================== اجرای اصلی ==================
def main():
    print("🚀 شروع تحلیل...")

    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("❌ توکن یا chat_id تنظیم نشده.")
        return

    all_messages = []
    header = (
        f"📊 <b>تحلیل شباهت کندل ۱۰ ارز برتر</b>\n"
        f"📅 تاریخ مرجع: <b>{TARGET_DATE}</b>\n"
        f"📚 منبع: <b>Yahoo Finance (2019-2024)</b>\n"
        f"{'─' * 30}\n"
    )

    for coin in COINS:
        print(f"\n🔍 تحلیل {coin['name']} ({coin['symbol']})...")
        df = fetch_data(coin["symbol"], DATA_START_DATE, DATA_END_DATE)

        if df is None or len(df) < 200:
            print(f"  ⚠️ داده کافی برای {coin['name']} نیست. رد شد.")
            continue

        similar_df, target_price = find_similar_candles(
            df, TARGET_DATE, top_n=TOP_N_SIMILAR, min_gap_days=MIN_GAP_DAYS
        )

        if similar_df is None or similar_df.empty:
            print(f"  ⚠️ کندل مشابهی برای {coin['name']} پیدا نشد.")
            continue

        prob_df = calculate_outcomes(df, similar_df)
        if prob_df.empty:
            continue

        msg = format_coin_result(coin["name"], coin["symbol"],
                                 similar_df, target_price, prob_df)
        all_messages.append(msg)
        time.sleep(0.5)

    if not all_messages:
        print("❌ هیچ نتیجه‌ای برای ارسال نیست.")
        return

    full_msg = header + "\n\n".join(all_messages)
    full_msg += "\n\n⚠️ <i>این تحلیل صرفاً آماری است و توصیه سرمایه‌گذاری نیست.</i>"

    print("\n" + "=" * 50)
    print(full_msg[:2000] + "...\n(متن کامل ارسال شد)")
    print("=" * 50)

    send_long(TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, full_msg)
    print("✅ ارسال به تلگرام انجام شد.")


if __name__ == "__main__":
    main()
