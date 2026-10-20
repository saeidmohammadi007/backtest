"""
ADX + Alligator | همه‌کاره: اسکنر + بک‌تست (ماتریس هدف/ریزش + SL/TP با آموزش/تست)

اجرا:
  python main.py scan                  # اسکن آخرین کندل بسته‌شده ۴ ساعته
  python main.py backtest              # بک‌تست کامل، خروجی report.txt
  python main.py backtest --telegram   # ارسال گزارش به تلگرام
  python main.py scan --telegram       # ارسال سیگنال‌ها به تلگرام

متغیرهای محیطی برای تلگرام: TELEGRAM_TOKEN و CHAT_ID
نیازمندی‌ها: ccxt pandas numpy requests
"""
import os
import sys
import argparse

import ccxt
import numpy as np
import pandas as pd
import requests

# ================== تنظیمات ==================
EXCHANGE_ID = "lbank"
TIMEFRAME = "4h"
TOP_N = 20                      # تعداد ارزهای برتر (فقط موجود در صرافی)

# شرط سیگنال
HL_TREND = 35
HL_RANGE = 20
BULLISH_ALLIGATOR = True        # True: lips > teeth > jaw
ONLY_NEW_SIGNALS = True         # فقط کندل شروع سیگنال

# بک‌تست
LOOKAHEAD = 100                 # کندل
TARGETS_UP = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
DRAWDOWN_LEVELS = [2, 3, 4, 5, 6, 7, 8, 9, 10]
TP_LIST = [10, 15, 20, 30, 40, 50]
SL_LIST = [2, 3, 4, 5, 6, 8, 10]
FEE = 0.001                     # کارمزد هر طرف
SLIP = 0.001                    # لغزش هر طرف
TRAIN_FRAC = 0.6
MIN_TRADES = 30

STABLECOINS = {
    "USDT", "USDC", "USDS", "DAI", "BUSD", "TUSD", "FDUSD", "USDD", "USDE",
    "PYUSD", "FRAX", "LUSD", "GUSD", "USTC", "USD1", "RLUSD", "EURC", "EURT",
    "XAUT", "PAXG", "SUSDE", "BUIDL", "USDTB", "USDF", "BSC-USD",
    # wrapped / staked
    "WBTC", "STETH", "WSTETH", "WETH", "WEETH", "WBETH", "CBBTC",
}

exchange = getattr(ccxt, EXCHANGE_ID)({
    "enableRateLimit": True,
    "options": {"defaultType": "spot"},
})


# ================== تلگرام ==================
def send_telegram(text):
    token, chat_id = os.getenv("TELEGRAM_TOKEN"), os.getenv("CHAT_ID")
    if not token or not chat_id:
        print("⚠️ TELEGRAM_TOKEN / CHAT_ID تنظیم نشده؛ پیام ارسال نشد.")
        return
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    # محدودیت ۴۰۹۶ کاراکتر؛ تکه‌تکه می‌فرستیم
    chunks, cur = [], ""
    for line in text.split("\n"):
        if len(cur) + len(line) + 1 > 3800:
            chunks.append(cur)
            cur = ""
        cur += line + "\n"
    if cur.strip():
        chunks.append(cur)
    for ch in chunks:
        try:
            r = requests.post(url, data={"chat_id": chat_id, "text": ch}, timeout=30)
            if not r.ok:
                print(f"❌ تلگرام: {r.status_code} {r.text[:200]}")
        except Exception as e:
            print(f"❌ تلگرام: {e}")


# ================== لیست ارزها ==================
def get_stable_symbols():
    try:
        r = requests.get(
            "https://api.coingecko.com/api/v3/coins/markets",
            params={"vs_currency": "usd", "category": "stablecoins",
                    "order": "market_cap_desc", "per_page": 250, "page": 1},
            timeout=30)
        r.raise_for_status()
        return {c["symbol"].upper() for c in r.json()}
    except Exception:
        return set()


def get_top_coins():
    """۲۰ ارز برتر بازار (بدون استیبل و wrapped) که در صرافی جفت USDT دارند."""
    stable = STABLECOINS | get_stable_symbols()
    r = requests.get(
        "https://api.coingecko.com/api/v3/coins/markets",
        params={"vs_currency": "usd", "order": "market_cap_desc",
                "per_page": 100, "page": 1, "sparkline": "false"},
        timeout=30)
    r.raise_for_status()
    coins = []
    for c in r.json():
        sym = c["symbol"].upper()
        if sym in stable:
            continue
        if f"{sym}/USDT" not in exchange.markets:
            continue
        coins.append(sym)
        if len(coins) >= TOP_N:
            break
    return coins


# ================== داده ==================
def _to_df(ohlcv):
    df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = df[col].astype(float)
    return df


def resample_1h_to_4h(df_1h):
    if df_1h is None or len(df_1h) < 4:
        return None
    df = df_1h.copy()
    df["bucket"] = df["timestamp"].dt.floor("4h")
    counts = df.groupby("bucket").size()
    df = df[df["bucket"].isin(counts[counts == 4].index)]
    if len(df) < 4:
        return None
    agg = df.groupby("bucket").agg(
        open=("open", "first"), high=("high", "max"), low=("low", "min"),
        close=("close", "last"), volume=("volume", "sum"),
    ).reset_index().rename(columns={"bucket": "timestamp"})
    return agg


def get_ohlc_4h(sym, limit=1000):
    """۴h مستقیم؛ اگر نشد 1h→4h. کندل آخرِ نابسته حذف می‌شود."""
    symbol = f"{sym}/USDT"
    if symbol not in exchange.markets:
        return None, "نماد در صرافی نیست"
    try:
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe="4h", limit=limit)
        if ohlcv and len(ohlcv) >= 60:
            return _to_df(ohlcv).iloc[:-1].reset_index(drop=True), "4h مستقیم"
    except Exception:
        pass
    try:
        o1 = exchange.fetch_ohlcv(symbol, timeframe="1h", limit=2000)
    except Exception as e:
        return None, f"خطا: {e}"
    if not o1 or len(o1) < 50:
        return None, "1h هم نشد"
    df4 = resample_1h_to_4h(_to_df(o1))
    if df4 is None or len(df4) < 60:
        return None, "تبدیل 1h→4h کافی نبود"
    return df4, "1h→4h"


# ================== اندیکاتورها ==================
def rma(s, n):
    return s.ewm(alpha=1.0 / n, adjust=False).mean()


def calc_adx(df, di_len=14, adx_len=14):
    high, low, close = df["high"], df["low"], df["close"]
    up, down = high.diff(), -low.diff()
    plus_dm = np.where((up > down) & (up > 0), up, 0.0)
    minus_dm = np.where((down > up) & (down > 0), down, 0.0)
    tr = pd.concat([high - low, (high - close.shift()).abs(),
                    (low - close.shift()).abs()], axis=1).max(axis=1)
    atr = rma(tr, di_len)
    plus_di = 100 * rma(pd.Series(plus_dm, index=df.index), di_len) / atr
    minus_di = 100 * rma(pd.Series(minus_dm, index=df.index), di_len) / atr
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    df["plus_di"], df["minus_di"], df["adx"] = plus_di, minus_di, rma(dx, adx_len)
    return df


def calc_alligator(df):
    median = (df["high"] + df["low"]) / 2
    df["jaw"] = rma(median, 13).shift(8)
    df["teeth"] = rma(median, 8).shift(5)
    df["lips"] = rma(median, 5).shift(3)
    return df


def detect_signals(df):
    adx_ok = ((df["adx"] > HL_TREND)
              & (df["adx"] > df["adx"].shift(1))
              & (df["plus_di"] > df["minus_di"])
              & (df["plus_di"] >= HL_TREND))
    if BULLISH_ALLIGATOR:
        all_ok = (df["lips"] > df["teeth"]) & (df["teeth"] > df["jaw"])
    else:
        all_ok = (df["jaw"] > df["teeth"]) & (df["teeth"] > df["lips"])
    raw = (adx_ok & all_ok & (df["adx"] > HL_RANGE)).fillna(False)
    if ONLY_NEW_SIGNALS:
        raw = raw & ~raw.shift(1, fill_value=False)
    df["signal"] = raw
    return df


def prepare(sym):
    df, src = get_ohlc_4h(sym)
    if df is None or len(df) < 60:
        return None, src
    return detect_signals(calc_alligator(calc_adx(df))), src


# ================== ماتریس هدف/ریزش ==================
def analyze_matrix(df):
    hit = {t: {d: 0 for d in DRAWDOWN_LEVELS} for t in TARGETS_UP}
    reached = {t: 0 for t in TARGETS_UP}
    total = 0
    for s in df.index[df["signal"]].tolist():
        if s + LOOKAHEAD >= len(df):
            continue
        entry = df.loc[s, "close"]
        if entry <= 0:
            continue
        future = df.loc[s + 1: s + LOOKAHEAD]
        total += 1
        for t in TARGETS_UP:
            mask = (future["high"] >= entry * (1 + t / 100.0)).values
            if not mask.any():
                continue
            reached[t] += 1
            seg = future.iloc[: int(mask.argmax()) + 1]
            dd = (entry - seg["low"].min()) / entry * 100
            for d in DRAWDOWN_LEVELS:
                if dd < d:
                    hit[t][d] += 1
    return {"total": total, "reached": reached, "matrix": hit}


def matrix_table(res):
    total = res["total"]
    if total == 0:
        return "  (بدون سیگنال)"
    header = "هدف/ریزش |" + "|".join(f"<{d}٪".rjust(5) for d in DRAWDOWN_LEVELS)
    lines = [header, "-" * len(header)]
    for t in TARGETS_UP:
        row = [f"{t:>3}٪     |"]
        row += [f"{res['matrix'][t][d] / total * 100:>4.0f}٪" for d in DRAWDOWN_LEVELS]
        lines.append("|".join(row))
    lines.append("")
    lines.append("رسیده فقط: " + " | ".join(
        f"{t}٪→{res['reached'][t] / total * 100:.0f}٪" for t in TARGETS_UP))
    return "\n".join(lines)


# ================== شبیه‌سازی SL/TP ==================
def simulate(df, tp, sl):
    """ورود: open کندل بعد؛ اگر SL و TP هم‌کندل شدند SL؛ بدون هم‌پوشانی؛ خروج زمانی بعد از LOOKAHEAD."""
    trades, n, last_exit = [], len(df), -1
    for s in df.index[df["signal"]].tolist():
        e = s + 1
        if e >= n or e <= last_exit or e + LOOKAHEAD > n - 1:
            continue
        entry = df.loc[e, "open"] * (1 + SLIP)
        if entry <= 0:
            continue
        tp_p, sl_p = entry * (1 + tp / 100), entry * (1 - sl / 100)
        exit_p, exit_i = None, None
        for j in range(e, e + LOOKAHEAD + 1):
            o, h, l = df.loc[j, "open"], df.loc[j, "high"], df.loc[j, "low"]
            if l <= sl_p:
                exit_p, exit_i = min(o, sl_p) * (1 - SLIP), j
                break
            if h >= tp_p:
                exit_p, exit_i = tp_p * (1 - SLIP), j
                break
        if exit_p is None:
            exit_i = e + LOOKAHEAD
            exit_p = df.loc[exit_i, "close"] * (1 - SLIP)
        trades.append({"time": df.loc[e, "timestamp"], "ret": exit_p / entry - 1 - 2 * FEE})
        last_exit = exit_i
    return trades


def stats(rets):
    n = len(rets)
    if n == 0:
        return {"n": 0, "win": 0.0, "avg": 0.0, "pf": 0.0}
    wins, losses = rets[rets > 0], rets[rets <= 0]
    gl = -losses.sum()
    return {"n": n, "win": len(wins) / n * 100, "avg": rets.mean() * 100,
            "pf": wins.sum() / gl if gl > 0 else float("inf")}


# ================== حالت‌ها ==================
def load_all(coins):
    dfs, src_info = {}, {}
    for sym in coins:
        try:
            df, src = prepare(sym)
        except Exception as e:
            print(f"❌ {sym}: {e}")
            continue
        if df is None:
            print(f"⚠️ {sym}: {src}")
            continue
        dfs[sym], src_info[sym] = df, src
        print(f"✅ {sym}: {int(df['signal'].sum())} سیگنال ({len(df)} کندل، {src})")
    return dfs, src_info


def run_backtest(use_telegram):
    coins = get_top_coins()
    print(f"ارزها: {', '.join(coins)}\n")
    dfs, src_info = load_all(coins)
    if not dfs:
        print("داده‌ای نیست.")
        return

    # --- بخش ۱: ماتریس هدف/ریزش ---
    lines = ["=" * 70, f"ADX + Alligator | {TIMEFRAME} | {EXCHANGE_ID.upper()}", "=" * 70, ""]
    g_m = {t: {d: 0 for d in DRAWDOWN_LEVELS} for t in TARGETS_UP}
    g_r = {t: 0 for t in TARGETS_UP}
    g_total = 0
    for sym, df in dfs.items():
        res = analyze_matrix(df)
        if res["total"] == 0:
            continue
        lines += [f"---- {sym} | {res['total']} سیگنال | {src_info[sym]} ----", matrix_table(res), ""]
        g_total += res["total"]
        for t in TARGETS_UP:
            g_r[t] += res["reached"][t]
            for d in DRAWDOWN_LEVELS:
                g_m[t][d] += res["matrix"][t][d]
    lines += ["=" * 70, f"خلاصه کل | سیگنال کل: {g_total}", "=" * 70,
              matrix_table({"total": g_total, "reached": g_r, "matrix": g_m}), ""]

    # --- بخش ۲: SL/TP با آموزش/تست ---
    t_min = min(d["timestamp"].min() for d in dfs.values())
    t_max = max(d["timestamp"].max() for d in dfs.values())
    cutoff = t_min + (t_max - t_min) * TRAIN_FRAC
    rows = []
    for tp in TP_LIST:
        for sl in SL_LIST:
            tr = []
            for df in dfs.values():
                tr += simulate(df, tp, sl)
            if not tr:
                continue
            tdf = pd.DataFrame(tr)
            a = stats(tdf.loc[tdf["time"] < cutoff, "ret"])
            b = stats(tdf.loc[tdf["time"] >= cutoff, "ret"])
            rows.append({"TP": tp, "SL": sl, "a": a, "b": b})
    rows = [r for r in rows if r["a"]["n"] >= MIN_TRADES]
    rows.sort(key=lambda r: r["a"]["avg"], reverse=True)

    lines += ["=" * 78,
              f"SL/TP | کارمزد+لغزش هر طرف: {(FEE + SLIP) * 100:.2f}٪ | "
              f"آموزش: {t_min:%Y-%m-%d}→{cutoff:%Y-%m-%d} | تست: →{t_max:%Y-%m-%d}",
              "=" * 78]
    hdr = (f"{'TP/SL':>8} | {'آموزش n':>8} {'win%':>6} {'avg%':>7} {'PF':>5}"
           f" | {'تست n':>6} {'win%':>6} {'avg%':>7} {'PF':>5}")
    lines += [hdr, "-" * len(hdr)]
    for r in rows[:10]:
        a, b = r["a"], r["b"]
        lines.append(f"{r['TP']:>4}/{r['SL']:<3} | {a['n']:>8} {a['win']:>6.1f} {a['avg']:>7.2f} {a['pf']:>5.2f}"
                     f" | {b['n']:>6} {b['win']:>6.1f} {b['avg']:>7.2f} {b['pf']:>5.2f}")
    if not rows:
        lines.append(f"هیچ ترکیبی حداقل {MIN_TRADES} معامله در آموزش نداشت.")
    lines += ["", "avg% = میانگین سود هر معامله بعد از کارمزد | PF = سود کل ÷ ضرر کل",
              "اگر avg% تست منفی یا خیلی کمتر از آموزش است، ترکیب احتمالاً overfit بوده."]

    report = "\n".join(lines)
    print("\n" + report)
    with open("report.txt", "w", encoding="utf-8") as f:
        f.write(report)
    if use_telegram:
        send_telegram(report)


def run_scan(use_telegram):
    coins = get_top_coins()
    print(f"ارزها: {', '.join(coins)}\n")
    found = []
    for sym in coins:
        try:
            df, src = prepare(sym)
        except Exception as e:
            print(f"❌ {sym}: {e}")
            continue
        if df is None:
            print(f"⚠️ {sym}: {src}")
            continue
        last = df.iloc[-1]
        if bool(last["signal"]):
            found.append(f"🟢 {sym}/USDT | close={last['close']:.6g} | ADX={last['adx']:.1f} "
                         f"| +DI={last['plus_di']:.1f} | کندل: {last['timestamp']:%Y-%m-%d %H:%M} UTC")
            print(found[-1])
        else:
            print(f"— {sym}: سیگنال ندارد")

    if found:
        msg = f"📡 ADX + Alligator | {TIMEFRAME} | {EXCHANGE_ID.upper()}\n\n" + "\n".join(found)
        if use_telegram:
            send_telegram(msg)
    else:
        print("\nسیگنالی در آخرین کندل بسته‌شده نبود.")


def main():
    ap = argparse.ArgumentParser(description="ADX + Alligator scanner/backtester")
    ap.add_argument("mode", choices=["scan", "backtest"], nargs="?", default="scan")
    ap.add_argument("--telegram", action="store_true", help="ارسال خروجی به تلگرام")
    args = ap.parse_args()

    print(f"بارگذاری بازارهای {EXCHANGE_ID.upper()}...")
    try:
        exchange.load_markets()
    except Exception as e:
        print(f"❌ خطا: {e}")
        sys.exit(1)

    # در GitHub Actions اگر توکن تنظیم شده باشد، خودکار ارسال می‌کنیم
    use_tg = args.telegram or (args.mode == "scan" and bool(os.getenv("TELEGRAM_TOKEN")))
    if args.mode == "scan":
        run_scan(use_tg)
    else:
        run_backtest(use_tg)


if __name__ == "__main__":
    main()
