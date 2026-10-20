"""
Backtest: MACD cross-up (candle i) + ADX End-Trend X in [i+1, i+TRI_AFTER]
           + at least one small red triangle (shortE) in [j-10, j-1]
on top-100 coins, 4h timeframe, data from LBank SPOT via ccxt.
"""
import os
import time
from datetime import datetime, timezone

import ccxt
import numpy as np
import pandas as pd
import requests

# ============ CONFIG ============
TOP_COINS = [
    # --- Top 20 ---
    "BTC", "ETH", "BNB", "SOL", "XRP",
    "DOGE", "ADA", "AVAX", "TRX", "LINK",
    "DOT", "MATIC", "TON", "SHIB", "LTC",
    "BCH", "UNI", "XLM", "NEAR", "APT",
    # --- 21..40 ---
    "ICP", "FIL", "ETC", "HBAR", "ATOM",
    "IMX", "INJ", "OP", "ARB", "SUI",
    "STX", "GRT", "RNDR", "VET", "MKR",
    "LDO", "THETA", "ALGO", "FTM", "AAVE",
    # --- 41..60 ---
    "EGLD", "FLOW", "SAND", "MANA", "AXS",
    "EOS", "XTZ", "KAVA", "SNX", "CRV",
    "COMP", "ZEC", "DASH", "ENJ", "CHZ",
    "BAT", "ONE", "ZIL", "IOTA", "NEO",
    # --- 61..80 ---
    "QTUM", "OMG", "WAVES", "KSM", "RUNE",
    "CAKE", "1INCH", "SUSHI", "YFI", "BAL",
    "UMA", "KNC", "REN", "STORJ", "ANKR",
    "SKL", "OCEAN", "FET", "AGIX", "ROSE",
    # --- 81..100 ---
    "CELO", "AR", "MASK", "DYDX", "GMX",
    "ENS", "PEPE", "WLD", "SEI", "TIA",
    "JUP", "PYTH", "ORDI", "JTO", "BONK",
    "WIF", "BLUR", "MEME", "GALA", "APE",
]

SL_LEVELS = [2, 3, 4, 5]
TP_LEVELS = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
LOOKAHEAD = 100
TRI_AFTER = 3

# شرط جدید: مثلث قرمز کوچک در [j-10, j-1]
RED_TRI_BEFORE_MIN = 1
RED_TRI_BEFORE_MAX = 10

RANGE_DAYS = int(os.getenv("RANGE_DAYS", "700"))
WARMUP = 100
MIN_CANDLES = 200

FEE_PCT = 0.1
RANDOM_SEED = 42
TG_LIST_MAX = 40

MACD_FAST, MACD_SLOW, MACD_SIG = 12, 26, 9
ADX_SIG_LEN, DI_LEN, LVL_RANGE, LVL_TREND = 14, 14, 20, 35
TF_SECONDS = 4 * 3600

# تاخیر بین ارزها برای جلوگیری از rate-limit
SLEEP_BETWEEN_COINS = float(os.getenv("SLEEP_BETWEEN_COINS", "0.5"))

TG_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TG_CHAT = os.getenv("TELEGRAM_CHAT_ID", "")

OUT_DIR = "results"
EX = ccxt.lbank({"enableRateLimit": True})


# ============ FETCH ============
def fetch_klines(coin):
    symbol = f"{coin}/USDT"
    tf_ms = TF_SECONDS * 1000
    since = EX.milliseconds() - RANGE_DAYS * 86400 * 1000
    rows, prev_last = [], None
    for _ in range(10):
        batch = None
        for attempt in range(3):
            try:
                batch = EX.fetch_ohlcv(symbol, "4h", since=since, limit=2000)
                break
            except ccxt.BadSymbol:
                return None
            except Exception as e:
                print(f"[{coin}] fetch error ({attempt + 1}/3): {e}", flush=True)
                time.sleep(1 + attempt)
        if not batch:
            break
        rows.extend(batch)
        last = batch[-1][0]
        if prev_last is not None and last <= prev_last:
            break
        prev_last = last
        if last + tf_ms >= EX.milliseconds():
            break
        since = last + tf_ms

    if not rows:
        return None
    df = pd.DataFrame(rows, columns=["t", "o", "h", "l", "c", "v"])
    df["t"] = df["t"] / 1000.0
    df = df[["t", "o", "h", "l", "c"]].astype(float).dropna()
    df = df.drop_duplicates("t").sort_values("t")
    df = df[df["t"] + TF_SECONDS <= time.time()]
    return df.reset_index(drop=True) if not df.empty else None


# ============ INDICATORS ============
def rma(x: pd.Series, n: int) -> pd.Series:
    v = x.to_numpy(dtype=float)
    out = np.full(len(v), np.nan)
    valid = np.where(~np.isnan(v))[0]
    if len(valid) < n:
        return pd.Series(out, index=x.index)
    s = valid[0] + n - 1
    out[s] = np.nanmean(v[valid[0]:s + 1])
    for i in range(s + 1, len(v)):
        out[i] = (out[i - 1] * (n - 1) + v[i]) / n if not np.isnan(v[i]) else np.nan
    return pd.Series(out, index=x.index)


def adx_end_trend(df: pd.DataFrame):
    """Returns (end_trend, shortE) as bool Series."""
    h, l, c = df["h"], df["l"], df["c"]
    up = h.diff()
    down = -l.diff()
    plus_dm = np.where(up.isna(), np.nan,
                       np.where((up > down) & (up > 0), up, 0.0))
    minus_dm = np.where(down.isna(), np.nan,
                        np.where((down > up) & (down > 0), down, 0.0))
    pc = c.shift(1)
    tr = pd.concat([h - l, (h - pc).abs(), (l - pc).abs()],
                   axis=1).max(axis=1, skipna=False)
    tr_r = rma(tr, DI_LEN)
    plus = (100 * rma(pd.Series(plus_dm, index=df.index), DI_LEN) / tr_r).ffill()
    minus = (100 * rma(pd.Series(minus_dm, index=df.index), DI_LEN) / tr_r).ffill()
    sm = plus + minus
    sig = 100 * rma((plus - minus).abs() / sm.where(sm != 0, 1), ADX_SIG_LEN)

    n = len(df)
    P, M, G = plus.to_numpy(), minus.to_numpy(), sig.to_numpy()
    end_trend = np.zeros(n, dtype=bool)
    shortE_arr = np.zeros(n, dtype=bool)
    cond = 0.0

    for i in range(1, n):
        if np.isnan(G[i]) or np.isnan(G[i - 1]) or np.isnan(P[i]) or np.isnan(M[i]):
            continue
        hl = G[i] <= LVL_RANGE
        hl_prev = G[i - 1] <= LVL_RANGE
        di_up = P[i] >= M[i]
        di_dn = M[i] > P[i]
        di_up_prev = (P[i - 1] >= M[i - 1]) if not np.isnan(M[i - 1]) else False
        di_dn_prev = (M[i - 1] > P[i - 1]) if not np.isnan(M[i - 1]) else False
        sig_up = G[i] > G[i - 1]
        cross = (P[i] > M[i] and P[i - 1] <= M[i - 1]) or \
                (P[i] < M[i] and P[i - 1] >= M[i - 1])

        entryLong = (not hl and di_up and sig_up and not di_up_prev) or \
                    (not hl and di_up and sig_up and G[i] > LVL_RANGE and hl_prev)
        entryShort = (not hl and di_dn and sig_up and not di_dn_prev) or \
                     (not hl and di_dn and sig_up and G[i] > LVL_RANGE and hl_prev)
        entryLongStr = not hl and di_up and sig_up and P[i] >= LVL_TREND
        entryShortSt = not hl and di_dn and sig_up and M[i] > LVL_TREND
        exitLong = (cross and di_up_prev) or (hl and not hl_prev)
        exitShort = (cross and di_dn_prev) or (hl and not hl_prev)

        shortE_arr[i] = entryShort        # ← مثلث قرمز کوچک

        prev = cond
        if prev != 1 and entryLongStr:
            cond = 1
        elif prev != -1 and entryShortSt:
            cond = -1
        elif prev != 0.5 and entryLong:
            cond = 0.5
        elif prev != -0.5 and entryShort:
            cond = -0.5
        elif prev != 0 and exitLong:
            cond = 0
        elif prev != 0 and exitShort:
            cond = 0

        end_trend[i] = (prev != 0) and (cond == 0)

    return (pd.Series(end_trend, index=df.index),
            pd.Series(shortE_arr, index=df.index))


def macd_cross_up(df: pd.DataFrame) -> pd.Series:
    c = df["c"]
    macd = c.ewm(span=MACD_FAST, adjust=False).mean() \
        - c.ewm(span=MACD_SLOW, adjust=False).mean()
    sgn = macd.ewm(span=MACD_SIG, adjust=False).mean()
    return (macd.shift(1) <= sgn.shift(1)) & (macd > sgn)


# ============ SIGNAL DETECTION ============
def detect_signals(df: pd.DataFrame):
    cross = macd_cross_up(df).to_numpy()
    end_tr_series, shortE_series = adx_end_trend(df)
    end_tr = end_tr_series.to_numpy()
    shortE = shortE_series.to_numpy()
    n = len(df)
    signals, dropped, no_red_tri = [], 0, 0

    for i in range(WARMUP, n):
        if not cross[i]:
            continue
        window = end_tr[i + 1:min(n, i + TRI_AFTER + 1)]
        hits = np.where(window)[0]
        if len(hits) == 0:
            continue
        j = i + 1 + int(hits[0])

        # شرط جدید: مثلث قرمز کوچک در [j-10, j-1]
        lo = max(0, j - RED_TRI_BEFORE_MAX)
        hi = j - RED_TRI_BEFORE_MIN + 1
        if hi <= lo or not shortE[lo:hi].any():
            no_red_tri += 1
            continue

        if j + LOOKAHEAD >= n:
            dropped += 1
            continue
        signals.append((i, j))
    return signals, dropped, no_red_tri


def fmt_ts(t):
    return datetime.fromtimestamp(float(t), tz=timezone.utc).strftime("%Y-%m-%d %H:%M")


def describe_signals(coin, df, signals):
    rows = []
    for i, j in signals:
        rows.append({
            "coin": coin,
            "cross_candle_open": fmt_ts(df["t"].iloc[i]),
            "x_candle_open": fmt_ts(df["t"].iloc[j]),
            "entry_time": fmt_ts(df["t"].iloc[j] + TF_SECONDS),
            "entry_price": float(df["c"].iloc[j]),
        })
    return rows


# ============ BACKTEST ============
def backtest_coin(df: pd.DataFrame, entries):
    outcomes = {(sl, tp): [] for sl in SL_LEVELS for tp in TP_LEVELS}
    for j in entries:
        entry = float(df["c"].iloc[j])
        hi = df["h"].iloc[j + 1:j + 1 + LOOKAHEAD].to_numpy(dtype=float)
        lo = df["l"].iloc[j + 1:j + 1 + LOOKAHEAD].to_numpy(dtype=float)
        run_hi = np.maximum.accumulate(hi)
        run_lo = np.minimum.accumulate(lo)

        sl_first = {}
        for sl in SL_LEVELS:
            mask = run_lo <= entry * (1 - sl / 100)
            sl_first[sl] = int(np.argmax(mask)) if mask.any() else None

        for tp in TP_LEVELS:
            mask = run_hi >= entry * (1 + (tp + FEE_PCT) / 100)
            tp_first = int(np.argmax(mask)) if mask.any() else None
            for sl in SL_LEVELS:
                sl_idx = sl_first[sl]
                win = (tp_first is not None and
                       (sl_idx is None or tp_first < sl_idx))
                outcomes[(sl, tp)].append(win)
    return outcomes, len(entries)


def random_baseline(df: pd.DataFrame, n_signals: int, rng):
    empty = {(sl, tp): [] for sl in SL_LEVELS for tp in TP_LEVELS}
    lo_i, hi_i = WARMUP, len(df) - LOOKAHEAD - 1
    if n_signals == 0 or hi_i <= lo_i:
        return empty
    idx = rng.choice(np.arange(lo_i, hi_i), size=n_signals, replace=True)
    out, _ = backtest_coin(df, [int(x) for x in idx])
    return out


# ============ COIN FILTER ============
def filter_available_coins(coins):
    """فقط ارزهایی که در LBank Spot موجود هستند را برمی‌گرداند."""
    try:
        markets = EX.load_markets()
        available = set()
        for sym, m in markets.items():
            if m.get("spot") and m.get("quote") == "USDT" and m.get("active"):
                available.add(m["base"])
        filtered = [c for c in coins if c in available]
        missing = [c for c in coins if c not in available]
        print(f"[filter] {len(filtered)}/{len(coins)} coins available on LBank spot",
              flush=True)
        if missing:
            print(f"[filter] missing ({len(missing)}): {missing}", flush=True)
        return filtered
    except Exception as e:
        print(f"[filter] failed to load markets: {e} — using full list", flush=True)
        return coins


# ============ FORMAT ============
def rate(arr):
    return sum(arr) / len(arr) * 100 if arr else float("nan")


def md_table(outcomes, title):
    lines = [f"### {title}",
             "| SL\\TP |" + "|".join(f" {tp}% |" for tp in TP_LEVELS),
             "|-------|" + "|".join("-------|" for _ in TP_LEVELS)]
    for sl in SL_LEVELS:
        row = f"| **{sl}%** |"
        for tp in TP_LEVELS:
            row += f" {rate(outcomes[(sl, tp)]):5.1f} |"
        lines.append(row)
    return "\n".join(lines)


def md_diff_table(sig, rnd):
    lines = ["### Signal minus Random (percentage points)",
             "| SL\\TP |" + "|".join(f" {tp}% |" for tp in TP_LEVELS),
             "|-------|" + "|".join("-------|" for _ in TP_LEVELS)]
    for sl in SL_LEVELS:
        row = f"| **{sl}%** |"
        for tp in TP_LEVELS:
            row += f" {rate(sig[(sl, tp)]) - rate(rnd[(sl, tp)]):+5.1f} |"
        lines.append(row)
    return "\n".join(lines)


def md_signal_list(rows):
    lines = ["## Signals found (UTC)",
             "| # | Coin | Cross candle (open) | X candle (open) | Entry time | Entry price |",
             "|---|------|---------------------|-----------------|------------|-------------|"]
    for n, r in enumerate(rows, 1):
        lines.append(f"| {n} | {r['coin']} | {r['cross_candle_open']} | "
                     f"{r['x_candle_open']} | {r['entry_time']} | "
                     f"{r['entry_price']:.6g} |")
    return "\n".join(lines)


def mono_table(outcomes, rnd=None):
    col_w = 7
    head = "SL\\TP".ljust(6) + "".join(f"{tp}%".rjust(col_w) for tp in TP_LEVELS)
    lines = [head]
    for sl in SL_LEVELS:
        row = f"{sl}%".ljust(6)
        for tp in TP_LEVELS:
            v = rate(outcomes[(sl, tp)])
            if rnd is not None:
                v -= rate(rnd[(sl, tp)])
                row += f"{v:+.1f}".rjust(col_w)
            else:
                row += f"{v:.1f}".rjust(col_w)
        lines.append(row)
    return "\n".join(lines)


def mono_signal_list(rows, limit):
    shown = rows[-limit:]
    lines = [f"{'Coin':<6}{'Entry time (UTC)':<18}{'Price':>12}"]
    for r in shown:
        lines.append(f"{r['coin']:<6}{r['entry_time']:<18}{r['entry_price']:>12.6g}")
    return "\n".join(lines), len(shown)


def telegram(text):
    if not TG_TOKEN or not TG_CHAT:
        print("[telegram] not configured — skipping")
        return
    for i in range(0, len(text), 3800):
        try:
            requests.post(
                f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage",
                data={"chat_id": TG_CHAT, "text": text[i:i + 3800],
                      "parse_mode": "HTML"},
                timeout=20,
            )
        except Exception as e:
            print(f"[telegram] error: {e}")


# ============ MAIN ============
def main():
    coins = filter_available_coins(TOP_COINS)

    print(f"config: LBank spot | RANGE_DAYS={RANGE_DAYS} LOOKAHEAD={LOOKAHEAD} "
          f"TRI_AFTER={TRI_AFTER} RED_TRI=[j-{RED_TRI_BEFORE_MAX},j-{RED_TRI_BEFORE_MIN}] "
          f"FEE={FEE_PCT}% coins={len(coins)}", flush=True)

    rng = np.random.default_rng(RANDOM_SEED)
    keys = [(sl, tp) for sl in SL_LEVELS for tp in TP_LEVELS]
    all_outcomes = {k: [] for k in keys}
    rand_outcomes = {k: [] for k in keys}
    total_signals = 0
    total_dropped = 0
    total_no_red_tri = 0
    per_coin = {}
    found = []

    for idx, coin in enumerate(coins, 1):
        print(f"[{idx}/{len(coins)}] [{coin}] fetching...", flush=True)
        df = fetch_klines(coin)
        if df is None or len(df) < MIN_CANDLES:
            n = 0 if df is None else len(df)
            print(f"[{coin}] skipped (candles={n})", flush=True)
            per_coin[coin] = {"signals": 0, "dropped": 0, "no_red_tri": 0,
                              "candles": n, "skipped": True}
            time.sleep(SLEEP_BETWEEN_COINS)
            continue

        signals, dropped, no_red_tri = detect_signals(df)
        entries = [j for _, j in signals]
        outcomes, n_valid = backtest_coin(df, entries)
        rb = random_baseline(df, n_valid, rng)
        rows = describe_signals(coin, df, signals)
        found.extend(rows)

        per_coin[coin] = {"signals": n_valid, "dropped": dropped,
                          "no_red_tri": no_red_tri,
                          "candles": len(df), "skipped": False}
        total_signals += n_valid
        total_dropped += dropped
        total_no_red_tri += no_red_tri

        for k in keys:
            all_outcomes[k].extend(outcomes[k])
            rand_outcomes[k].extend(rb[k])

        print(f"[{coin}] candles={len(df)} signals={n_valid} "
              f"dropped_no_future={dropped} no_red_tri={no_red_tri}", flush=True)
        for r in rows:
            print(f"    {r['coin']}  cross={r['cross_candle_open']}  "
                  f"entry={r['entry_time']}  price={r['entry_price']:.6g}",
                  flush=True)

        time.sleep(SLEEP_BETWEEN_COINS)

    found.sort(key=lambda r: r["entry_time"])
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    lines = [
        "# Backtest — MACD cross-up + ADX End-Trend ✕ + red triangle before ✕",
        f"**✕ window:** `[i+1, i+{TRI_AFTER}]` | "
        f"**red-triangle window:** `[j-{RED_TRI_BEFORE_MAX}, j-{RED_TRI_BEFORE_MIN}]`",
        f"**Forward:** {LOOKAHEAD} candles | **Fee:** {FEE_PCT}% "
        f"| **Tie rule:** same-candle TP&SL -> loss",
        f"Generated: {ts}",
        f"Coins: {len(coins)} (requested {len(TOP_COINS)}) | "
        f"Signals: {total_signals} "
        f"| Dropped (no future): {total_dropped} "
        f"| Rejected (no red tri): {total_no_red_tri}",
        "",
        md_table(all_outcomes, "Win rate % (signal)"),
        "",
        md_table(rand_outcomes, "Win rate % (random entries)"),
        "",
        md_diff_table(all_outcomes, rand_outcomes),
        "",
        md_signal_list(found),
        "",
        "## Per-coin signals",
    ]
    for coin, info in per_coin.items():
        if info["skipped"]:
            lines.append(f"- **{coin}**: _skipped (candles={info['candles']})_")
        else:
            lines.append(f"- **{coin}**: signals={info['signals']} "
                         f"· dropped={info['dropped']} "
                         f"· no_red_tri={info['no_red_tri']} "
                         f"· candles={info['candles']}")
    md = "\n".join(lines)
    print("\n" + md, flush=True)

    os.makedirs(OUT_DIR, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    with open(f"{OUT_DIR}/backtest_{stamp}.md", "w", encoding="utf-8") as f:
        f.write(md)
    with open(f"{OUT_DIR}/latest.md", "w", encoding="utf-8") as f:
        f.write(md)
    pd.DataFrame(found).to_csv(f"{OUT_DIR}/signals.csv", index=False)

    tg_msg = (
        f"📊 <b>Backtest</b> — MACD↑ + ✕ + red-tri-before-✕ (4h, LBank spot)\n"
        f"✕ in [i+1..i+{TRI_AFTER}] · red △ in [j-{RED_TRI_BEFORE_MAX}..j-{RED_TRI_BEFORE_MIN}]\n"
        f"Forward: {LOOKAHEAD} bars | Fee: {FEE_PCT}%\n"
        f"Coins: {len(coins)} | Signals: {total_signals} | "
        f"Dropped: {total_dropped} | NoRedTri: {total_no_red_tri}\n\n"
        f"<b>Win rate %</b>\n<pre>{mono_table(all_outcomes)}</pre>\n"
        f"<b>Signal − Random (pp)</b>\n"
        f"<pre>{mono_table(all_outcomes, rand_outcomes)}</pre>"
    )
    telegram(tg_msg)

    if found:
        txt, shown = mono_signal_list(found, TG_LIST_MAX)
        telegram(f"🗓 <b>Signals</b> (last {shown} of {len(found)}, UTC)\n"
                 f"<pre>{txt}</pre>")


if __name__ == "__main__":
    main()
