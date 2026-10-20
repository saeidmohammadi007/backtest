"""
Backtest: MACD cross-up (candle i) + ADX End-Trend X in [i+1, i+3]
on top-20 coins, 4h timeframe.

Signal rule (unchanged):
  1. MACD cross-up on candle i
  2. ADX End-Trend (orange X in "ADX Di+ Di- [Gu5]") in candles [i+1, i+3]

Entry: close of the FIRST candle j in [i+1, i+3] where the X appears
       (no look-ahead: the X is known at that candle's close).
Forward: next LOOKAHEAD candles after j.
Win = TP hit before SL (same-candle tie -> loss).
Fees: FEE_PCT is added to the TP distance.
Baseline: random entries with the same count per coin, same SL/TP.
"""
import os
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import requests

# ============ CONFIG ============
TOP_COINS = [
    "BTC", "ETH", "BNB", "SOL", "XRP",
    "DOGE", "ADA", "AVAX", "TRX", "LINK",
    "DOT", "MATIC", "TON", "SHIB", "LTC",
    "BCH", "UNI", "XLM", "NEAR", "APT",
]

SL_LEVELS = [2, 3, 4, 5]
TP_LEVELS = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
LOOKAHEAD = 100
TRI_AFTER = 3
RANGE_DAYS = int(os.getenv("RANGE_DAYS", "700"))
WARMUP = 100
MIN_CANDLES = 200

FEE_PCT = 0.1
RANDOM_SEED = 42

MACD_FAST, MACD_SLOW, MACD_SIG = 12, 26, 9
ADX_SIG_LEN, DI_LEN, LVL_RANGE, LVL_TREND = 14, 14, 20, 35
TF_SECONDS = 4 * 3600

TG_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TG_CHAT = os.getenv("TELEGRAM_CHAT_ID", "")

YAHOO = "https://query1.finance.yahoo.com/v8/finance/chart"
S = requests.Session()
S.headers["User-Agent"] = "Mozilla/5.0"

OUT_DIR = "results"


# ============ FETCH ============
def get_json(url, params=None, tries=4):
    for i in range(tries):
        try:
            r = S.get(url, params=params, timeout=30)
            if r.status_code == 200:
                return r.json()
            if r.status_code == 404:
                return None
        except Exception:
            pass
        time.sleep(1 + i)
    return None


def to_4h(df: pd.DataFrame, now=None) -> pd.DataFrame:
    now = now or time.time()
    d = df.copy()
    d["dt"] = pd.to_datetime(d["t"], unit="s", utc=True)
    d = d.set_index("dt")
    g = d.resample("4h", origin="epoch").agg(
        {"o": "first", "h": "max", "l": "min", "c": "last", "t": "first"}
    )
    g["n"] = d["c"].resample("4h", origin="epoch").count()
    g = g[g["n"] == 4].dropna()
    g["t"] = ((g.index - pd.Timestamp("1970-01-01", tz="UTC"))
              // pd.Timedelta(seconds=1)).astype(float)
    g = g[g["t"] + TF_SECONDS <= now]
    return g[["t", "o", "h", "l", "c"]].reset_index(drop=True)


def fetch_klines(ticker):
    j = get_json(f"{YAHOO}/{ticker}",
                 {"interval": "1h", "range": f"{RANGE_DAYS}d"})
    try:
        r = j["chart"]["result"][0]
        q = r["indicators"]["quote"][0]
        df = pd.DataFrame({
            "t": r["timestamp"], "o": q["open"], "h": q["high"],
            "l": q["low"], "c": q["close"],
        }).dropna()
        df = df.astype(float).drop_duplicates("t").sort_values("t")
    except Exception:
        return None
    if df.empty:
        return None
    return to_4h(df)


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


def adx_end_trend(df: pd.DataFrame) -> pd.Series:
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

    return pd.Series(end_trend, index=df.index)


def macd_cross_up(df: pd.DataFrame) -> pd.Series:
    c = df["c"]
    macd = c.ewm(span=MACD_FAST, adjust=False).mean() \
        - c.ewm(span=MACD_SLOW, adjust=False).mean()
    sgn = macd.ewm(span=MACD_SIG, adjust=False).mean()
    return (macd.shift(1) <= sgn.shift(1)) & (macd > sgn)


# ============ SIGNAL DETECTION ============
def detect_signals(df: pd.DataFrame):
    """Returns entry candle indices j (first X after the cross)."""
    cross = macd_cross_up(df).to_numpy()
    end_tr = adx_end_trend(df).to_numpy()
    n = len(df)
    signals, dropped = [], 0
    for i in range(WARMUP, n):
        if not cross[i]:
            continue
        window = end_tr[i + 1:min(n, i + TRI_AFTER + 1)]
        hits = np.where(window)[0]
        if len(hits) == 0:
            continue
        j = i + 1 + int(hits[0])
        if j + LOOKAHEAD >= n:
            dropped += 1
            continue
        signals.append(j)
    return signals, dropped


# ============ BACKTEST ============
def backtest_coin(df: pd.DataFrame, signals):
    outcomes = {(sl, tp): [] for sl in SL_LEVELS for tp in TP_LEVELS}
    for j in signals:
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
    return outcomes, len(signals)


def random_baseline(df: pd.DataFrame, n_signals: int, rng):
    empty = {(sl, tp): [] for sl in SL_LEVELS for tp in TP_LEVELS}
    lo_i, hi_i = WARMUP, len(df) - LOOKAHEAD - 1
    if n_signals == 0 or hi_i <= lo_i:
        return empty
    idx = rng.choice(np.arange(lo_i, hi_i), size=n_signals, replace=True)
    out, _ = backtest_coin(df, [int(x) for x in idx])
    return out


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
    print(f"config: RANGE_DAYS={RANGE_DAYS} LOOKAHEAD={LOOKAHEAD} "
          f"TRI_AFTER={TRI_AFTER} FEE={FEE_PCT}% coins={len(TOP_COINS)}",
          flush=True)

    rng = np.random.default_rng(RANDOM_SEED)
    keys = [(sl, tp) for sl in SL_LEVELS for tp in TP_LEVELS]
    all_outcomes = {k: [] for k in keys}
    rand_outcomes = {k: [] for k in keys}
    total_signals = 0
    total_dropped = 0
    per_coin = {}

    for coin in TOP_COINS:
        print(f"[{coin}] fetching...", flush=True)
        df = fetch_klines(f"{coin}-USD")
        if df is None or len(df) < MIN_CANDLES:
            n = 0 if df is None else len(df)
            print(f"[{coin}] skipped (candles={n})", flush=True)
            per_coin[coin] = {"signals": 0, "dropped": 0,
                              "candles": n, "skipped": True}
            time.sleep(0.5)
            continue

        signals, dropped = detect_signals(df)
        outcomes, n_valid = backtest_coin(df, signals)
        rb = random_baseline(df, n_valid, rng)

        per_coin[coin] = {"signals": n_valid, "dropped": dropped,
                          "candles": len(df), "skipped": False}
        total_signals += n_valid
        total_dropped += dropped

        for k in keys:
            all_outcomes[k].extend(outcomes[k])
            rand_outcomes[k].extend(rb[k])

        print(f"[{coin}] candles={len(df)} signals={n_valid} "
              f"dropped_no_future={dropped}", flush=True)
        time.sleep(0.5)

    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    lines = [
        "# Backtest — MACD cross-up + ADX End-Trend X (4h)",
        f"**Entry:** close of first X in `[i+1, i+{TRI_AFTER}]` after the cross",
        f"**Forward window:** {LOOKAHEAD} candles | **Fee:** {FEE_PCT}% "
        f"| **Tie rule:** same-candle TP&SL -> loss",
        f"Generated: {ts}",
        f"Coins: {len(TOP_COINS)} | Signals: {total_signals} "
        f"| Dropped (no future data): {total_dropped}",
        "",
        md_table(all_outcomes, "Win rate % (signal)"),
        "",
        md_table(rand_outcomes, "Win rate % (random entries, same count)"),
        "",
        md_diff_table(all_outcomes, rand_outcomes),
        "",
        "## Per-coin signals",
    ]
    for coin, info in per_coin.items():
        if info["skipped"]:
            lines.append(f"- **{coin}**: _skipped (candles={info['candles']})_")
        else:
            lines.append(f"- **{coin}**: signals={info['signals']} "
                         f"· dropped={info['dropped']} · candles={info['candles']}")
    md = "\n".join(lines)
    print("\n" + md, flush=True)

    os.makedirs(OUT_DIR, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    with open(f"{OUT_DIR}/backtest_{stamp}.md", "w", encoding="utf-8") as f:
        f.write(md)
    with open(f"{OUT_DIR}/latest.md", "w", encoding="utf-8") as f:
        f.write(md)

    tg_msg = (
        f"📊 <b>Backtest</b> — MACD↑ + End-Trend ✕ (entry at ✕, 4h)\n"
        f"Forward: {LOOKAHEAD} bars | Fee: {FEE_PCT}%\n"
        f"Coins: {len(TOP_COINS)} | Signals: {total_signals} "
        f"| Dropped: {total_dropped}\n\n"
        f"<b>Win rate %</b>\n<pre>{mono_table(all_outcomes)}</pre>\n"
        f"<b>Signal − Random (pp)</b>\n"
        f"<pre>{mono_table(all_outcomes, rand_outcomes)}</pre>"
    )
    telegram(tg_msg)


if __name__ == "__main__":
    main()
