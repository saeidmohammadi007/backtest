"""
Backtest: MACD cross-up + ADX red triangle (1-15 candles BEFORE cross)
on top-20 coins, 4h timeframe.

Signal rule:
  1. MACD cross-up on candle i
  2. ADX red triangle (shortE) in candles [i-15, i-1]
     -> 1 to 15 candles BEFORE the cross
     -> NOT including the cross candle itself
     -> NOT after the cross
  3. At least LOOKAHEAD=100 candles exist AFTER candle i
     (signals too close to end of data are dropped)

For each signal, tests next 100 candles for each (SL, TP) pair.
Win = TP hit before SL (same-candle tie -> loss, conservative).
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
LOOKAHEAD  = 100                    # تعداد کندل بعدی که باید موجود باشد
TRI_BEFORE = 15                     # مثلث: ۱ تا ۱۵ کندل قبل از کراس
RANGE_DAYS = int(os.getenv("RANGE_DAYS", "700"))
WARMUP     = 100                    # warmup برای ADX/MACD
MIN_CANDLES = 200

MACD_FAST, MACD_SLOW, MACD_SIG = 12, 26, 9
ADX_SIG_LEN, DI_LEN, LVL_RANGE, LVL_TREND = 14, 14, 20, 35
TF_SECONDS = 4 * 3600

TG_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TG_CHAT  = os.getenv("TELEGRAM_CHAT_ID", "")

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


def adx_short_triangle(df: pd.DataFrame) -> pd.Series:
    h, l, c = df["h"], df["l"], df["c"]
    up = h.diff()
    down = -l.diff()
    plus_dm  = np.where(up.isna(),   np.nan, np.where((up > down) & (up > 0), up, 0.0))
    minus_dm = np.where(down.isna(), np.nan, np.where((down > up) & (down > 0), down, 0.0))
    pc = c.shift(1)
    tr = pd.concat([h - l, (h - pc).abs(), (l - pc).abs()],
                   axis=1).max(axis=1, skipna=False)
    tr_r = rma(tr, DI_LEN)
    plus  = (100 * rma(pd.Series(plus_dm,  index=df.index), DI_LEN) / tr_r).ffill()
    minus = (100 * rma(pd.Series(minus_dm, index=df.index), DI_LEN) / tr_r).ffill()
    sm = plus + minus
    sig = 100 * rma((plus - minus).abs() / sm.where(sm != 0, 1), ADX_SIG_LEN)

    n = len(df)
    P, M, G = plus.to_numpy(), minus.to_numpy(), sig.to_numpy()
    short_e = np.zeros(n, dtype=bool)
    cond = 0.0
    for i in range(1, n):
        if np.isnan(G[i]) or np.isnan(G[i - 1]) or np.isnan(P[i]) or np.isnan(M[i]):
            continue
        hl       = G[i]     <= LVL_RANGE
        hl_prev  = G[i - 1] <= LVL_RANGE
        di_up    = P[i] >= M[i]
        di_dn    = M[i] >  P[i]
        di_dn_prev = (M[i - 1] >  P[i - 1]) if not np.isnan(M[i - 1]) else False
        di_up_prev = (P[i - 1] >= M[i - 1]) if not np.isnan(M[i - 1]) else False
        sig_up = G[i] > G[i - 1]
        cross = (P[i] > M[i] and P[i - 1] <= M[i - 1]) or \
                (P[i] < M[i] and P[i - 1] >= M[i - 1])

        entry_long  = (not hl and di_up and sig_up and not di_up_prev) or \
                      (not hl and di_up and sig_up and G[i] > LVL_RANGE and hl_prev)
        entry_short = (not hl and di_dn and sig_up and not di_dn_prev) or \
                      (not hl and di_dn and sig_up and G[i] > LVL_RANGE and hl_prev)
        entry_long_str = not hl and di_up and sig_up and P[i] >= LVL_TREND
        entry_short_st = not hl and di_dn and sig_up and M[i] >  LVL_TREND
        exit_long  = (cross and di_up_prev) or (hl and not hl_prev)
        exit_short = (cross and di_dn_prev) or (hl and not hl_prev)

        prev = cond
        if   prev != 1    and entry_long_str:  cond = 1
        elif prev != -1   and entry_short_st:  cond = -1
        elif prev != 0.5  and entry_long:      cond = 0.5
        elif prev != -0.5 and entry_short:     cond = -0.5
        elif prev != 0    and exit_long:       cond = 0
        elif prev != 0    and exit_short:      cond = 0
        short_e[i] = (prev != -0.5 and cond == -0.5)
    return pd.Series(short_e, index=df.index)


def macd_cross_up(df: pd.DataFrame) -> pd.Series:
    c = df["c"]
    macd = c.ewm(span=MACD_FAST, adjust=False).mean() \
         - c.ewm(span=MACD_SLOW, adjust=False).mean()
    sgn = macd.ewm(span=MACD_SIG, adjust=False).mean()
    return (macd.shift(1) <= sgn.shift(1)) & (macd > sgn)


# ============ SIGNAL DETECTION ============
def detect_signals(df: pd.DataFrame):
    """
    Return (signals, dropped_no_future).

    A signal at candle i requires:
      - MACD cross-up on candle i
      - ADX red triangle in [i-TRI_BEFORE, i-1]
        (1..TRI_BEFORE candles BEFORE cross, NOT the cross itself)
      - At least LOOKAHEAD candles exist AFTER candle i
        (otherwise dropped and counted in dropped_no_future)
    """
    cross = macd_cross_up(df).to_numpy()
    tri   = adx_short_triangle(df).to_numpy()
    n     = len(df)
    signals = []
    dropped = 0
    for i in range(WARMUP, n):
        if not cross[i]:
            continue
        start = max(0, i - TRI_BEFORE)
        if not tri[start:i].any():
            continue
        if i + LOOKAHEAD >= n:
            dropped += 1
            continue
        signals.append(i)
    return signals, dropped


# ============ BACKTEST ============
def backtest_coin(df: pd.DataFrame, signals):
    """Signals already guaranteed to have LOOKAHEAD candles after them."""
    outcomes = {(sl, tp): [] for sl in SL_LEVELS for tp in TP_LEVELS}

    for i in signals:
        entry = float(df["c"].iloc[i])
        hi = df["h"].iloc[i + 1:i + 1 + LOOKAHEAD].to_numpy(dtype=float)
        lo = df["l"].iloc[i + 1:i + 1 + LOOKAHEAD].to_numpy(dtype=float)
        run_hi = np.maximum.accumulate(hi)
        run_lo = np.minimum.accumulate(lo)

        sl_first = {}
        for sl in SL_LEVELS:
            price = entry * (1 - sl / 100)
            mask = run_lo <= price
            sl_first[sl] = int(np.argmax(mask)) if mask.any() else None

        for tp in TP_LEVELS:
            price = entry * (1 + tp / 100)
            mask = run_hi >= price
            tp_first = int(np.argmax(mask)) if mask.any() else None
            for sl in SL_LEVELS:
                sl_idx = sl_first[sl]
                # win only if TP hit strictly BEFORE SL (same-candle -> loss)
                win = (tp_first is not None and
                       (sl_idx is None or tp_first < sl_idx))
                outcomes[(sl, tp)].append(win)

    return outcomes, len(signals)


# ============ FORMAT ============
def md_table(all_outcomes, total, total_dropped):
    lines = []
    header = "| SL\\TP |" + "|".join(f" {tp}% |" for tp in TP_LEVELS)
    sep    = "|-------|" + "|".join("-------|" for _ in TP_LEVELS)
    lines.append(header)
    lines.append(sep)
    for sl in SL_LEVELS:
        row = f"| **{sl}%** |"
        for tp in TP_LEVELS:
            arr = all_outcomes[(sl, tp)]
            if arr:
                rate = sum(arr) / len(arr) * 100
                row += f" {rate:5.1f} |"
            else:
                row += "   -   |"
        lines.append(row)
    lines.append("")
    lines.append(f"**Total signals analyzed:** {total}")
    lines.append(f"**Dropped (no {LOOKAHEAD} candles of future data):** {total_dropped}")
    lines.append(f"**Triangle window:** `[i-{TRI_BEFORE}, i-1]` "
                 f"(1..{TRI_BEFORE} candles BEFORE cross)")
    lines.append(f"**Forward window:** {LOOKAHEAD} candles")
    lines.append(f"**Tie rule:** same-candle TP&SL -> loss (conservative)")
    return "\n".join(lines)


def mono_table(all_outcomes, total, total_dropped):
    col_w = 6
    head = "SL\\TP".ljust(6) + "".join(f"{tp}%".rjust(col_w) for tp in TP_LEVELS)
    lines = [head]
    for sl in SL_LEVELS:
        row = f"{sl}%".ljust(6)
        for tp in TP_LEVELS:
            arr = all_outcomes[(sl, tp)]
            val = f"{sum(arr)/len(arr)*100:.1f}" if arr else "-"
            row += val.rjust(col_w)
        lines.append(row)
    lines.append(f"\nSignals: {total} | Dropped (no future): {total_dropped}")
    lines.append(f"Triangle: 1..{TRI_BEFORE} bars before cross | Forward: {LOOKAHEAD} bars")
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
          f"TRI_BEFORE={TRI_BEFORE} coins={len(TOP_COINS)}", flush=True)
    print(f"rule: MACD cross-up on i AND red triangle in [i-{TRI_BEFORE}, i-1] "
          f"AND >= {LOOKAHEAD} candles after i", flush=True)

    all_outcomes   = {(sl, tp): [] for sl in SL_LEVELS for tp in TP_LEVELS}
    total_signals  = 0
    total_dropped  = 0
    per_coin       = {}

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

        per_coin[coin] = {
            "signals": n_valid,
            "dropped": dropped,
            "candles": len(df),
            "skipped": False,
        }
        total_signals += n_valid
        total_dropped += dropped

        for k, v in outcomes.items():
            all_outcomes[k].extend(v)

        print(f"[{coin}] candles={len(df)} signals={n_valid} "
              f"dropped_no_future={dropped}", flush=True)
        time.sleep(0.5)

    # ---- print markdown ----
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    lines = [
        "# Backtest — MACD cross-up + ADX red triangle (4h)",
        f"**Rule:** triangle in `[i-{TRI_BEFORE}, i-1]` "
        f"(1..{TRI_BEFORE} candles BEFORE cross, not cross, not after)",
        f"**Forward window:** {LOOKAHEAD} candles required after signal",
        f"**Tie rule:** same-candle TP&SL → loss (conservative)",
        f"Generated: {ts}",
        f"Coins: {len(TOP_COINS)} | Signals analyzed: {total_signals} "
        f"| Dropped (no future data): {total_dropped}",
        "",
        md_table(all_outcomes, total_signals, total_dropped),
        "",
        "## Per-coin signals",
    ]
    for coin, info in per_coin.items():
        if info["skipped"]:
            lines.append(f"- **{coin}**: _skipped (candles={info['candles']})_")
        else:
            lines.append(
                f"- **{coin}**: signals={info['signals']} "
                f"· dropped={info['dropped']} · candles={info['candles']}"
            )
    md = "\n".join(lines)
    print("\n" + md, flush=True)

    # ---- save artifact ----
    os.makedirs(OUT_DIR, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    with open(f"{OUT_DIR}/backtest_{stamp}.md", "w", encoding="utf-8") as f:
        f.write(md)
    with open(f"{OUT_DIR}/latest.md", "w", encoding="utf-8") as f:
        f.write(md)

    # ---- telegram ----
    tg_msg = (
        f"📊 <b>Backtest</b> — MACD↑ + 🔺ADX (4h)\n"
        f"Triangle: 1..{TRI_BEFORE} bars BEFORE cross\n"
        f"Forward: {LOOKAHEAD} bars required\n"
        f"Coins: {len(TOP_COINS)} | Signals: {total_signals} "
        f"| Dropped: {total_dropped}\n\n"
        f"<pre>{mono_table(all_outcomes, total_signals, total_dropped)}</pre>"
    )
    telegram(tg_msg)


if __name__ == "__main__":
    main()
