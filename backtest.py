"""
Backtest: MACD cross-up + ADX End-Trend in [i+1, i+3]
on top-100 coins, 4h timeframe.

Signal rule (UNCHANGED — same as original):
  1. MACD cross-up on candle i
  2. ADX End-Trend (longX or shortX — orange ✕ in "ADX Di+ Di- [Gu5]")
     in candles [i+1, i+3]
     -> 1 to 3 candles AFTER the cross
     -> NOT including the cross candle itself
     -> NOT before the cross
  3. At least LOOKAHEAD=100 candles exist AFTER candle i
     (signals too close to end of data are dropped)

For each signal, tests next 100 candles for each (SL, TP) pair.
Win = TP hit before SL (same-candle tie -> loss, conservative).

⚠️ WARNING: End-Trend is AFTER the cross, so entry at candle i close
   uses future information (look-ahead bias). Results are optimistic.
"""

import os
import time
import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import requests


# ============ CONFIG ============
FALLBACK_TOP_COINS = [
    "BTC", "ETH", "BNB", "SOL", "XRP", "DOGE", "ADA", "TRX", "AVAX", "LINK",
    "TON", "SHIB", "DOT", "MATIC", "LTC", "BCH", "UNI", "NEAR", "APT", "ICP",
    "XLM", "ETC", "FIL", "ATOM", "ARB", "OP", "IMX", "INJ", "HBAR", "SUI",
    "STX", "SEI", "TIA", "GRT", "RUNE", "AAVE", "MKR", "LDO", "ALGO", "QNT",
    "FTM", "EGLD", "FLOW", "SAND", "MANA", "AXS", "GALA", "THETA", "KAVA", "ROSE",
    "CHZ", "CRV", "SNX", "COMP", "ZEC", "DASH", "XMR", "WAVES", "IOTA", "NEO",
    "EOS", "XTZ", "ONE", "ENJ", "BAT", "ZIL", "KSM", "WOO", "CAKE", "DYDX",
    "RPL", "BLUR", "APE", "GMX", "OSMO", "JUP", "PYTH", "WLD", "FET", "RNDR",
    "AGIX", "OCEAN", "AKT", "AR", "HNT", "KAS", "MINA", "CFX", "MASK", "ENS",
    "LPT", "SSV", "PENDLE", "MAGIC", "HOOK", "ID", "ARKM", "ACE", "NFP", "AI",
]

TOP_N_COINS = int(os.getenv("TOP_N_COINS", "100"))
SL_LEVELS = [2, 3, 4, 5]
TP_LEVELS = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
LOOKAHEAD = 100
TRI_AFTER = 3
RANGE_DAYS = int(os.getenv("RANGE_DAYS", "365"))
WARMUP = 100
MIN_CANDLES = 200

# --- Tie rule (default 'loss' matches original exactly) ---
TIE_RULE = os.getenv("TIE_RULE", "loss").lower()   # loss / win / half

# --- Trading costs (default 0 = matches original exactly) ---
FEE_PCT = float(os.getenv("FEE_PCT", "0"))
SLIPPAGE_PCT = float(os.getenv("SLIPPAGE_PCT", "0"))

MIN_SIG_FOR_RANK = int(os.getenv("MIN_SIG_FOR_RANK", "10"))
REF_SL = int(os.getenv("REF_SL", "3"))
REF_TP = int(os.getenv("REF_TP", "10"))

MACD_FAST, MACD_SLOW, MACD_SIG = 12, 26, 9
ADX_SIG_LEN, DI_LEN, LVL_RANGE, LVL_TREND = 14, 14, 20, 35
TF_SECONDS = 4 * 3600

TG_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TG_CHAT = os.getenv("TELEGRAM_CHAT_ID", "")

YAHOO = "https://query1.finance.yahoo.com/v8/finance/chart"
S = requests.Session()
S.headers["User-Agent"] = "Mozilla/5.0"

OUT_DIR = "results"
CACHE_DIR = "cache"
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)
CACHE_TTL = int(os.getenv("CACHE_TTL", "3600"))


# ============ REGIME LABELS ============
REGIMES = [
    ("2021-01-01", "2021-11-10", "bull_2021"),
    ("2021-11-11", "2022-06-18", "bear_2021H2"),
    ("2022-06-19", "2022-11-21", "bear_2022"),
    ("2022-11-22", "2023-10-15", "recovery_2023"),
    ("2023-10-16", "2024-03-14", "bull_2023H2"),
    ("2024-03-15", "2024-09-30", "range_2024"),
    ("2024-10-01", "2025-01-20", "bull_2024H2"),
    ("2025-01-21", "2025-04-30", "correction_2025"),
    ("2025-05-01", "2025-10-31", "mixed_2025"),
]


def label_date(ts_sec: float) -> str:
    dt = datetime.fromtimestamp(ts_sec, tz=timezone.utc).strftime("%Y-%m-%d")
    for start, end, lbl in REGIMES:
        if start <= dt <= end:
            return lbl
    return "other"


# ============ FETCH ============
def get_json(url, params=None, tries=5):
    for i in range(tries):
        try:
            r = S.get(url, params=params, timeout=30)
            if r.status_code == 200:
                return r.json()
            if r.status_code == 404:
                return None
            if r.status_code == 429:
                time.sleep(5 * (i + 1))
                continue
        except Exception:
            pass
        time.sleep(2 + i * 2)
    return None


def get_top_coins(n=100):
    try:
        r = S.get(
            "https://api.coingecko.com/api/v3/coins/markets",
            params={"vs_currency": "usd", "order": "market_cap_desc",
                    "per_page": n, "page": 1, "sparkline": "false"},
            timeout=30,
        )
        if r.status_code != 200:
            print(f"[coingecko] status={r.status_code}, using fallback", flush=True)
            return None
        data = r.json()
        symbols = [c["symbol"].upper() for c in data]
        seen, out = set(), []
        for s in symbols:
            if s not in seen:
                seen.add(s)
                out.append(s)
        return out
    except Exception as e:
        print(f"[coingecko] error: {e}", flush=True)
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
    g = g[g["n"] == 4].dropna()   # ← same as original
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


def fetch_klines_cached(ticker):
    safe = ticker.replace("/", "_")
    base = os.path.join(CACHE_DIR, f"{safe}_{RANGE_DAYS}")
    for ext in (".parquet", ".pkl"):
        path = base + ext
        if os.path.exists(path):
            age = time.time() - os.path.getmtime(path)
            if age < CACHE_TTL:
                try:
                    return pd.read_parquet(path) if ext == ".parquet" else pd.read_pickle(path)
                except Exception:
                    pass
    df = fetch_klines(ticker)
    if df is not None and len(df) > 0:
        try:
            df.to_parquet(base + ".parquet", index=False)
        except Exception:
            try:
                df.to_pickle(base + ".pkl")
            except Exception:
                pass
    return df


# ============ INDICATORS (UNCHANGED) ============
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
    """Same as original — equivalent of longX or shortX in Pine 'ADX Di+ Di- [Gu5]'."""
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


# ============ SIGNAL DETECTION (UNCHANGED — exactly the same loop) ============
def detect_signals(df: pd.DataFrame):
    """
    A signal at candle i requires:
      - MACD cross-up on candle i
      - ADX End-Trend in [i+1, i+TRI_AFTER]
        (1..TRI_AFTER candles AFTER the cross, NOT the cross, NOT before)
      - At least LOOKAHEAD candles exist AFTER candle i
    """
    cross = macd_cross_up(df).to_numpy()
    end_tr = adx_end_trend(df).to_numpy()
    n = len(df)
    signals = []
    dropped = 0

    for i in range(WARMUP, n):
        if not cross[i]:
            continue
        start = i + 1
        end = min(n, i + TRI_AFTER + 1)
        if start >= end:
            continue
        if not end_tr[start:end].any():
            continue
        if i + LOOKAHEAD >= n:
            dropped += 1
            continue
        signals.append(i)
    return signals, dropped


# ============ BACKTEST ============
def backtest_coin(df: pd.DataFrame, signals):
    """
    Same as original:
      entry = close of bar i
      hi/lo = forward window [i+1, i+LOOKAHEAD]
      win = TP hit strictly before SL (tie -> loss)
    Costs default to 0 (FEE_PCT=SLIPPAGE_PCT=0) so behavior matches original exactly.
    """
    outcomes = {(sl, tp): [] for sl in SL_LEVELS for tp in TP_LEVELS}
    regimes = {(sl, tp): [] for sl in SL_LEVELS for tp in TP_LEVELS}
    cost = (FEE_PCT + SLIPPAGE_PCT) / 100.0  # 0 by default

    for i in signals:
        entry = float(df["c"].iloc[i]) * (1 + cost)
        hi = df["h"].iloc[i + 1:i + 1 + LOOKAHEAD].to_numpy(dtype=float)
        lo = df["l"].iloc[i + 1:i + 1 + LOOKAHEAD].to_numpy(dtype=float)
        ts = float(df["t"].iloc[i])
        reg = label_date(ts)

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
                if tp_first is None:
                    win = 0.0
                elif sl_idx is None:
                    win = 1.0
                elif tp_first < sl_idx:
                    win = 1.0
                elif sl_idx < tp_first:
                    win = 0.0
                else:  # same-candle tie
                    win = {"win": 1.0, "half": 0.5}.get(TIE_RULE, 0.0)
                outcomes[(sl, tp)].append(win)
                regimes[(sl, tp)].append(reg)

    return outcomes, regimes, len(signals)


# ============ RANKING ============
def coin_ranking(per_coin, ref_sl, ref_tp, min_sig=MIN_SIG_FOR_RANK):
    rows = []
    for coin, info in per_coin.items():
        if info.get("skipped"):
            continue
        n_sig = info["signals"]
        if n_sig < min_sig:
            continue
        outs = info.get("outcomes", {})
        ref_arr = outs.get((ref_sl, ref_tp), [])
        ref_wins = float(sum(ref_arr))
        ref_losses = len(ref_arr) - ref_wins
        ref_rate = (ref_wins / len(ref_arr) * 100) if ref_arr else None

        best = None
        for sl in SL_LEVELS:
            for tp in TP_LEVELS:
                arr = outs.get((sl, tp), [])
                if len(arr) >= min_sig and sum(arr) >= 2:
                    rate = sum(arr) / len(arr) * 100
                    if best is None or rate > best[2]:
                        best = (sl, tp, rate, len(arr), sum(arr))

        all_rates = []
        for sl in SL_LEVELS:
            for tp in TP_LEVELS:
                arr = outs.get((sl, tp), [])
                if arr:
                    all_rates.append(sum(arr) / len(arr) * 100)
        avg_rate = sum(all_rates) / len(all_rates) if all_rates else None

        rows.append({
            "coin": coin, "signals": n_sig,
            "ref_wins": ref_wins, "ref_losses": ref_losses,
            "ref_rate": ref_rate,
            "best_sl": best[0] if best else None,
            "best_tp": best[1] if best else None,
            "best_rate": best[2] if best else None,
            "avg_rate": avg_rate,
        })
    rows.sort(key=lambda r: r["ref_rate"] if r["ref_rate"] is not None else -1,
              reverse=True)
    return rows


# ============ REGIME STATS ============
def regime_stats(all_outcomes, all_regimes, sl, tp):
    arr = all_outcomes[(sl, tp)]
    reg_arr = all_regimes[(sl, tp)]
    by = {}
    for w, r in zip(arr, reg_arr):
        by.setdefault(r, []).append(w)
    lines = [f"### Regime breakdown — SL={sl}% / TP={tp}%",
             "| Regime | Signals | Win rate |",
             "|--------|---------|----------|"]
    for r in sorted(by.keys()):
        ws = by[r]
        wr = sum(ws) / len(ws) * 100
        lines.append(f"| {r} | {len(ws)} | {wr:5.1f}% |")
    return "\n".join(lines)


# ============ FORMAT ============
def md_table(all_outcomes, total, total_dropped):
    lines = []
    header = "| SL\\TP |" + "|".join(f" {tp}% |" for tp in TP_LEVELS)
    sep = "|-------|" + "|".join("-------|" for _ in TP_LEVELS)
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
    lines.append(f"**End-Trend window:** `[i+1, i+{TRI_AFTER}]` "
                 f"(1..{TRI_AFTER} candles AFTER cross, not cross, not before)")
    lines.append("**End-Trend meaning:** exit from any ADX trend "
                 "(weak/strong, long/short) → 0 (orange ✕)")
    lines.append(f"**Forward window:** {LOOKAHEAD} candles")
    lines.append(f"**Tie rule:** {TIE_RULE}")
    lines.append(f"**Costs:** fee={FEE_PCT}%/side, slippage={SLIPPAGE_PCT}%/side")
    lines.append(f"**⚠️ Warning:** look-ahead bias (entry at i, "
                 f"End-Trend in i+1..i+{TRI_AFTER})")
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
    lines.append(f"\nSignals: {total} | Dropped: {total_dropped}")
    lines.append(f"End-Trend: 1..{TRI_AFTER} bars AFTER cross | Fwd: {LOOKAHEAD} bars")
    lines.append(f"Tie: {TIE_RULE} | Fee/slip: {FEE_PCT}%/{SLIPPAGE_PCT}% per side")
    lines.append(f"⚠️ look-ahead bias: entry at i, End-Trend i+1..i+{TRI_AFTER}")
    return "\n".join(lines)


def telegram(text):
    if not TG_TOKEN or not TG_CHAT:
        print("[telegram] not configured — skipping", flush=True)
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
            print(f"[telegram] error: {e}", flush=True)


# ============ MAIN ============
def main():
    print(f"[setup] TIE_RULE={TIE_RULE} FEE={FEE_PCT}% SLIP={SLIPPAGE_PCT}%",
          flush=True)
    print(f"[setup] loading top {TOP_N_COINS} coins...", flush=True)
    auto = get_top_coins(TOP_N_COINS)
    if auto:
        TOP_COINS = auto
        print(f"[setup] loaded {len(TOP_COINS)} coins from CoinGecko", flush=True)
    else:
        TOP_COINS = FALLBACK_TOP_COINS[:TOP_N_COINS]
        print(f"[setup] using fallback list ({len(TOP_COINS)} coins)", flush=True)

    print(f"config: RANGE_DAYS={RANGE_DAYS} LOOKAHEAD={LOOKAHEAD} "
          f"TRI_AFTER={TRI_AFTER} coins={len(TOP_COINS)}", flush=True)
    print(f"rule: MACD cross-up on i AND End-Trend in [i+1, i+{TRI_AFTER}] "
          f"AND >= {LOOKAHEAD} candles after i", flush=True)
    print(f"⚠️ WARNING: look-ahead bias — entry at i, "
          f"End-Trend in i+1..i+{TRI_AFTER}", flush=True)

    all_outcomes = {(sl, tp): [] for sl in SL_LEVELS for tp in TP_LEVELS}
    all_regimes = {(sl, tp): [] for sl in SL_LEVELS for tp in TP_LEVELS}
    total_signals = 0
    total_dropped = 0
    per_coin = {}

    for idx, coin in enumerate(TOP_COINS):
        label = f"[{idx+1}/{len(TOP_COINS)}] {coin}"
        print(f"{label} fetching...", flush=True)
        df = fetch_klines_cached(f"{coin}-USD")
        if df is None or len(df) < MIN_CANDLES:
            n = 0 if df is None else len(df)
            print(f"{label} skipped (candles={n})", flush=True)
            per_coin[coin] = {"signals": 0, "dropped": 0,
                              "candles": n, "skipped": True, "outcomes": {}}
            time.sleep(1.2)
            continue

        try:
            signals, dropped = detect_signals(df)
            outcomes, regimes, n_valid = backtest_coin(df, signals)
        except Exception as e:
            print(f"{label} error: {e}", flush=True)
            per_coin[coin] = {"signals": 0, "dropped": 0,
                              "candles": len(df), "skipped": True, "outcomes": {}}
            time.sleep(1.2)
            continue

        per_coin[coin] = {
            "signals": n_valid, "dropped": dropped,
            "candles": len(df), "skipped": False,
            "outcomes": outcomes, "regimes": regimes,
        }
        total_signals += n_valid
        total_dropped += dropped
        for k, v in outcomes.items():
            all_outcomes[k].extend(v)
        for k, v in regimes.items():
            all_regimes[k].extend(v)

        print(f"{label} candles={len(df)} signals={n_valid} "
              f"dropped={dropped}", flush=True)

        if (idx + 1) % 10 == 0:
            telegram(f"⏳ progress: {idx+1}/{len(TOP_COINS)} "
                     f"· signals: {total_signals}")
        time.sleep(1.2)

    ranking_rows = coin_ranking(per_coin, REF_SL, REF_TP)

    # ── MD report ──
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    lines = [
        "# Backtest — MACD cross-up + ADX End-Trend AFTER cross (4h)",
        f"**Rule:** End-Trend in `[i+1, i+{TRI_AFTER}]` "
        f"(1..{TRI_AFTER} candles AFTER cross, not cross, not before)",
        "**End-Trend:** exit from any ADX trend (weak/strong, long/short) → 0 "
        "(orange ✕ in 'ADX Di+ Di- [Gu5]')",
        f"**Forward window:** {LOOKAHEAD} candles required after signal",
        f"**Tie rule:** {TIE_RULE}",
        f"**Costs:** fee={FEE_PCT}%/side, slippage={SLIPPAGE_PCT}%/side",
        f"**⚠️ Warning:** look-ahead bias — entry at `i`, "
        f"End-Trend at `i+1..i+{TRI_AFTER}`",
        f"**Generated:** {ts}",
        f"**Coins:** {len(TOP_COINS)} | Signals: {total_signals} | "
        f"Dropped: {total_dropped}",
        "",
        "## Overall win-rate matrix",
        "",
        md_table(all_outcomes, total_signals, total_dropped),
        "",
        f"## Regime breakdown at SL={REF_SL}% / TP={REF_TP}%",
        "",
        regime_stats(all_outcomes, all_regimes, REF_SL, REF_TP),
        "",
        f"## 🏆 Coin ranking (SL={REF_SL}% / TP={REF_TP}%)",
        "",
        f"_min signals for ranking: {MIN_SIG_FOR_RANK}_",
        "",
        "| # | Coin | Signals | Win@ref | Loss@ref | WR@ref | Best SL/TP | Best WR |",
        "|---|------|---------|---------|----------|--------|------------|---------|",
    ]
    for i, r in enumerate(ranking_rows[:30], 1):
        ref = f"{r['ref_rate']:.1f}%" if r["ref_rate"] is not None else "-"
        best = f"{r['best_sl']}%/{r['best_tp']}%" if r["best_sl"] else "-"
        bwr = f"{r['best_rate']:.1f}%" if r["best_rate"] is not None else "-"
        lines.append(
            f"| {i} | **{r['coin']}** | {r['signals']} | "
            f"{r['ref_wins']:.1f} | {r['ref_losses']:.1f} | {ref} | "
            f"{best} | {bwr} |"
        )

    md = "\n".join(lines)
    print("\n" + md, flush=True)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    with open(f"{OUT_DIR}/backtest_{stamp}.md", "w", encoding="utf-8") as f:
        f.write(md)
    with open(f"{OUT_DIR}/latest.md", "w", encoding="utf-8") as f:
        f.write(md)

    # JSON snapshot
    raw = {
        "generated": ts,
        "tie_rule": TIE_RULE, "fee_pct": FEE_PCT, "slippage_pct": SLIPPAGE_PCT,
        "ref_sl": REF_SL, "ref_tp": REF_TP,
        "total_signals": total_signals, "total_dropped": total_dropped,
        "ranking": ranking_rows,
        "matrix": {
            f"{sl}_{tp}": {
                "n": len(all_outcomes[(sl, tp)]),
                "wins": float(sum(all_outcomes[(sl, tp)])),
                "wr": (float(sum(all_outcomes[(sl, tp)])) /
                       len(all_outcomes[(sl, tp)]) * 100)
                      if all_outcomes[(sl, tp)] else None,
            }
            for sl in SL_LEVELS for tp in TP_LEVELS
        },
    }
    with open(f"{OUT_DIR}/latest.json", "w", encoding="utf-8") as f:
        json.dump(raw, f, indent=2, default=str)

    # Telegram
    tg_msg = (
        f"📊 <b>Backtest</b> — MACD↑ + End-Trend ✕ AFTER cross (4h)\n"
        f"End-Trend: 1..{TRI_AFTER} bars AFTER cross\n"
        f"⚠️ look-ahead bias: entry at i, End-Trend i+1..i+{TRI_AFTER}\n"
        f"Fee: {FEE_PCT}%/side · Slip: {SLIPPAGE_PCT}%/side · Tie: {TIE_RULE}\n"
        f"Coins: {len(TOP_COINS)} | Signals: {total_signals} | "
        f"Dropped: {total_dropped}\n\n"
        f"<pre>{mono_table(all_outcomes, total_signals, total_dropped)}</pre>"
    )
    telegram(tg_msg)


if __name__ == "__main__":
    main()
