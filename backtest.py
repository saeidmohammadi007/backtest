# ====== جایگزین کن در اسکریپت اصلی ======
# شرط سیگنال دقیقاً همان است: MACD cross-up در i و End-Trend در [i+1, i+3]
# فقط نقطه ورود عوض می‌شود: close اولین کندلی که End-Trend در آن دیده شد.

FEE_PCT = 0.1      # کارمزد+slippage رفت‌وبرگشت (درصد)
RANDOM_SEED = 42


def detect_signals(df: pd.DataFrame):
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
        j = i + 1 + int(hits[0])          # کندل تأیید = نقطه ورود
        if j + LOOKAHEAD >= n:
            dropped += 1
            continue
        signals.append(j)
    return signals, dropped


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
            # کارمزد از سود کم می‌شود: باید tp + FEE_PCT حرکت کند
            mask = run_hi >= entry * (1 + (tp + FEE_PCT) / 100)
            tp_first = int(np.argmax(mask)) if mask.any() else None
            for sl in SL_LEVELS:
                sl_idx = sl_first[sl]
                win = (tp_first is not None and
                       (sl_idx is None or tp_first < sl_idx))
                outcomes[(sl, tp)].append(win)
    return outcomes, len(signals)


def random_baseline(df: pd.DataFrame, n_signals: int, rng):
    """ورود تصادفی با همان تعداد سیگنال، همان SL/TP."""
    lo_i, hi_i = WARMUP, len(df) - LOOKAHEAD - 1
    if n_signals == 0 or hi_i <= lo_i:
        return {(sl, tp): [] for sl in SL_LEVELS for tp in TP_LEVELS}
    idx = rng.choice(np.arange(lo_i, hi_i), size=n_signals, replace=True)
    out, _ = backtest_coin(df, list(idx))
    return out


# ====== تغییر در main() ======
# قبل از حلقه:
#   rng = np.random.default_rng(RANDOM_SEED)
#   rand_outcomes = {(sl, tp): [] for sl in SL_LEVELS for tp in TP_LEVELS}
#
# داخل حلقه، بعد از backtest_coin:
#   rb = random_baseline(df, n_valid, rng)
#   for k, v in rb.items():
#       rand_outcomes[k].extend(v)
#
# بعد از حلقه، جدول اختلاف (سیگنال - تصادفی) را چاپ کن:
#   for sl in SL_LEVELS:
#       row = f"{sl}%".ljust(6)
#       for tp in TP_LEVELS:
#           a, b = all_outcomes[(sl, tp)], rand_outcomes[(sl, tp)]
#           d = (sum(a)/len(a) - sum(b)/len(b)) * 100 if a and b else float("nan")
#           row += f"{d:+.1f}".rjust(7)
#       print(row)
