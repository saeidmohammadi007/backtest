import requests
import pandas as pd
import numpy as np
import time
from datetime import datetime

# ================= 配置 =================
BASE_URL = "https://api.lbkex.com"
TIMEFRAME = "day1"          # 可选: day1, hour4, hour1
KLINE_SIZE = 1000           # 每次请求的K线数量 (最大2000)
TOP_N = 100                 # 取成交量前100的USDT交易对
ADX_LEN = 14
DI_LEN = 14
HL_RANGE = 20
HL_TREND = 35
FORWARD_BARS = 100          # 信号后观察多少根K线
DRAWDOWN_LEVELS = [0.02, 0.03, 0.04, 0.05]   # 回撤阈值
TARGET_LEVELS = [0.10, 0.20, 0.40, 0.80, 1.00] # 目标涨幅

# ================= 工具函数 =================
def rma(series, length):
    """Wilder's Moving Average (RMA)"""
    alpha = 1.0 / length
    return series.ewm(alpha=alpha, adjust=False, min_periods=length).mean()

def fetch_all_usdt_pairs():
    """获取LBank所有USDT交易对"""
    url = f"{BASE_URL}/v1/currencyPairs.do"
    resp = requests.get(url, timeout=15)
    data = resp.json()
    if data.get("result") != "true":
        raise Exception(f"获取交易对失败: {data}")
    pairs = [p for p in data["data"] if p.endswith("_usdt")]
    print(f"共获取到 {len(pairs)} 个 USDT 交易对")
    return pairs

def fetch_tickers(pairs):
    """批量获取24h行情，用于排序成交量"""
    url = f"{BASE_URL}/v2/ticker/24hr.do"
    params = {"symbol": "all"}
    resp = requests.get(url, params=params, timeout=15)
    data = resp.json()
    if data.get("result") != "true":
        raise Exception(f"获取ticker失败: {data}")
    
    ticker_map = {}
    for item in data["data"]:
        sym = item.get("symbol", "").lower()
        if sym.endswith("_usdt"):
            ticker_map[sym] = {
                "turnover": float(item.get("turnover", 0)),
                "vol": float(item.get("vol", 0)),
            }
    ranked = sorted(
        [(sym, info) for sym, info in ticker_map.items() if sym in pairs],
        key=lambda x: x[1]["turnover"],
        reverse=True
    )
    return [r[0] for r in ranked[:TOP_N]]

def fetch_klines(symbol, timeframe="day1", size=KLINE_SIZE):
    """获取K线数据"""
    url = f"{BASE_URL}/v2/kline.do"
    params = {
        "symbol": symbol,
        "size": size,
        "type": timeframe,
        "time": int(time.time())
    }
    resp = requests.get(url, params=params, timeout=20)
    data = resp.json()
    if data.get("result") != "true":
        print(f"  [警告] {symbol} K线获取失败: {data.get('msg', data)}")
        return None
    
    rows = data["data"]
    if not rows:
        return None
    
    df = pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna().reset_index(drop=True)
    return df

# ================= ADX 逻辑 (移植 Pine) =================
def compute_adx_gu5(df, di_len=14, sig_len=14):
    """完整移植 ADX Di+ Di- [Gu5] 的核心逻辑"""
    high = df["high"]
    low = df["low"]
    close = df["close"]
    
    # --- f_dirMov ---
    up = high.diff()
    down = -low.diff()
    
    plus_dm = np.where((up > down) & (up > 0), up, 0.0)
    minus_dm = np.where((down > up) & (down > 0), down, 0.0)
    
    plus_dm = pd.Series(plus_dm, index=df.index)
    minus_dm = pd.Series(minus_dm, index=df.index)
    plus_dm.iloc[0] = np.nan
    minus_dm.iloc[0] = np.nan
    
    tr = pd.concat([high - low, (high - close.shift()).abs(), (low - close.shift()).abs()], axis=1).max(axis=1)
    true_range = rma(tr, di_len)
    
    plus = 100 * rma(plus_dm, di_len) / true_range
    minus = 100 * rma(minus_dm, di_len) / true_range
    
    plus = plus.ffill()
    minus = minus.ffill()
    
    # --- f_sig ---
    sum_dm = plus + minus
    sum_dm = sum_dm.replace(0, 1)
    sig = 100 * rma((plus - minus).abs() / sum_dm, sig_len)
    sig = sig.ffill()
    
    df["sig"] = sig
    df["diPlus"] = plus
    df["diMinus"] = minus
    
    # --- 信号逻辑 ---
    hlRange = sig <= HL_RANGE
    diUp = plus >= minus
    diUpUp = plus >= HL_TREND
    diDn = minus > plus
    diDnDn = minus > HL_TREND
    crossDi = ((plus > minus) & (plus.shift(1) <= minus.shift(1))) | ((plus < minus) & (plus.shift(1) >= minus.shift(1)))
    sigUp = sig > sig.shift(1)
    
    # condition 状态机
    condition = np.zeros(len(df))
    condition[0] = 0.0
    
    for i in range(1, len(df)):
        prev = condition[i-1]
        
        if not hlRange.iloc[i] and diUp.iloc[i] and sigUp.iloc[i] and diUpUp.iloc[i]:
            condition[i] = 1
        elif not hlRange.iloc[i] and diDn.iloc[i] and sigUp.iloc[i] and diDnDn.iloc[i]:
            condition[i] = -1
        elif not hlRange.iloc[i] and diUp.iloc[i] and sigUp.iloc[i] and not diUp.iloc[i-1]:
            condition[i] = 0.5
        elif not hlRange.iloc[i] and diDn.iloc[i] and sigUp.iloc[i] and not diDn.iloc[i-1]:
            condition[i] = -0.5
        elif (crossDi.iloc[i] and diUp.iloc[i-1]) or (hlRange.iloc[i] and not hlRange.iloc[i-1]):
            condition[i] = 0
        elif (crossDi.iloc[i] and diDn.iloc[i-1]) or (hlRange.iloc[i] and not hlRange.iloc[i-1]):
            condition[i] = 0
        else:
            condition[i] = prev
    
    df["condition"] = condition
    df["hlRange"] = hlRange.values
    
    df["longX"] = (df["condition"].shift(1).isin([0.5, 1.0])) & (df["condition"] == 0)
    df["shortX"] = (df["condition"].shift(1).isin([-0.5, -1.0])) & (df["condition"] == 0)
    df["longEStr"] = (df["condition"].shift(1) != 1) & (df["condition"] == 1)
    df["shortEStr"] = (df["condition"].shift(1) != -1) & (df["condition"] == -1)
    
    return df

# ================= 模式检测与统计 =================
def detect_pattern(df):
    """
    检测：
    - 大绿三角 (longEStr) 之前 10~60 根内出现 longX 或 shortX
    - 两者之间没有其他信号 (longEStr/shortEStr/longX/shortX)
    - 两者之间至少有一根黄色K线 (hlRange == True，即 ADX <= 20)
    返回匹配的信号索引列表（以 longEStr 的位置为准）
    """
    signal_cols = ["longX", "shortX", "longEStr", "shortEStr"]
    yellow = df["hlRange"]

    matches = []
    for idx in df.index[df["longEStr"]]:
        found = False
        for back in range(10, 61):
            prev_idx = idx - back
            if prev_idx < 0:
                break
            if df.loc[prev_idx, "longX"] or df.loc[prev_idx, "shortX"]:
                # 检查中间有无其他信号
                between_signals = df.loc[prev_idx+1:idx-1, signal_cols].any(axis=1)
                if between_signals.any():
                    break
                # 检查中间是否至少有一根黄色K线
                between_yellow = yellow.loc[prev_idx+1:idx-1]
                if not between_yellow.any():
                    break
                found = True
                break
        if found:
            matches.append(idx)

    return matches

def evaluate_forward(df, entry_idx, drawdown_levels, target_levels, forward_bars=100):
    """评估信号后的表现：先回撤到阈值，然后反弹到目标涨幅"""
    entry_price = df.loc[entry_idx, "close"]
    end_idx = min(entry_idx + forward_bars, len(df) - 1)
    
    future_high = df.loc[entry_idx+1:end_idx, "high"].values
    future_low = df.loc[entry_idx+1:end_idx, "low"].values
    
    if len(future_high) == 0:
        return None
    
    results = {}
    for dd in drawdown_levels:
        dd_price = entry_price * (1 - dd)
        touch_indices = np.where(future_low <= dd_price)[0]
        if len(touch_indices) == 0:
            results[dd] = {t: False for t in target_levels}
            results[dd]["_touched"] = False
            continue
        
        touch_idx = touch_indices[0]
        results[dd] = {"_touched": True}
        
        remaining_high = future_high[touch_idx:]
        for t in target_levels:
            target_price = entry_price * (1 + t)
            results[dd][t] = bool(np.any(remaining_high >= target_price))
    
    return results

# ================= 主流程 =================
def main():
    print("=" * 60)
    print("LBank ADX 形态统计脚本")
    print(f"时间框架: {TIMEFRAME}, 前 {TOP_N} 个 USDT 交易对")
    print("条件: × → 黄色区间 → 大绿三角")
    print("=" * 60)
    
    all_pairs = fetch_all_usdt_pairs()
    time.sleep(0.5)
    
    top_pairs = fetch_tickers(all_pairs)
    print(f"成交量前 {len(top_pairs)} 的交易对: {top_pairs[:5]} ...")
    time.sleep(0.5)
    
    all_matches = []
    
    for i, symbol in enumerate(top_pairs):
        print(f"[{i+1}/{len(top_pairs)}] 处理 {symbol} ...", end=" ")
        
        df = fetch_klines(symbol, TIMEFRAME, KLINE_SIZE)
        if df is None or len(df) < 200:
            print("数据不足，跳过")
            time.sleep(0.2)
            continue
        
        try:
            df = compute_adx_gu5(df, DI_LEN, ADX_LEN)
        except Exception as e:
            print(f"ADX计算失败: {e}")
            time.sleep(0.2)
            continue
        
        matches = detect_pattern(df)
        
        if not matches:
            print(f"无匹配形态 (共{len(df)}根K线)")
            time.sleep(0.2)
            continue
        
        print(f"找到 {len(matches)} 个匹配形态")
        
        for idx in matches:
            res = evaluate_forward(df, idx, DRAWDOWN_LEVELS, TARGET_LEVELS, FORWARD_BARS)
            if res:
                all_matches.append({
                    "symbol": symbol,
                    "timestamp": df.loc[idx, "timestamp"],
                    "entry_price": df.loc[idx, "close"],
                    "results": res,
                })
        
        time.sleep(0.3)
    
    # 汇总统计
    print("\n" + "=" * 60)
    print("统计结果")
    print("=" * 60)
    
    if not all_matches:
        print("没有找到任何匹配的形态。")
        return
    
    print(f"总匹配形态数: {len(all_matches)}")
    
    header = f"{'回撤':<8}" + "".join([f"{'→'+str(int(t*100))+'%':<14}" for t in TARGET_LEVELS])
    print(header)
    print("-" * 80)
    
    for dd in DRAWDOWN_LEVELS:
        row = f"{int(dd*100)}%{'':<5}"
        touched_count = sum(1 for m in all_matches if m["results"].get(dd, {}).get("_touched"))
        for t in TARGET_LEVELS:
            hit_count = 0
            for m in all_matches:
                r = m["results"].get(dd)
                if r and r.get("_touched") and r.get(t):
                    hit_count += 1
            pct = (hit_count / touched_count * 100) if touched_count > 0 else 0
            row += f"{pct:.1f}%({hit_count}/{touched_count})".ljust(16)
        print(row)
    
    print("\n匹配样本明细:")
    for m in all_matches[:20]:
        print(f"  {m['symbol']} @ {m['timestamp']}  entry={m['entry_price']}")
        for dd in DRAWDOWN_LEVELS:
            r = m["results"].get(dd, {})
            if r.get("_touched"):
                hits = [str(int(t*100))+"%" for t in TARGET_LEVELS if r.get(t)]
                print(f"    回撤{int(dd*100)}% → 达成: {hits if hits else '无'}")
            else:
                print(f"    回撤{int(dd*100)}% → 未触及")

if __name__ == "__main__":
    main()
