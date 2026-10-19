# Backtest — MACD cross-up + ADX End-Trend AFTER cross (4h)
**Rule:** End-Trend in `[i+1, i+3]` (1..3 candles AFTER cross, not cross, not before)
**End-Trend:** exit from any ADX trend (weak/strong, long/short) → 0 (orange ✕ in 'ADX Di+ Di- [Gu5]')
**Forward window:** 100 candles required after signal
**Tie rule:** same-candle TP&SL → loss (conservative)
**⚠️ Warning:** look-ahead bias — entry at `i`, End-Trend at `i+1..i+3`
Generated: 2026-10-19 08:27 UTC
Coins: 20 | Signals analyzed: 208 | Dropped (no future data): 4

| SL\TP | 10% || 20% || 30% || 40% || 50% || 60% || 70% || 80% || 90% || 100% |
|-------|-------||-------||-------||-------||-------||-------||-------||-------||-------||-------|
| **2%** |  28.4 |  11.1 |   8.7 |   2.4 |   1.0 |   0.0 |   0.0 |   0.0 |   0.0 |   0.0 |
| **3%** |  35.1 |  13.5 |   9.1 |   2.4 |   1.0 |   0.0 |   0.0 |   0.0 |   0.0 |   0.0 |
| **4%** |  40.9 |  16.3 |  10.6 |   2.9 |   1.4 |   0.0 |   0.0 |   0.0 |   0.0 |   0.0 |
| **5%** |  41.3 |  16.8 |  11.1 |   3.4 |   1.9 |   0.5 |   0.5 |   0.5 |   0.5 |   0.5 |

**Total signals analyzed:** 208
**Dropped (no 100 candles of future data):** 4
**End-Trend window:** `[i+1, i+3]` (1..3 candles AFTER cross, not cross, not before)
**End-Trend meaning:** exit from any ADX trend (weak/strong, long/short) -> 0 (orange ✕)
**Forward window:** 100 candles
**Tie rule:** same-candle TP&SL -> loss (conservative)
**⚠️ Warning:** look-ahead bias (entry at i, End-Trend in i+1..i+3)

## Per-coin signals
- **BTC**: signals=16 · dropped=1 · candles=4184
- **ETH**: signals=9 · dropped=0 · candles=4181
- **BNB**: signals=13 · dropped=0 · candles=4184
- **SOL**: signals=11 · dropped=0 · candles=4183
- **XRP**: signals=14 · dropped=0 · candles=4184
- **DOGE**: signals=12 · dropped=0 · candles=4182
- **ADA**: signals=10 · dropped=0 · candles=4182
- **AVAX**: signals=12 · dropped=1 · candles=4182
- **TRX**: signals=17 · dropped=2 · candles=4184
- **LINK**: signals=9 · dropped=0 · candles=4182
- **DOT**: signals=10 · dropped=0 · candles=4183
- **MATIC**: _skipped (candles=0)_
- **TON**: signals=5 · dropped=0 · candles=4144
- **SHIB**: signals=13 · dropped=0 · candles=4184
- **LTC**: signals=18 · dropped=0 · candles=4183
- **BCH**: signals=7 · dropped=0 · candles=4184
- **UNI**: _skipped (candles=0)_
- **XLM**: signals=17 · dropped=0 · candles=4182
- **NEAR**: signals=15 · dropped=0 · candles=4183
- **APT**: _skipped (candles=0)_