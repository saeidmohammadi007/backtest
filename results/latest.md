# Backtest — MACD cross-up + ADX End-Trend (4h)
**Rule:** End-Trend in `[i-3, i+3]` (3 candles before cross + cross candle + 3 after)
**End-Trend:** exit from any ADX trend (weak/strong, long/short) → 0 (orange ✕ in 'ADX Di+ Di- [Gu5]')
**Forward window:** 100 candles required after signal
**Tie rule:** same-candle TP&SL → loss (conservative)
Generated: 2026-10-19 08:18 UTC
Coins: 20 | Signals analyzed: 403 | Dropped (no future data): 8

| SL\TP | 10% || 20% || 30% || 40% || 50% || 60% || 70% || 80% || 90% || 100% |
|-------|-------||-------||-------||-------||-------||-------||-------||-------||-------||-------|
| **2%** |  23.8 |   7.9 |   5.2 |   1.7 |   0.5 |   0.0 |   0.0 |   0.0 |   0.0 |   0.0 |
| **3%** |  29.3 |   9.7 |   6.0 |   2.0 |   0.7 |   0.0 |   0.0 |   0.0 |   0.0 |   0.0 |
| **4%** |  34.0 |  12.7 |   7.4 |   2.7 |   1.0 |   0.0 |   0.0 |   0.0 |   0.0 |   0.0 |
| **5%** |  35.2 |  13.4 |   7.7 |   3.0 |   1.2 |   0.2 |   0.2 |   0.2 |   0.2 |   0.2 |

**Total signals analyzed:** 403
**Dropped (no 100 candles of future data):** 8
**End-Trend window:** `[i-3, i+3]` (3 before + cross candle + 3 after)
**End-Trend meaning:** exit from any ADX trend (weak/strong, long/short) -> 0 (orange ✕)
**Forward window:** 100 candles
**Tie rule:** same-candle TP&SL -> loss (conservative)

## Per-coin signals
- **BTC**: signals=33 · dropped=1 · candles=4184
- **ETH**: signals=22 · dropped=0 · candles=4181
- **BNB**: signals=21 · dropped=0 · candles=4184
- **SOL**: signals=21 · dropped=0 · candles=4183
- **XRP**: signals=27 · dropped=0 · candles=4184
- **DOGE**: signals=22 · dropped=0 · candles=4182
- **ADA**: signals=21 · dropped=1 · candles=4182
- **AVAX**: signals=25 · dropped=1 · candles=4182
- **TRX**: signals=23 · dropped=2 · candles=4184
- **LINK**: signals=20 · dropped=1 · candles=4182
- **DOT**: signals=21 · dropped=1 · candles=4183
- **MATIC**: _skipped (candles=0)_
- **TON**: signals=22 · dropped=0 · candles=4144
- **SHIB**: signals=25 · dropped=0 · candles=4184
- **LTC**: signals=27 · dropped=0 · candles=4183
- **BCH**: signals=24 · dropped=0 · candles=4184
- **UNI**: _skipped (candles=0)_
- **XLM**: signals=27 · dropped=0 · candles=4182
- **NEAR**: signals=22 · dropped=1 · candles=4183
- **APT**: _skipped (candles=0)_