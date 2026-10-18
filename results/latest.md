# Backtest — MACD cross-up + ADX red triangle (4h)
**Rule:** triangle in `[i-15, i-1]` (1..15 candles BEFORE cross, not cross, not after)
**Forward window:** 100 candles required after signal
**Tie rule:** same-candle TP&SL → loss (conservative)
Generated: 2026-10-18 18:10 UTC
Coins: 20 | Signals analyzed: 363 | Dropped (no future data): 8

| SL\TP | 10% || 20% || 30% || 40% || 50% || 60% || 70% || 80% || 90% || 100% |
|-------|-------||-------||-------||-------||-------||-------||-------||-------||-------||-------|
| **2%** |  16.5 |   7.4 |   5.2 |   2.2 |   0.8 |   0.3 |   0.0 |   0.0 |   0.0 |   0.0 |
| **3%** |  22.3 |  10.5 |   6.6 |   2.8 |   1.4 |   0.6 |   0.0 |   0.0 |   0.0 |   0.0 |
| **4%** |  24.8 |  11.6 |   7.4 |   3.3 |   1.9 |   0.8 |   0.0 |   0.0 |   0.0 |   0.0 |
| **5%** |  27.8 |  13.8 |   9.1 |   3.9 |   2.2 |   1.1 |   0.0 |   0.0 |   0.0 |   0.0 |

**Total signals analyzed:** 363
**Dropped (no 100 candles of future data):** 8
**Triangle window:** `[i-15, i-1]` (1..15 candles BEFORE cross)
**Forward window:** 100 candles
**Tie rule:** same-candle TP&SL -> loss (conservative)

## Per-coin signals
- **BTC**: signals=30 · dropped=2 · candles=4186
- **ETH**: signals=23 · dropped=0 · candles=4183
- **BNB**: signals=21 · dropped=0 · candles=4186
- **SOL**: signals=21 · dropped=0 · candles=4185
- **XRP**: signals=28 · dropped=0 · candles=4186
- **DOGE**: signals=21 · dropped=0 · candles=4184
- **ADA**: signals=22 · dropped=1 · candles=4184
- **AVAX**: signals=20 · dropped=0 · candles=4184
- **TRX**: signals=6 · dropped=1 · candles=4186
- **LINK**: signals=18 · dropped=2 · candles=4184
- **DOT**: signals=24 · dropped=1 · candles=4185
- **MATIC**: _skipped (candles=0)_
- **TON**: signals=5 · dropped=0 · candles=4146
- **SHIB**: signals=25 · dropped=0 · candles=4186
- **LTC**: signals=22 · dropped=1 · candles=4185
- **BCH**: signals=22 · dropped=0 · candles=4186
- **UNI**: _skipped (candles=0)_
- **XLM**: signals=33 · dropped=0 · candles=4184
- **NEAR**: signals=22 · dropped=0 · candles=4185
- **APT**: _skipped (candles=0)_