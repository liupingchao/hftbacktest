# Small-Cap Live-Test Protocol

Task: `0615T008`

This protocol defines the risk gate for a future `0615T009` small-cap live test. It is a design and dry-run artifact only. It does not connect to venues, read credentials, start live processes, place orders, cancel orders, change strategy behavior, or authorize deployment/promotion.

## Caps

- Symbol: `BTCUSDT`
- Duration: `10` minutes maximum
- Max gross notional: `25 USDT`
- Max single order notional: `5 USDT`
- Max position notional: `10 USDT`
- Max loss: `2 USDT`
- Maker-only / post-only: required
- Default-on: forbidden

## Kill Switch

The future live test must stop submitting and cancel all if any trigger fires:

- realized or unrealized loss reaches `2 USDT`
- position notional reaches `10 USDT`
- reject count reaches `3`
- latency p99 reaches `5000 ms`

## Required Evidence

The future `0615T009` task must collect deployment manifest, start/stop markers, live audit, raw market data, authorized private order response artifacts, authorized account inventory artifacts, authorized economics fee/rebate artifacts, shutdown cancel proof, archive, and checksum.

## Approval Boundary

`0615T009` requires explicit total-control approval of the live window. Passing this protocol task alone does not start live and does not authorize scaling, promotion, PnL proof, or maker viability proof.
