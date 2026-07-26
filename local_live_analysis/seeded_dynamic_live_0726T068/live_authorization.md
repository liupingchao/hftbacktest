# T068 Exact Live Authorization

- Authorized at conversation turn on 2026-07-26.
- Earliest start:
  `2026-07-26T09:00:00Z`.
- Host:
  `awsserver1 / i-02c64c088f311cbc1 / ap-northeast-1`.
- Source:
  `cross-exchange/a0bc92898ecea43cbdc4219efc1acbcd69580969`.
- Venue/symbol:
  Hyperliquid `BTC`.
- Profile:
  `two-sided-seeded-dynamic-manager`.
- Windows:
  three sequential `1800s` windows with artifact IDs `01/02/03`.
- Execution:
  real post-only `Alo` submit and cancel are authorized.
- Credential/account reads:
  `/home/admin/XEMM_rust_latest/.env`, private account and open-orders reads
  are authorized. Raw credentials must not be pulled locally.
- Per-window caps:
  `0.005 BTC` max order size, `2` max submissions, `0.01 BTC` max position,
  `1 USDC` max loss.
- Required identity guard:
  order-submit account must equal fill-pullback account and post-state account.
