# 最新 QA 验收结果

## 0722T048

状态：`已通过`

- P0/P1/P2：均无。
- Dynamic expected 模式要求 profile `two-sided-dynamic-manager` 和显式
  `--enable-dynamic-spread`；fixed profile 继续 fail-closed。
- Acceptance 测试：`245 passed in 9.39s`。
- T047 repaired acceptance：exit `0`。
- Provenance `113/0`、config/control `76/0`、decision `43/0`、
  lifecycle `78/0`、economics `6/0`、optimism `6/0`。
- T047 terminal checksum `110/110`，runtime source `63/63`，estimator 和
  fill-feedback replay 均 snapshot match。
- T047 只有一个实际 live window，`2` submissions、`0` fills、final open
  orders `0`、BTC position `0.0`、estimated loss `0.0 USDC`。
- Dynamic activation 为 true，但 candidate 是 `fallback_fixed`，reason 为
  `cold_start_or_invalid_side_intensity_fit`；actual quote behavior 未变化。
- 不支持 stable PnL、fill-rate、maker viability、promotion 或 multi-level
  claim。
