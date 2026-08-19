# 线程回报

执行线程：
- 业务线程-python/research-postprocess

任务ID：
- 0807T002

状态：
- 已通过

是否进行QA验收：
- 是

QA说明：
- 第二轮独立 QA 已通过，P0-P3 findings 均为 0。验收报告为
  `.workflow/reports/0807T002-qa.md`。

files：
- `.workflow/tasks/0807T002.md`
- AMD immutable raw：
  `/home/molly/project/hftbacktest/local_live_analysis/0807T001_skhynix_4h_continuous`
- AMD passing working campaign：
  `/home/molly/project/hftbacktest/local_live_analysis/0807T002_skhynix_4h_postprocessed_campaign`
- AMD first-failure evidence：
  `/home/molly/project/hftbacktest/local_live_analysis/0807T002_skhynix_4h_postprocessed_campaign_failed-binance-hard-gate`
- AMD full pipeline：
  `/home/molly/project/hftbacktest/local_live_analysis/0807T002_skhynix_basis_postprocess`
- local compact evidence：
  `/Users/liu/Documents/hftbacktest/local_live_analysis/0807T002_skhynix_postprocess_compact`

action：
- 已完成 Skill、output contract、workflow 和 runtime source 阅读。
- 定位第一次失败为默认 `binance_reconnect_nonzero` hard gate。
- 实现显式 Binance recovered reconnect proof、timeline boundary/state
  clear、R0 epoch/degraded sidecar、R1 exact mask admission 和 basis 双
  venue history reset。
- 第一轮独立 QA 定位到 opt-in 下 Binance reconnect count fail-open 风险。
  已补齐 reconnect/attempt/subscription/bridge/disconnect/bootstrap 严格
  对账、缺失/伪造 0 hostile tests、R0/R1 Binance epoch sidecar 和
  timeline epoch 一致性检查。
- 预 QA passing campaign/pipeline/compact 分别以
  `pre-qa-p1-gap` 后缀保留；最终路径使用修复后代码从 immutable raw
  全量重建。
- 保留默认 fail-closed；伪造 bridge、缺失恢复数据和超时恢复仍拒绝。
- 在全新 AMD working copy 完成 supervisor postprocess、Skill inspect、
  `basis-research run`、`validate`、`report` 和 `resume`。

verify：
- 本机与 AMD 相关回归均为 `108 passed`，`py_compile` 和 scoped
  `git diff --check` 通过。
- immutable raw 在本机与 AMD 均为同一组 `39` 文件；运行前后 inventory
  SHA256 均为
  `a30654f32cea375c49e4b26e39ed3c0de436e6d1f08de959601db749f262a003`。
- strict quality `passes=true`，共 `7` degraded intervals：
  `4` core reconnect、`3` auxiliary reconnect。
- Binance count contract 为 reconnect `2`、attempt/subscription/bridge/
  bootstrap 各 `3`、disconnect event `2`，全部闭合。
- Binance 两次 core mask 为 `596.505957ms` 和 `919.407405ms`；
  timeline 记录 `3` Binance epochs 与 `2` reconnect boundaries。
- common L2 `501,220` 行，source-age gate 无 stale row，旧 L2 未跨
  reconnect forward-fill。
- R0：Binance hot `3,998,895`、Hyperliquid hot `207,634`、auxiliary
  `42,704`、mask rows `8`。
- R1 exact masks/horizon masks/reconciliation 全部通过；horizon mask
  exclusions `10,214`，cross-epoch/future/timestamp errors 均为 `0`。
- R1 schema 为 `cross_exchange_alignment_acceptance_v4`，label schema
  为 `cross_exchange_frozen_decision_labels_v4`，共 `714,063` 行。
- basis：`2,751,082` state rows、`2,745,399` book eligible、
  `1,976,897` feature eligible。
- basis 对 Binance/Hyperliquid 旧 epoch BBO 分别抑制 `816`/`4` 条；
  两个 venue 的 epoch regression 和 old-state leak 均为 `0`。
- validator `passes=true`、`stage_count=4`；resume
  `reused_stage_count=4`、`source_immutable=true`。resume 后最终 pipeline
  manifest SHA256 为
  `6d5e7ec31a461569dec9392945263a4e7e1a4221d3a5f8ef1206691ebcffc4d0`。

done：
- 0807 4H 数据已形成可审计的 segmented replay、R0、R1 和
  point-in-time basis/dislocation 数据集。
- 完整 artifact 留在 amdserver；本机 `83` 文件、约 `1.3MB` compact
  evidence 可审阅 manifest、report、quality 和 provenance。compact
  明确省略大型 `.csv.gz`，不能独立运行完整 validator。

blockers：
- 无。

commit：
- 无

提交信息：
- 无
