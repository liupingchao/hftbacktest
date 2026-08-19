执行线程：
- 业务线程-python/cross-exchange-research

任务ID：
- 0801T007

状态：
- 待验收

是否进行QA验收：
- 是

QA说明：
- 首轮、第二轮独立 QA 均为 `未通过`；本报告已更新为 R2 修复后的第三轮
  验收输入。第二轮只剩 consumption actual-row-count closure 一项 P1。

files：
- `examples/hyperliquid/cross_exchange_liquidity_response_baseline_v2.py`
- `examples/hyperliquid/test_cross_exchange_liquidity_response_baseline_v2.py`
- `local_live_analysis/skhynix_liquidity_response_case_hierarchy/baseline_v2/`
- `.workflow/tasks/0801T007.md`
- `.workflow/reports/0801T007-business.md`
- `task_plan.md`
- `progress.md`
- `findings.md`

action：
- 新增独立 `discovery-freeze` 与 `post-selection-evaluate` CLI。
- discovery 只打开 `segment_0001-0003` 的三份 M1 episode 和三份
  timeline；open-file audit 中 `0004-0008` 文件数为 `0`。
- 从 discovery M1 rows 重建 Episode v2，并与 T006 的 `60,820` atom
  count 和 canonical SHA
  `bce395180f2773011d323a7647efa7513709cdf4190aa89c53572c2f4d220b2d`
  对账。
- 构建 25 个 response targets，以及 pre-state、shock path、
  intervening flow 和 data-quality feature allowlist。
- adverse `h1000/h2000` markout 只写入独立 evaluation artifact；
  adverse/PnL/outcome-like baseline feature 数为 `0`。
- matched-neighbor 使用完整 evidence/label interval purge，minimum
  `50`、maximum `200` neighbors；ACF 使用 per-segment absolute 1-second
  wall-clock bins，不压缩空白秒，至少 `30` 对样本，冻结 embargo 为
  `max(60s, estimated lag)=60s`，并输出 `30/60/120s` sensitivity。
- R2 对 query/candidate 双侧都使用完整
  `[episode_start, max(episode_end, outcome_known_from)]` evidence/label
  interval；首轮 QA 发现的真实 false-eligible neighbor 路径已封闭。
- 仅以 discovery leave-one-segment-out CV 选择 official baseline；
  quantile HistGradientBoosting 在三个 discovery folds 的 median MAE 和
  quantile loss 上均胜出。
- residual scale floor 固定为 discovery target MAD 的 `10%`。
- discovery freeze 增加 exact manifest/contract keys、3+3 input
  cardinality、segment/role set、actual row-count/SHA、opened-file audit、
  output key/path/schema/row-count/SHA closure、model/contract binding、
  input末端 rehash和 candidate full-manifest compare。
- consumption manifest 增加 exact keys、1+5+5 input cardinality、
  segment/role set、actual row-count/SHA 和 frozen contract/model binding。
  intent 在首次 outcome 文件读取前持久化，随后立即扫描十份 CSV 校验
  实际行数；协调更新 M1 row-count、M1/contract/post SHA 也不能绕过
  validator。
- `baseline_manifest.json` 冻结 matched-neighbor availability 和实际分布：
  `82,410` available、`0` unavailable、neighbor count
  `min=186/p50=200/max=200`。
- sklearn HGB 完整 pickle 在真实规模重复拟合中出现无语义字节漂移；
  改为 deterministic gzip JSON portable trees，并实现与 sklearn
  `predict()` 精确一致的 predictor。模型写出前规范化 `-0.0` 为 `0.0`。
- 首轮失败包完整归档于
  `/tmp/0801T007-baseline-v2-pre-r1-failed-qa`，未在原包上伪装升级。
- 第二轮失败的 R1 包完整归档于
  `/tmp/0801T007-baseline-v2-r1-failed-qa`。
- R2 第一次打开 `0004-0008` 数据前，以 exclusive create 写入唯一
  consumption manifest；绑定 M1 manifest、五份 episode 和五份 timeline
  的 row-count/SHA。run ID 为
  `1b0e19dd-7526-4aaa-9127-3d2a69d77ffa`。
- `0004-0008` 的全部 feature、adverse evaluation 和 prediction rows
  只使用 `post_selection` 标签；正式 held-out 授权为 `false`。
- consumption manifest 一旦存在，discovery freeze 永久拒绝覆盖，即使
  post-selection evaluation 在中途失败。

verify：
- `python -m pytest examples/hyperliquid/test_cross_exchange_liquidity_response_case_hierarchy.py examples/hyperliquid/test_cross_exchange_liquidity_response_baseline_v2.py -q`
  -> `32 passed`。
- baseline v2 focused tests -> `10 passed`。
- `ruff`、`py_compile`、CLI help 和 `git diff --check` -> pass。
- hostile tests 覆盖完整 episode interval、`minimum_neighbors=49`、未知
  contract key、伪造 input SHA、neighbor summary drift、row-count drift、
  malformed consumption unknown key/cardinality/row-count，以及协调修改 M1
  row-count 并重算 M1/contract SHA；完整入口均 fail closed。
- portable HGB 与 sklearn prediction 逐值一致；相同模型结构两次写出
  byte-identical，signed-zero 不进入模型包。
- 真实 discovery freeze：
  - episode rows `3,500`；
  - targets `25`；
  - CV predictions `164,820`；
  - calibration rows `150`；
  - embargo sensitivity rows `54`；
  - official baseline `quantile_hist_gradient_boosting`；
  - opened held-out data files `0`。
- 真实 discovery 同 task/config 重复构建成功；程序内 full-manifest gate
  和外部全目录 SHA diff 均完全一致。
- 真实 discovery matched-neighbor summary：
  - query count `82,410`；
  - available `82,410`；
  - unavailable `0`；
  - neighbor count `min=186/p05=200/p50=200/p95=200/max=200`。
- 真实 post-selection：
  - episode rows `9,177`；
  - predictions `216,456`；
  - available predictions `216,456`；
  - opened files `10`，恰为五份 episode 与五份 timeline；
  - consumption inputs `11`，含一个 M1 manifest；
  - consumption `row_count` fields `11/11`；
  - 十份 CSV 实际总行数 `426,625`；
  - exact `held_out` token 在全部 13 个正式 artifacts 中为 `0`。
- post-selection 重复构建复用同一 run ID，前后全目录 SHA diff 为空。
- post-selection 发布前后，八个 discovery artifact SHA 全部不变。
- internal discovery/post-selection validators 对真实 R2 包均返回 pass。
- 25 个 target 的 feature allowlist 共 `772` 项，adverse/PnL/profit/markout
  命中数为 `0`；13 个正式 artifacts 中 exact legacy held-out token 为
  `0`。

done：
- discovery manifest SHA-256
  `8879df16b2a573f2931bb5fa9e9b99247ff355dd66acf335fb850c5217b4b1db`。
- frozen research contract SHA-256
  `5eba39a7cd4d0f47e7ef5dcb236d50bc14c2565a0956cb484cefc5f47a337ce5`。
- portable model bundle SHA-256
  `91b18ea8acf2d0edfe3677a7930a62c3d9bdca7ee30fae1ac036a0f41080c6a9`。
- discovery feature/prediction/calibration/sensitivity SHA 分别为：
  - `7cb6bdcb1a995174b7c7a3c6d4bd5b543520eec75183cb92aa2c16f8d916b4da`
  - `52d52ea0765840ea5c38f74bdcce6900bdddc6611ace204a9777c4a9f72e3c98`
  - `4ec598901ac24eba2328df8d513b4aea7205cf4bea85434f773f71199d944012`
  - `8f7e223fa37e1e4a52ffa37045d7b3e69daf05c700b72d1f02788c41f9496770`
- consumption manifest SHA-256
  `f2ec1cda682e7b3526693a06d278195de5792c8f83b6a748929533a4841454da`。
- post-selection feature/prediction/manifest SHA 分别为：
  - `b649f1a2827dfabbadbb3ab37fdfe95c9b6399f83900526d586b99f13b2c8f39`
  - `664488ee698b597a359500d72d2f96c1cc129a1ec9696366f4261327c9eb04c6`
  - `d9548eeaec69a9c132fe8c6536f082830f2b23511e831a6a344b1105821ad149`
- 本任务只完成 conditional baseline 和 post-selection residual 输入；
  不声明正式 held-out motif 支持、tradable signal、maker identity、fill 或
  PnL 证据。

blockers：
- 无；等待独立 QA。T008 在 QA `已通过` 前保持锁定。

commit：
- 无

提交信息：
- 无
