# 业务线程回报

执行线程：
- 业务线程-python/research

任务ID：
- 0814T001

状态：
- 待验收

是否进行QA验收：
- 是

QA说明：
- 无

files：
- `examples/hyperliquid/cross_exchange_trigger_aligned_episode_contract.py`
- `examples/hyperliquid/test_cross_exchange_trigger_aligned_episode_contract.py`
- `local_live_analysis/skhynix_trigger_aligned_episode_research_v1/`
- `.workflow/tasks/0814T001.md`
- `.workflow/reports/0814T001-business-r1.md`
- `task_plan.md`
- `progress.md`
- `findings.md`

action：
- 修复 `Aug07AccessPolicy`：raw/compact 在 root 判断前同时拒绝请求路径和
  resolved path 的 full-event name/suffix；`raw.gz` 后缀、任意
  `*.csv.gz`、未知 relative path 均 fail closed。
- compact content reads 只允许 exact relative-path metadata allowlist。
  input inventory、collection context、cadence metadata 和 manifest binding
  全部经同一个 policy object 读取。
- Aug07 ledger 由实际成功 content-read records 重算
  `event_rows_opened/event_row_read_count`；research manifest 从 ledger
  取值，verify-only 会独立重算并拒绝自报漂移。
- cadence validator 解析 canonical `missing_fields`；named metric 必须
  为空，`unavailable` 的全部七个 metric 必须为空，
  `partial_compact_metadata` 的 available/missing metric 必须精确互补。
- topology validator 要求 exact 四 session set/cardinality，重建完整
  canonical acquisition payload，精确绑定每个 session 的 collection
  interval、physical topology、collector/supervisor SHA、runtime、channel、
  endpoint、clock、collection mode 和 provenance，并重算 physical/
  collection fingerprints。
- frozen-contract validator 对完整 generated contract 做 recursive
  canonical exact equality，覆盖 Family A/B、landmarks、rejected
  nullability、observed-at、outcomes、interval censoring、hypotheses、
  CRPS/Brier/interval-log-loss、A0/B0 normalized loss 和 Aug07 first-read。
- 永久测试加入 QA 四个 `ACCEPTED_FAIL_OPEN` 反例，并覆盖
  missing-key、extra-key、same-schema、session cardinality、recomputed
  topology mutation 和组合漂移。
- 重新发布正式 package，完成两次独立完整构建和全目录比较。
- 未运行 trigger density、detector parity、Episode v3、case retrieval、
  outcomes、models 或 actionability；未打开 Aug07 full-event rows。

verify：
- `python -m pytest -q examples/hyperliquid/test_cross_exchange_trigger_aligned_episode_contract.py`
  通过：`36 passed`。
- `python -m ruff check examples/hyperliquid/cross_exchange_trigger_aligned_episode_contract.py examples/hyperliquid/test_cross_exchange_trigger_aligned_episode_contract.py`
  通过。
- `python -m compileall -q examples/hyperliquid/cross_exchange_trigger_aligned_episode_contract.py examples/hyperliquid/test_cross_exchange_trigger_aligned_episode_contract.py`
  通过。
- `python examples/hyperliquid/cross_exchange_trigger_aligned_episode_contract.py --help`
  通过。
- 正式 package `--verify-only` 通过。
- 正式 package、`/tmp/0814T001-r1-build-a` 和
  `/tmp/0814T001-r1-build-b` 全部 `14` 个文件 path/bytes/SHA256 一致；
  full-directory inventory digest 为
  `9166ea429a61dd7481de826808257b1c85d14280d98ca526f2e8d710fa693c4c`。
- 三次构建的 core package SHA256 均为
  `9c2756f966a1a05816f9e6a91d7361d76cfea10fc3c587944ca2f0c1d2eb7c96`。
- runtime source 与当前 generator SHA256 均为
  `f675d8cff62432e4be7fb2560345213ac85bfd203c40be75a495ee0c7132a426`。
- repaired exact compact allowlist 下，三次 source inventory before/after
  均为 `890` files、`2,834,242,005` bytes、SHA256
  `4b9c6f53ff43cefa3f606c8c8d78e403e5eda230d0044dc35e0b1ed1d40d8908`。
- 旧包 `961` files 包含 compact 根目录的非 allowlist metadata；新 count
  是 admission scope 收紧结果，不是 source mutation。
- 三次 Aug07 ledger 均为 `51` actual metadata reads、`13` unique exact
  allowlist paths、`39` raw stat-only paths、`0` forbidden content reads、
  `event_row_read_count=0`、`event_rows_opened=false`。
- 三次 `input_inventory.csv` 的 Aug07 raw 均为 `39` rows，所有
  `content_opened_for_sha256=false`。
- `git diff --check` 通过。

done：
- Jul30、Aug03、Aug04、Aug07 exact session set/cardinality 和 canonical
  acquisition payload 均通过。前三者为
  `admitted_with_explicit_measurement_limitations`；Aug07 为
  `admitted_contract_freeze_only_event_rows_locked`。
- 显式不可用项保持为历史 `nq`、RPI marker、权威 KRX holiday/special
  session calendar、collector clock sync offset/uncertainty，以及 Aug07
  compact 未提供的 cadence/source-age/no-new-information metrics。
- 正式输出路径：
  `local_live_analysis/skhynix_trigger_aligned_episode_research_v1/`。
- 当前任务状态为 `待验收`，等待独立 re-QA；所有后续研究门禁继续锁定。

blockers：
- 无

commit：
- 无

提交信息：
- 无
