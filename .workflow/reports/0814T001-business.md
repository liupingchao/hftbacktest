# 业务线程回报

执行线程：
- 业务线程-python/research

任务ID：
- 0814T001

状态：
- 待验收

是否进行QA验收：
- 否

QA说明：
- 当前任务结果暂不进入QA验收，待总控确认后再决定是否派发QA验收。

files：
- `examples/hyperliquid/cross_exchange_trigger_aligned_episode_contract.py`
- `examples/hyperliquid/test_cross_exchange_trigger_aligned_episode_contract.py`
- `local_live_analysis/skhynix_trigger_aligned_episode_research_v1/frozen_research_contract.json`
- `local_live_analysis/skhynix_trigger_aligned_episode_research_v1/input_inventory.csv`
- `local_live_analysis/skhynix_trigger_aligned_episode_research_v1/research_manifest.json`
- `local_live_analysis/skhynix_trigger_aligned_episode_research_v1/consumption_ledgers/aug07_access_ledger.json`
- `local_live_analysis/skhynix_trigger_aligned_episode_research_v1/consumption_ledgers/source_inventory_before.json`
- `local_live_analysis/skhynix_trigger_aligned_episode_research_v1/consumption_ledgers/source_inventory_after.json`
- `local_live_analysis/skhynix_trigger_aligned_episode_research_v1/data_admission/channel_inventory.csv`
- `local_live_analysis/skhynix_trigger_aligned_episode_research_v1/data_admission/data_admission.md`
- `local_live_analysis/skhynix_trigger_aligned_episode_research_v1/data_admission/hyperliquid_feed_cadence.csv`
- `local_live_analysis/skhynix_trigger_aligned_episode_research_v1/data_admission/input_manifest_bindings.csv`
- `local_live_analysis/skhynix_trigger_aligned_episode_research_v1/data_admission/session_topology.csv`
- `local_live_analysis/skhynix_trigger_aligned_episode_research_v1/data_admission/unavailable_fields.csv`
- `local_live_analysis/skhynix_trigger_aligned_episode_research_v1/data_admission/underlying_regime_coverage.csv`
- `local_live_analysis/skhynix_trigger_aligned_episode_research_v1/runtime_source/cross_exchange_trigger_aligned_episode_contract.py`
- `.workflow/tasks/0814T001.md`
- `.workflow/reports/0814T001-business.md`
- `task_plan.md`
- `progress.md`
- `findings.md`

action：
- 新增纯 input-freeze/data-admission 生成器，使用 JSON/CSV 结构化 parser、
  canonical serialization、content-addressed manifest、文件锁和目录级原子
  publication。
- 冻结 Jul30、Aug03、Aug04、Aug07 的 canonical source identity、
  manifest/hash、完整文件 inventory、采集时段、采集 runtime 和 topology；
  研究执行 host 与采集 topology 分开记录。
- 对 Jul30/Aug03/Aug04 允许的 Binance/Hyperliquid raw 输入执行完整结构化
  扫描并与 manifest 对账；确认 Binance 历史流为
  `@trade + @depth@0ms + @bookTicker`、trade `q` 存在、`nq/RPI`
  unavailable。
- 发布 Hyperliquid BBO/trades/fast-L2 的 inter-arrival、source-age 和
  no-new-information 分布；Aug07 compact metadata 没有的分位数保持空值并
  明确 unavailable 原因。
- 冻结 KRX timezone/calendar/state contract。因为冻结输入缺少权威 KRX
  holiday/special-session calendar，正式状态为
  `unknown_calendar_state`，仅发布 nominal clock coverage，禁止未来价格
  推断。
- 冻结 Family A/Family B、Candidate/Confirmed landmarks、逐 feature
  `observed_at_ns <= decision_landmark_ns`、episode horizon、outcome、
  interval censoring、evidence labels、primary hypotheses、CRPS/Brier/
  interval-log-loss 和 A0/B0 normalized-loss 合同。
- Aug07 使用 fail-closed first-read policy，只读取 local raw campaign
  manifest、compact metadata/inventory 和 stat 信息；未打开完整
  raw/R0/R1/basis event rows。
- 未构建 Episode v3，未运行 trigger density、detector parity、outcome、
  model 或 actionability，未启动采集，未访问 private/order/cancel。

verify：
- `python examples/hyperliquid/cross_exchange_trigger_aligned_episode_contract.py --help`
  通过。
- `python -m compileall -q examples/hyperliquid/cross_exchange_trigger_aligned_episode_contract.py examples/hyperliquid/test_cross_exchange_trigger_aligned_episode_contract.py`
  通过。
- `python -m pytest -q examples/hyperliquid/test_cross_exchange_trigger_aligned_episode_contract.py`
  通过：`22 passed`。
- `python -m ruff check examples/hyperliquid/cross_exchange_trigger_aligned_episode_contract.py examples/hyperliquid/test_cross_exchange_trigger_aligned_episode_contract.py`
  通过。
- 两次独立完整构建通过；每次发布 `13` 个 manifest-tracked core
  artifacts，核心包 SHA256 均为
  `4267c9506499082bef157991757899601d66741a665345fe75522adf27122752`。
- 第二次构建使用 `--compare-to` 与第一次比较，结果
  `identical=true`；额外比较完整目录确认全部 `14` 个文件的路径、bytes
  和 SHA256 相同。
- 正式包 `--verify-only --compare-to` 通过，`source_inventory_unchanged=true`
  且 `aug07_full_event_rows_opened=false`。
- 正式包 `runtime_source/cross_exchange_trigger_aligned_episode_contract.py`
  与当前生成器 bytes 相同。
- 源 inventory 前后均为 `961` files、`2,835,271,526` bytes，SHA256
  均为
  `fc94f6d3dd8b0d1527c03dae839201ca494807264ecfdaa1218ecdb24aaff3a0`。
- `git diff --check` 通过。

done：
- 四个 session 均完成 input identity、topology、measurement limitation、
  KRX context、HL cadence 和研究合同准入。Jul30/Aug03/Aug04 状态为
  `admitted_with_explicit_measurement_limitations`；Aug07 状态为
  `admitted_contract_freeze_only_event_rows_locked`。
- 显式不可用项包括历史 `nq`、RPI 标记、权威 KRX
  holiday/special-session calendar、采集机时钟同步偏差/不确定性，以及
  Aug07 compact 未提供的 p01/p10、trade source age 和
  no-new-information 分布。
- 输出路径：
  `local_live_analysis/skhynix_trigger_aligned_episode_research_v1/`。
- 当前业务状态为 `待验收`；后续 trigger density、detector parity、
  Episode v3、case retrieval、outcome modeling 和 Aug07 first-read 仍被
  QA 门禁锁定。

blockers：
- 无

commit：
- 无

提交信息：
- 无
