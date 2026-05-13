```md
执行线程：
- 业务线程-python

任务ID：
- 0513T005

状态：
- 待验收

是否进行QA验收：
- 是

QA说明：
- 正常验收。T005 是 Step 2 planning-only 任务，只检查计划是否覆盖 latency / market-data integrity baseline 的指标、样本、artifact、验收方案和后续任务拆分；不检查代码实现或实验结果。

files：
- .workflow/tasks/0513T005.md
- .workflow/reports/0513T005-business.md
- task_plan.md
- progress.md
- findings.md
- .workflow/dashboard.html
- .workflow/dispatch_suggestions.md

action：
- 阅读 T005 任务边界、`task_plan.md` Step 2、`progress.md`、`findings.md`。
- 只读检查现有工具和 artifact 结构：
  - `align_live_run.py`
  - `latency_from_audit.py`
  - `pipeline_live_raw.py`
  - `compare_audit.py`
  - `maker_acceptance.py`
  - `local_live_analysis/5-13-day-control-15min`
  - `local_live_analysis/5-11-night-active`
  - `local_live_analysis/5-10-day-control-1h-06`
  - `local_live_analysis/5-9-noon`
  - `local_live_analysis/5-9-small`
- 制定 Step 2 的实施计划、指标清单、样本优先级、输出 artifact 设计、验收方案和后续任务拆分。
- 未修改任何 strategy / backtest / deploy / core / converter 代码。
- 未运行新 replay，未启动 live，未采集新样本。

Step 2 实施计划：
- 阶段 A：artifact inventory / reproducibility check
  - 对每个样本建立 input manifest：audit live CSV、raw gzip、converted npz manifest、normal replay report、audit replay report、maker acceptance、latency stats、archive sha256。
  - 区分 pre-T004 legacy samples 与 T004-standard samples。现有 `5-13-day-control-15min` 是最新 no-rule sanity sample，但采集发生在 T004 前，所以可用于诊断，不可作为 T004-standard run-live 流程证明。
  - 后续 fresh live 样本必须使用 T004 标准流程：git sync -> run-local config -> `run_live.sh` preflight -> `deployment_manifest.json` / `start_marker.json` / `stop_marker.json` -> align -> acceptance。
- 阶段 B：latency baseline
  - 从 live audit / `live_order_latency_stats.json` 计算 order entry latency：`exch_ts - req_ts`。
  - 计算 order response latency：`resp_ts - exch_ts`。
  - 计算 feed latency：`feed_latency_ns`，并按 p50/p90/p99/max、>50ms、>250ms、>1s bucket 输出。
  - 计算 event-to-order latency：`req_ts - (ts_exch + feed_latency_ns)`，无法可靠计算时必须标为 `unavailable` 并说明字段原因。
  - 计算 strategy compute latency：`auditlatency_ms`。
  - 计算 cadence / jitter：decision `ts_local` deltas、nearest live-to-BT lag、replay lag bucket、startup-excluded lag gate。
  - 输出 latency regime：normal / degraded / unusable。阈值先作为诊断阈值，不作为 live promotion gate。
- 阶段 C：market-data integrity baseline
  - raw gzip 层：检查 Binance `depthUpdate` 的 `U/u/pu` 连续性、bookTicker 事件存在性、raw gzip EOF/trailer 状态、raw event local/exchange timestamp 范围。
  - converted npz 层：记录当前 schema 只能保留 `ev|exch_ts|local_ts|px|qty|order_id|ival|fval`，不能保留 `U/u/pu` / `lastUpdateId`；因此 raw update-id 诊断必须读 raw gzip 或另建 core/data task。
  - live/replay audit 层：比较 best bid/ask、target bid/ask tick、top5 tick、top5 qty、top5 sum abs diff、top5 first divergence、market/top5 source fields、overlay source fields。
  - bookTicker/depth consistency：优先在 raw 层做 best bid/ask consistency bucket；如果当前 raw parser 不够，必须输出 `requires raw/core task`，不能伪造字段。
  - startup exclusion：单独报告 leading outside-dual-gate rows、post-startup outside-dual-gate rows、startup top5 divergence rows。
- 阶段 D：sample usability classification
  - `compressed_action_path_only`：action/planned/reject/throttle/working-order/replay-lag hard gates可用，但 top5/full L2/provenance 不足。
  - `pricing_research_candidate`：除 action-path gates 外，best bid/ask drift、target tick drift、latency regime、market source fields、startup exclusion 都可解释；top5 若仍不稳定，必须限制 pricing features 不使用 top5/OFI/queue。
  - `queue_fill_research_candidate`：必须有 raw update-id continuity、bookTicker/depth consistency、per-decision top-N provenance、latency/jitter regime、top5 quality；当前已知样本大概率不能直接达到这个等级。
  - `unusable`：raw/audit artifact 缺失、lag gate不可解释、schema/source fields缺失、raw discontinuity严重，或样本不是 no-rule/control 但被误用。
- 阶段 E：decision output
  - 输出每个样本可用于什么研究，不可用于什么研究。
  - 明确下一步进入 Step 3 market-view acceptance gate、core/data provenance task，或 fresh T004-standard no-rule live sample task。

指标清单：
- Latency：
  - feed latency p50/p90/p99/max、bucket rows。
  - order entry p50/p90/p99/max、entry_gt_5ms_ratio。
  - order response p50/p90/p99/max。
  - event-to-order p50/p90/p99/max 或 unavailable reason。
  - `auditlatency_ms` p50/p90/p99/max。
  - decision cadence delta p50/p90/p99/max。
  - replay lag / exchange replay lag p50/p90/p99/max、in-dual-gate rows、outside-dual-gate rows。
- Market data：
  - raw depth update continuity：`U/u/pu` gap count、duplicate count、out-of-order count、first gap context。
  - bookTicker/depth consistency：same-side best bid/ask match rate、crossed/locked count、drift tick bucket。
  - best bid/ask drift live vs normal replay / audit replay。
  - target bid/ask tick drift。
  - top5 tick match rate、top5 qty match rate、bid/ask sum abs diff p50/p90/p99/max、first divergence context。
  - market_view_source / top5_source / overlay source counts。
  - raw gzip EOF/trailer/data-quality flags。
- Provenance：
  - deployment manifest present/absent。
  - config hash / schema hash / commit hash if T004-standard manifest exists。
  - raw manifest path and data file list.
  - audit schema field count and required source fields present/absent。

样本优先级：
- P0 `5-13-day-control-15min`
  - 最新 no-rule sanity sample，已验证 T002 provenance field chain。
  - 用于 Step 2 快速验证 plan/scripts 的输出格式和 source-field semantics。
  - 边界：pre-T004 采集，没有 T004 deployment manifest，不能证明标准 run-live 流程。
- P1 `5-11-night-active`
  - 4H 长样本，事件量大，适合 latency regime、top5 mismatch bucket、cancel/fill lifecycle 背景统计。
  - 边界：pre-T004 legacy sample，不能单独作为 fresh deployment reproducibility proof。
- P2 `5-10-day-control-1h-06` / `5-9-noon` / `5-9-small`
  - cross-sample sanity checks，覆盖不同时段和已知 replay/action-path acceptance 样本。
  - 用于验证指标是否在多样本上稳定，不用于单样本 live promotion。
- P3 later fresh T004-standard no-rule control sample
  - 如果 Step 2 需要最终 deployment-provenance baseline，后续单独建任务采集。
  - 必须先通过 T004 preflight manifest。

输出 artifact 设计：
- 目标目录建议：`local_live_analysis/step2_market_data_baseline_<TASK_ID>/`。
- `sample_inventory.json`
  - 每个样本的 input files、archive sha256、T004 manifest presence、legacy/fresh 标记。
- `latency_summary.json` / `latency_summary.csv`
  - 每个样本 latency 指标和 latency regime。
- `market_data_integrity.json`
  - raw continuity、bookTicker/depth consistency、converted npz schema limits、audit/replay drift。
- `top5_mismatch_buckets.csv`
  - 按 sample、side、tick_match、qty_match、sum_abs_diff bucket、startup/post-startup 分组。
- `source_provenance_summary.json`
  - market/top5 source 字段分布、overlay 状态、缺失字段。
- `sample_usability_matrix.csv`
  - sample -> `compressed_action_path_only` / `pricing_research_candidate` / `queue_fill_research_candidate` / `unusable`。
- `step2_baseline_report.md`
  - 人可读结论：哪些样本能用于什么，哪些不能用于什么，下一步任务建议。

验收方案：
- T005 planning 验收通过条件：
  - 计划覆盖 latency、market-data integrity、provenance、样本优先级、artifact 输出、样本可用性分类、后续任务拆分。
  - 明确不实现代码、不运行 replay、不启动 live。
  - 明确 current audit/replay 不足以证明 full L2 / queue / OFI / microprice。
- 后续 Step 2 implementation 验收通过条件：
  - 指定样本全部生成上述 JSON/CSV/MD artifacts。
  - 每个样本都有 explicit usability classification 和不可用原因。
  - raw update-id / bookTicker provenance 无法获得时，报告必须输出 `requires core/data task`，不能伪造字段。
  - 不把 action-path acceptance、audit overlay 或 top5 source 字段误解释为 full L2 对齐。
  - 无 live、无 strategy change、无 promotion。
- 后续 Step 2 implementation 未通过条件：
  - 没有区分 legacy sample 与 T004-standard sample。
  - 没有分类样本可用性。
  - 没有 raw/depth/bookTicker 诊断或没有说明字段不可得。
  - 把 `audit_overlay` 或 `replay_depth` top5 当作真实 full L2 proof。
  - 用单样本指标授权 pricing rule / live promotion。
- 阻塞条件：
  - 本地缺失必要 raw gzip / audit CSV / alignment report。
  - raw gzip 无法解析 Binance update ids 或 bookTicker，且没有替代 raw/core task。
  - 当前 Python strategy layer 拿不到后续研究必须字段，如 `U/u/pu`、`lastUpdateId`、bookTicker provenance、per-decision top-N snapshot。

后续任务拆分建议：
- `0513T006`：实现 Step 2 read-only analyzer，对现有 P0/P1/P2 样本生成 latency / market-data integrity / usability artifacts；不启动 live。
- `0513T007`：如果 `0513T006` 发现 raw update-id / bookTicker provenance 不足，制定或实现 core/data provenance task，保留 `U/u/pu`、`lastUpdateId`、bookTicker provenance、per-decision top-N snapshot。
- `0513T008`：如需要 fresh baseline，使用 T004 标准 run-live 流程采集一个 no-rule control sample，再跑 Step 2 analyzer；必须单独授权 live。
- `0513T009`：在 Step 2 artifact 通过后，进入 Step 3，把 `maker_acceptance.py` 升级为 action-path + market-view quality gate。

verify：
- `python3 .workflow/build_dashboard.py` -> 待最终刷新。
- 人工边界检查：
  - 未修改 `examples/binance_tick_mm/*.py`。
  - 未修改 `examples/binance_tick_mm/deploy/*.sh`。
  - 未运行 replay。
  - 未启动 live。

done：
- 已完成 Step 2 planning-only 合同。
- 已明确现有样本的使用边界：它们能用于 latency/market-data 诊断和计划验证，但 pre-T004 样本不能证明标准化 run-live deployment flow。
- 已明确 Step 2 的核心判断：先证明 market-data 和 latency 是否足够可靠，再决定 pricing research、queue/fill research、Step 3 acceptance gate 或 core/data task。

blockers：
- 无。

commit：
- 无

提交信息：
- 无
```
