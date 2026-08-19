# 业务线程返修报告

执行线程：
- 业务线程-python/research

任务ID：
- `0815T003`

标题：
- SKHYNIX-JUL30-FAMILY-A-B-EPISODE-V3-BUILD 第一轮有界返修

状态：
- 待验收

日期：
- 2026-08-15（星期六）

是否进行QA验收：
- 是

QA说明：
- 无

commit：
- 无

提交信息：
- 无

## Findings And Evidence First

### P1-1 Confirmation provenance 已关闭

- Family B confirmation-derived T features 现在绑定 shared detector 首次满足
  depletion/drop predicate 的 exact common L2 `TimelineState`。
- `observed_at_ns` 精确等于 `t_confirm_ns`。
- `source_event_id` 使用
  `jul30:{segment_id}:common_l2_timeline:{common_seq}`。
- `source_book_version` 使用该 confirmation state 的 `common_seq`。
- 全量 accepted-source exact 复核：
  - confirmed candidates：`141768`
  - confirmation feature rows：`1984752`
  - whole-row mismatch：`0`
  - observed-at mismatch：`0`
  - source-event mismatch：`0`
  - source-book-version mismatch：`0`
- 旧的内部自洽包被新 admission 拒绝：
  `confirmation_features_family_b source-semantic drift`。

### P1-2 Frozen detector burst 已关闭

- `confirmed_burst_*_through_decision` 不再从 burst start 扫描到 confirm。
- producer/admission 从 accepted Stage 3 burst start/end/count/qty 和 Jul30
  Binance trade source 重建唯一 fixed-origin 10ms burst；opposite side、
  zero-economic reset 或 origin-relative gap 会终止 membership。
- through-decision prefix 只取该 frozen burst 内 `trade_ts <= t_confirm` 的
  rows。
- 全量 `141768` confirmed candidates：
  - frozen burst reconstruction failure：`0`
  - confirmed burst count mismatch：`0`
  - detector touch-through-decision mismatch：`0`
- 第一轮 QA 示例 `jul30:segment_0001:14` 已从错误的 `73` 修复为 detector
  truth `57`。

### P1-3 Candidate 左截断已关闭

- 所有 post-trigger first-event interval 从 Candidate 左截断。
- event scan 使用 `event_ts > t_candidate_ns`；不把 Candidate 前事件当成
  post-trigger non-event observation。
- 全量正式 outcomes：

| outcome | interval | right | segment | quality | lower < Candidate | lower = Candidate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| adverse target BBO | 214167 | 54253 | 96 | 6 | 0 | 110408 |
| target trade at/through quote | 206784 | 61613 | 119 | 6 | 0 | 115793 |
| impacted-side retreat | 206455 | 61960 | 101 | 6 | 0 | 76587 |

- interval semantics 仍为区间观测；未点化、未填零。

### P1-4 Source-semantic trust boundary 已关闭

- admission 不再把 package artifact/core/manifest 自洽当作 source truth。
- source 与 archived verifier 都从 accepted Stage 2/3 candidate truth、
  Jul30 R0 hot events、common timeline 和 quality intervals 重建并 exact
  比较：
  - anchors，包括
    `classification := accepted Stage 3 attribution`、quality/degraded
    flags 和 interval IDs；
  - Family A/B sparse first/last/count；
  - Family A/B full fixed-grid rows，包括 source age/degraded evidence；
  - Family A/B event-count exact source identities；
  - 全部 public-market outcome values、signs、predicates、first-event
    intervals 和 markouts；
  - 全部 Family B confirmation-derived T feature rows。
- classification contract 已冻结：
  - source：
    `accepted_stage3_candidate_audit_projection.attribution`
  - mapping：`classification := detector attribution`
  - allowed values：
    `cancel_driven / mixed / trade_driven / uncertain`
  - 正式包 classification mismatch：`0`
  - 正式包 quality-flags mismatch：`0`

Production-size coherent-rehash matrix：

| mutation | source verifier | archived verifier | failure boundary |
| --- | --- | --- | --- |
| classification relabel | rc=2 | rc=2 | classification frozen value/mapping |
| corrected degraded evidence deletion | rc=2 | rc=2 | anchors source-semantic digest |
| sparse count +1 | rc=2 | rc=2 | sparse-range source-semantic digest |
| nonexistent event ID | rc=2 | rc=2 | event-count source-semantic digest |
| markout +1 bps | rc=2 | rc=2 | outcomes source-semantic digest |
| source-age deletion | rc=2 | rc=2 | required source-age integer invariant |

- 每个 attack 都同步重算了 artifact records、core SHA 和 canonical
  manifest；拒绝原因来自 frozen/source semantics，不来自陈旧 package
  hash。
- archived verifier 继续与当前固定 worktree runtime source/tests exact
  byte identity 绑定。

### P2-1 Strict-pre baseline 已关闭

- vulnerable Hyperliquid quote baseline 使用
  `local_receipt_ts_ns < t_candidate_ns`。
- cross-venue baseline 使用
  `common_ts_ns < t_candidate_ns`。
- `event_ts == t_candidate_ns` 既不能成为 baseline，也不能被当作
  post-trigger first event。
- 新增 synthetic same-receipt-timestamp hostile regression；若 Candidate
  同时出现一个极端 target BBO，markout 和 first-event 仍使用前一条
  strict-pre quote。

## Code Changes

- `examples/hyperliquid/cross_exchange_trigger_aligned_episodes.py`
  - contract/calculation version 升级到 v2；
  - 新增 strict-pre source-store lookup；
  - 新增 exact confirmation TimelineState reconstruction；
  - 新增 frozen burst reconstruction 和 confirmation feature rows；
  - first-event Candidate 左截断；
  - source-semantic row-stream admission；
  - classification mapping/value contract；
  - interval lower-bound 和 confirmation provenance structural assertions。
- `examples/hyperliquid/cross_exchange_jul30_episode_v3_admission.py`
  - 明确为 source-semantic admission；
  - 正向输出 `source_semantic_verified=true`。
- 两个 focused test files
  - detector confirmation provenance；
  - frozen burst termination；
  - Candidate left truncation；
  - equal-timestamp strict-pre baseline；
  - source-semantic coherent mutation；
  - admission success/failure propagation。
- 正式 package 通过 staging 完整原子重建；未手改发布包。

## Formal Package Identities

- package：
  `local_live_analysis/skhynix_trigger_aligned_episode_research_v1_stage04_jul30_episode_v3/`
- schema：`episode_v3_jul30_v1`
- contract version：`skhynix_jul30_episode_v3_contract_v2`
- contract SHA256：
  `167acb400d6b288a36aa35b4b10b813d4527cbf0eabd8f5f366a3a46e3a767e5`
- core package SHA256：
  `5134f1fd31333f0c4c6a972e524b03e2a34bb4435abe6beddda4c62e34e63586`
- full inventory SHA256：
  `c581b65420a08ee2ec658369ecbfde240228127d1a70a243829fa1279feb77e6`
- manifest SHA256：
  `d00a68bc038ff21a9ec706d3403c983809f92ad9da914cf810fe377b72138764`
- artifacts / files / bytes：
  `106 / 107 / 1561201692`

Runtime/test bindings：

- builder：
  `9b8130f47111b7caab514ed486017ce196cc9d68a4966c11339b7a09cfe32dde`
- standalone admission：
  `e439b242e967b8d16d7a7e64e997859bf4e54bc6cfe19ffcb5e5fa5212b3ad60`
- builder focused tests：
  `d3685323210c05ea395019d46009ecaf4148daa387784815acf178df5c28bd18`
- admission focused tests：
  `1a39c333d0601d493b1fdcf0ea1e3ab6c24b085f16f7dd9b1c31dee22fe4e53c`

## Cardinality And Invariants

- anchors / Family A / outcomes：`268522 / 268522 / 268522`
- Family B / confirmed：`141768 / 141768`
- rejected Family A：`126754`
- sparse range A/B：`268522 / 141768`
- fixed grid A/B：`4564874 / 2410056`
- event-count A/B：`805566 / 425304`
- feature ledger A/B：`23092892 / 15310944`
- unavailable feature rows：`2863855`
- clusters / flows / 2000ms overlap blocks：
  `39928 / 10536 / 9`
- auxiliary-degraded grid rows：`4316`
- core-degraded grid rows：`0`
- interval / right / segment / quality / epoch censor：
  `627406 / 177826 / 316 / 18 / 0`
- point-coerced interval：`0`
- future feature observation mismatch：`0`
- anchor ordering mismatch：`0`
- cross-segment / cross-epoch path mismatch：`0 / 0`
- synthetic rejected confirm：`0`

## Determinism

- Formal：
  `local_live_analysis/skhynix_trigger_aligned_episode_research_v1_stage04_jul30_episode_v3/`
- isolated Build A：
  `/tmp/0815T003-r1-build-a/package`
- isolated Build B：
  `/tmp/0815T003-r1-build-b/package`
- 三者均独立完成 8 segments、结构 admission 和 source-semantic
  admission。
- 三者 `107` 个 relative paths、raw bytes、per-file SHA、manifest、
  contract、core 和 full inventory 全部一致。
- 三者共同 identities：
  - core：
    `5134f1fd31333f0c4c6a972e524b03e2a34bb4435abe6beddda4c62e34e63586`
  - full：
    `c581b65420a08ee2ec658369ecbfde240228127d1a70a243829fa1279feb77e6`
  - manifest：
    `d00a68bc038ff21a9ec706d3403c983809f92ad9da914cf810fe377b72138764`

## Verification

Positive verification：

- Stage 4 focused：
  `17 passed in 0.08s`
- Stage 1/2/3 inherited：
  `216 passed in 136.02s`
- R0/alignment/recovery inherited：
  `83 passed in 0.56s`
- total pytest evidence：
  `316 passed`
- source default-Python admission：
  `verified=true`, `source_semantic_verified=true`
- archived default-Python admission：
  `verified=true`, `source_semantic_verified=true`
- default-Python verify-only before/after package snapshot：
  - entries：`128 / 128`
  - files：`107 / 107`
  - path/type/bytes/SHA/mtime/ctime exact identical
  - package `__pycache__/.pyc`：`0`
- Ruff：`All checks passed!`
- external `PYTHONPYCACHEPREFIX` compileall：
  `rc=0`, `48` bytecode files
- deterministic gzip：
  `96/96` passed `gzip -t`
- builder/admission `--help`：passed
- scoped `git diff --check`：passed
- atomic partial-publication injection：
  - injected failure observed
  - output directory absent
  - staging directories `0`

Input immutability：

- before/after inventory rows：`394 / 394`
- before/after bytes：`1412604724 / 1412604724`
- before/after identity：
  `b64e3db82746297b4d9684ab1013d6c657221467a3a1083cc82b2124df3205a6`
- exact identical：`true`

Accepted dependencies remain：

- Stage 1 core/full：
  - `9c2756f966a1a05816f9e6a91d7361d76cfea10fc3c587944ca2f0c1d2eb7c96`
  - `c540cc056313716b3bdd2b9c0fe076cda15a7f152b399ae6a1283b3aa8aa6590`
- Stage 2 core/full：
  - `7b3d06c3225f77c9929cdd3fa40d69c0866d83dbb75dd584dcb7db6db22ad3f8`
  - `bdade16c53aed7bba54fdb9408ba3a8a03036d183720f589bda2a762448a3833`
- Stage 3 core/full：
  - `4939d1c1addce493edb2f368297d56b37edd0b123de01497dcdee6e77637eb9b`
  - `ff8e3434672226371051151cea838503877dca79ac7860cf179256362d75e404`

## Hard Boundary

- `jul30_legacy_episode_rows_read=false`
- `aug03_aug04_future_event_rows_read=false`
- `aug07_event_rows_read=false`
- `model_or_score_run=false`
- `case_retrieval_run=false`
- `actionability_run=false`
- `network_accessed=false`
- `private_or_order_endpoint_accessed=false`
- `new_collection=false`
- `own_order_fill_pnl_fields=false`
- input bindings 中 forbidden path：`0`
- research schema 中 forbidden field：`0`
- Stage 4 runtime network imports：`0`
- 未修改 task/status、`task_plan.md`、`progress.md`、`findings.md`、
  QA report/mirror、accepted Stage 1/2/3 或历史输入。

## Residual Risk

- 未发现剩余的 P0/P1/P2/P3 correctness defect。
- source-semantic admission 现在会重放高基数 grid/event/outcome truth，
  验证成本明显高于 package-only hash 检查；这是 fail-closed trust
  boundary 的运行成本，不是研究语义 limitation。
- source-age 删除当前以通用 integer parse error fail closed；错误诊断可在
  后续独立任务改善，但不影响拒绝结果。
- Stage 5、Aug03/Aug04 future-event transfer、Aug07 event rows、case、
  model、score、actionability、own-order lifecycle 和 randomized EV
  仍全部锁定。

## Done

- 第一轮 QA 的五个 findings 均已实现关闭并完成业务线程验证。
- 正式包与两个隔离构建已完成且全量一致。
- 本业务线程状态仅为 `待验收`。
- 需要全新独立 QA 复核后，才可由总控决定是否把 `0815T003` 置为
  `已通过` 并解锁 Stage 5。
