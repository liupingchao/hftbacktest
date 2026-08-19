# 业务线程返修报告

执行线程：
- 业务线程-python/research

任务ID：
- `0815T003`

标题：
- SKHYNIX-JUL30-FAMILY-A-B-EPISODE-V3-BUILD 第二轮有界返修

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

### P1 Complete Feature Projection 已关闭

- `_verify_source_semantics()` 不再只筛选 `14` 个 confirmation feature
  names。
- producer 与 verifier 现在复用同一个 source-derived `_feature_rows()`
  construction，按 segment、candidate、frozen feature order 流式重建：
  - Family A：`23092892` rows；
  - Family B：`15310944` rows。
- 每条 expected row 与 published observed row 比较完整 `FEATURE_FIELDS`：
  - `episode_id`
  - `candidate_id`
  - `family_view`
  - `decision_landmark`
  - `feature_name`
  - `value`
  - `observed_at_ns`
  - `source_event_id`
  - `source_book_version`
  - `calculation_version`
  - `availability_reason`
- 实现使用逐行 exact equality 和固定字段顺序、长度前缀的流式 SHA256。
  不持有全量 feature rows；内存边界为当前 segment source arrays、
  candidates 和常数个 row/digest state。

最终 source 与 archived verifier 的全量结果：

| projection | expected rows | observed rows | expected SHA256 | observed SHA256 | mismatch |
| --- | ---: | ---: | --- | --- | ---: |
| Family A features | 23092892 | 23092892 | `6cea08b2fa07d440722ae5e697459896eafa9c6461e70dec06968726620ddec1` | `6cea08b2fa07d440722ae5e697459896eafa9c6461e70dec06968726620ddec1` | 0 |
| Family B features | 15310944 | 15310944 | `2cea4acf84f78754cbf78ad258458b488a6133e145c50878f1fabff0d8335b9f` | `2cea4acf84f78754cbf78ad258458b488a6133e145c50878f1fabff0d8335b9f` | 0 |

### P1 Complete View Projection 已关闭

- Family A/B view rows 现在从 immutable candidate truth 直接调用
  `_view_row()` exact 重建。
- 完整比较全部 `VIEW_FIELDS`，覆盖 identity、landmark、population、
  quality eligibility/reason 和六个 artifact linkage fields。

| projection | expected rows | observed rows | expected SHA256 | observed SHA256 | mismatch |
| --- | ---: | ---: | --- | --- | ---: |
| Family A views | 268522 | 268522 | `97b9d7779217eb3752c335273324611ecaa0070121edba4c42d9eb63c0aec72a` | `97b9d7779217eb3752c335273324611ecaa0070121edba4c42d9eb63c0aec72a` | 0 |
| Family B views | 141768 | 141768 | `5146b19b4f030b7aa07ca69ce78b524b5c3e2f70464ec12cb5a5cad87cb827b3` | `5146b19b4f030b7aa07ca69ce78b524b5c3e2f70464ec12cb5a5cad87cb827b3` | 0 |

### Frozen Contract 和 Admission 声明已闭合

- contract version：
  `skhynix_jul30_episode_v3_contract_v3`。
- `exact_feature_projection` 明确冻结：
  - all `FEATURE_FIELDS`;
  - all `86` Family A feature names；
  - all `108` Family B feature names；
  - all feature rows in both families；
  - exact row counts `23092892 / 15310944`。
- `exact_view_projection` 明确冻结：
  - all `VIEW_FIELDS`;
  - both families；
  - exact row counts `268522 / 141768`；
  - population/linkage/eligibility/quality 全 payload。
- standalone admission 只有在 `verify_package()` 返回完整 scope 与 aggregate
  evidence 时才输出 `source_semantic_verified=true`；缺少或缩小 scope 会
  fail closed。
- source-semantic exact replay 前置于普通结构汇总，使 segment-local
  coherent mutation 尽早终止，同时不删除任何原结构验收。

## Production-Size Coherent-Rehash Matrix

全部攻击：

- 从最终正式 package 创建完整 APFS clone；
- deterministic gzip 重写目标 artifact；
- 同步重算 artifact records、core SHA 和 canonical manifest；
- 分别运行 current source verifier 与 clone 内 archived verifier；
- 两边均使用默认 Python，stderr 均为空。

第二轮新攻击：

| attack | source | archived | exact failure field |
| --- | ---: | ---: | --- |
| QA 原例 Family A `pre_binance_bid_px 948.13 -> 948.14` | rc=2 | rc=2 | `value` |
| Family A nonconfirmation observed-at | rc=2 | rc=2 | `observed_at_ns` |
| Family A nonconfirmation source event | rc=2 | rc=2 | `source_event_id` |
| Family A nonconfirmation book version | rc=2 | rc=2 | `source_book_version` |
| unavailable feature -> zero/available | rc=2 | rc=2 | value/provenance/availability |
| unavailable reason mutation | rc=2 | rc=2 | `availability_reason` |
| calculation version mutation | rc=2 | rc=2 | `calculation_version` |
| Family B nonconfirmation value | rc=2 | rc=2 | `value` |
| Family B nonconfirmation provenance | rc=2 | rc=2 | observed/source/book |
| Family A quality eligibility/reason | rc=2 | rc=2 | eligibility/reason |
| Family A artifact linkage | rc=2 | rc=2 | feature-ledger path |
| Family B quality eligibility/reason | rc=2 | rc=2 | eligibility/reason |
| Family B artifact linkage | rc=2 | rc=2 | outcome path |

第一轮 hostile 零回归：

| attack | source | archived | exact failure field |
| --- | ---: | ---: | --- |
| anchor classification | rc=2 | rc=2 | `classification` |
| degraded evidence deletion | rc=2 | rc=2 | `quality_flags_json` |
| sparse count | rc=2 | rc=2 | `binance_trade_count` |
| fixed-grid source age | rc=2 | rc=2 | `binance_source_age_ns` |
| event-count source identity | rc=2 | rc=2 | event source ID |
| outcome markout | rc=2 | rc=2 | 250ms target markout |
| confirmation value | rc=2 | rc=2 | confirmation feature value |
| confirmation provenance | rc=2 | rc=2 | source event/book version |

矩阵结论：
- 新攻击 `13/13`、source/archive `26/26` fail closed。
- 第一轮回归攻击 `8/8`、source/archive `16/16` fail closed。
- 没有攻击依赖陈旧 artifact/core/manifest hash 触发失败。

## First-Round Repair Zero Regression

### Confirmation State 和 Frozen Burst

- final source-derived Family B whole-row projection mismatch：`0`。
- confirmation features：
  - candidates：`141768`
  - per candidate：`14`
  - rows：`1984752`
  - `observed_at_ns != t_confirm_ns`：`0`
  - common-timeline source-event mismatch：`0`
  - empty confirmation values：`0`
- frozen fixed-origin burst reconstruction 仍由
  `_confirmation_feature_context()` 在 expected source stream 内执行；
  published complete Family B feature digest mismatch 为 `0`。
- targeted detector-state/frozen-burst regression：passed。

### Candidate Left Truncation 和 Strict-Pre

- 三类 first-event final scan：
  - interval-censored：`627406`
  - right-censored：`177826`
  - segment-censored：`316`
  - quality-censored：`18`
  - lower `< t_candidate_ns`：`0`
  - point/empty interval：`0`
- targeted equal-receipt strict-pre、same-receipt observation interval 和
  detector/frozen-burst tests：`3 passed`。

### Existing Exact Streams

最终 source/archive expected/observed whole-row mismatch 均为 `0`：

- anchors `268522`
- sparse A/B `268522 / 141768`
- fixed grid A/B `4564874 / 2410056`
- event count A/B `805566 / 425304`
- outcomes `268522`

## Formal Package Identities

- package：
  `local_live_analysis/skhynix_trigger_aligned_episode_research_v1_stage04_jul30_episode_v3/`
- schema：
  `episode_v3_jul30_v1`
- contract SHA256：
  `7e07af13742edadddcda32784b64e398586b7cff94c7fa9fdeb2d880b41bfb5c`
- core package SHA256：
  `4fc6ce074200a3c08f7a504f0ce2412f5dad8d7ad51aa44e1799f00f6dd886c9`
- full inventory SHA256：
  `1f3ddd61fa0bbbf3268ae6f09b303d4876e83f456d9e6ce63956b1306707dc72`
- manifest SHA256：
  `898d5143a8a47a36b8ec0f8572621cd7a8bc542b692425feba9fcfe4efd08165`
- artifacts / files / bytes：
  `106 / 107 / 1561252064`

Runtime/test bindings：

- builder：
  `3261297b6f277b82f0daa29f3d364839fc887c96085ca1fed964db888075dcb9`
- standalone admission：
  `2b3791f31a1342450cf710ff0e3ba5ea649d3d9ed2f62a9d8fec8e15bd946772`
- builder focused tests：
  `914880f066a0754fa362d72918c443590afebcfdcd3c6d0dbcc0216c74a37c69`
- admission focused tests：
  `b289b1475a97159c08ec8399ffbfe57a7cbd0a1549199dee0e465def4826d312`

## Determinism

- Formal：
  `local_live_analysis/skhynix_trigger_aligned_episode_research_v1_stage04_jul30_episode_v3/`
- isolated Build A：
  `/tmp/0815T003-r2-build-a.51i7Qh/package`
- isolated Build B：
  `/tmp/0815T003-r2-build-b.5wy6UQ/package`
- 三者均独立完成 8 segments、complete source-semantic admission、
  structural admission 和 atomic publication。
- 三者共同：
  - files：`107`
  - bytes：`1561252064`
  - core：
    `4fc6ce074200a3c08f7a504f0ce2412f5dad8d7ad51aa44e1799f00f6dd886c9`
  - full：
    `1f3ddd61fa0bbbf3268ae6f09b303d4876e83f456d9e6ce63956b1306707dc72`
- independent relative path/type/size/raw SHA comparison：
  - Formal == Build A：`true`
  - Formal == Build B：`true`
- residual staging directories：`0`。

## Cardinality And Invariants

- anchors / Family A / outcomes：
  `268522 / 268522 / 268522`
- Family B / confirmed：
  `141768 / 141768`
- rejected Family A：
  `126754`
- sparse A/B：
  `268522 / 141768`
- fixed grid A/B：
  `4564874 / 2410056`
- event count A/B：
  `805566 / 425304`
- feature ledger A/B：
  `23092892 / 15310944`
- unavailable feature rows：
  `2863855`
- clusters / flows / overlap blocks：
  `39928 / 10536 / 9`
- auxiliary/core degraded grid rows：
  `4316 / 0`
- interval/right/segment/quality/epoch：
  `627406 / 177826 / 316 / 18 / 0`
- point coerced：
  `0`
- future feature observation：
  `0`
- anchor ordering：
  `0`
- cross-segment / cross-epoch：
  `0 / 0`
- synthetic rejected confirm：
  `0`

## Verification

Tests：

- Stage 4 focused：
  `46 passed in 0.09s`
- Stage 1/2/3 inherited：
  `216 passed in 133.80s`
- R0/alignment/recovery：
  `83 passed in 0.49s`
- total unique related tests：
  `345 passed`
- targeted first-round zero-regression subset：
  `3 passed`

Static/integrity：

- Ruff：`All checks passed!`
- external compileall：
  `48` bytecode files，全部写到
  `/tmp/0815T003-r2-final-pycache.Q1sEgv`
- deterministic gzip：
  `96/96` passed
- builder/admission `--help`：passed
- scoped `git diff --check`：passed

Default-Python zero-write：

- final source admission：
  `verified=true`, `source_semantic_verified=true`
- final archived admission：
  `verified=true`, `source_semantic_verified=true`
- package before/after：
  - files：`107 / 107`
  - path/bytes/raw SHA/mtime_ns/ctime_ns exact identical：`true`
  - package `.pyc/__pycache__`：`0`

Atomic publication：

- injected `_process_segment` failure observed：`true`
- output directory visible：`false`
- residual staging directories：`0`
- partial markers：`0`

Input immutability：

- rows：`394`
- bytes：`1412620737`
- before/after/current identity：
  `1c8fd0b424844be81475eb28c0e2b2486a3a06787fb3cca03920ce2ab0b63476`
- exact identical：`true`

## Code Changes

- `examples/hyperliquid/cross_exchange_trigger_aligned_episodes.py`
  - contract v3；
  - complete feature/view projection contract；
  - bounded exact whole-row stream comparator；
  - complete Family A/B source-derived feature/view replay；
  - aggregate expected/observed count/digest evidence；
  - source-semantic replay 前置；
  - `verify_package()` 返回 scoped source-semantic verification evidence。
- `examples/hyperliquid/cross_exchange_jul30_episode_v3_admission.py`
  - success 前要求 complete scope/aggregate evidence；
  - 输出 source-semantic scope 和 aggregate results。
- `examples/hyperliquid/test_cross_exchange_trigger_aligned_episodes.py`
  - 逐字段 `FEATURE_FIELDS` / `VIEW_FIELDS` exact comparison；
  - missing/extra row；
  - complete projection contract/count/digest regressions。
- `examples/hyperliquid/test_cross_exchange_jul30_episode_v3_admission.py`
  - complete evidence success；
  - missing scope/evidence fail-closed。
- 正式 package 通过 staging 完整原子重建，未手改发布包。

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
- 未修改 task/status、`task_plan.md`、`progress.md`、`findings.md`、
  QA report/mirror、accepted Stage 1/2/3 或历史输入。

## Residual Risk

- 未发现剩余 P0/P1/P2/P3 correctness defect。
- complete source-semantic admission 会重放约 `38.4M` feature rows，并
  验证所有其他 exact streams；运行成本较高，但终止稳定，且内存按
  segment 有界。
- Stage 5、Aug03/Aug04 future rows、Aug07 rows、case/model/score、
  actionability、own-order lifecycle 和 randomized EV 仍全部锁定。

## Done

- 第二轮唯一 QA 缺陷已实现关闭。
- 第一轮五项修复和 hostile trust boundary 已完成零回归验证。
- 正式 package 与两个隔离构建逐文件一致。
- 本业务线程状态仅为 `待验收`。
- 需要全新独立 QA 通过后，才可由总控把 `0815T003` 置为 `已通过`
  并解锁 Stage 5。
