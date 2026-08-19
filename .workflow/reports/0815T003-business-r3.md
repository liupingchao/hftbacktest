# 业务线程返修报告

执行线程：
- 业务线程-python/research

任务ID：
- `0815T003`

标题：
- SKHYNIX-JUL30-FAMILY-A-B-EPISODE-V3-BUILD 第三轮有界返修

状态：
- 待验收

日期：
- 2026-08-16（星期日）

是否进行QA验收：
- 是

QA说明：
- 无

commit：
- 无

提交信息：
- 无

## Findings And Evidence First

### P1 Standalone Aggregate Attestation 已关闭

- Builder module 现在唯一拥有
  `_source_semantic_aggregate_contract()`，standalone CLI 不复制 projection
  名称、字段或计数常量。
- aggregate evidence contract version：
  `skhynix_jul30_source_semantic_aggregate_evidence_v1`。
- contract 精确冻结 `12` 个 projections：
  - `anchors`
  - `views_family_a`
  - `views_family_b`
  - `features_family_a`
  - `features_family_b`
  - `sparse_range_family_a`
  - `sparse_range_family_b`
  - `fixed_grid_family_a`
  - `fixed_grid_family_b`
  - `event_count_family_a`
  - `event_count_family_b`
  - `outcomes`
- 每个 projection 精确冻结：
  - ordered `fields`
  - `expected_rows`
  - `manifest_count_bindings`
- 每个 aggregate entry 的 exact key set 固定为：
  - `expected_rows`
  - `observed_rows`
  - `expected_sha256`
  - `observed_sha256`
  - `mismatch_rows`
  - `fields`
- `_validate_source_semantic_verification_evidence()` 在 standalone CLI
  输出 `source_semantic_verified=true` 前逐项要求：
  - semantic evidence exact top-level key set；
  - exact frozen scope；
  - aggregate 必须是 `dict`；
  - exact 12-projection key set，无缺失或额外项；
  - per-entry exact key set；
  - fields 内容和顺序 exact；
  - expected/observed rows 为 canonical Python `int`，拒绝
    `bool`/string，且等于 frozen count；
  - expected/observed digest 均为 lowercase canonical 64-hex SHA256，
    且两者相等；
  - mismatch rows 为 canonical `int 0`；
  - manifest `exact_counts` exact key set、canonical int 和 frozen values；
  - aggregate row counts 与 manifest `exact_counts` /
    `aggregate_output_counts` bindings 完全一致。
- Frozen contract artifact 同步发布 aggregate evidence contract、
  standalone attestation rules 和 12-projection universe。
- 外层 Episode contract version 保持
  `skhynix_jul30_episode_v3_contract_v3`，避免改变任何 research row 中
  已冻结的 `calculation_version/source_book_version`；本轮新增独立
  attestation contract version，并由 contract content SHA 绑定。

### Round 3 QA 最小伪造已关闭

QA 原始伪造结构仅保留 `features_family_a`，并设置：

```text
expected_rows=1
observed_rows=0
expected_sha256=bbbb...
observed_sha256=cccc...
mismatch_rows=999
fields=[]
```

重放结果：

| verifier | rc | failure boundary |
| --- | ---: | --- |
| current source | 2 | source-semantic aggregate projection universe drift |
| archived runtime | 2 | source-semantic aggregate projection universe drift |

两侧均不再输出 `verified=true` 或
`source_semantic_verified=true`。

## Negative Test Matrix

正向 fixture 现在构造完整 `12`-projection aggregate，不再以单个
projection 代替完整证据。

参数化 negative matrix 共 `32` 类，全部 `rc=2`：

1. aggregate 不是 dict。
2. 缺失 projection。
3. 额外 projection。
4. per-entry 不是 dict。
5. 缺失 per-entry key。
6. 额外 per-entry key。
7. fields 内容漂移。
8. fields 顺序漂移。
9. expected rows 数值漂移。
10. expected rows 为 bool。
11. expected rows 为 string。
12. observed rows 数值漂移。
13. observed rows 为 bool。
14. observed rows 为 string。
15. expected/observed digest 不一致。
16. expected digest 非法字符。
17. observed digest 非 64 位。
18. expected digest 大写。
19. observed digest 大写。
20. mismatch rows 非零。
21. mismatch rows 为 bool。
22. mismatch rows 为 string。
23. returned scope 漂移。
24. semantic evidence 额外 top-level key。
25. manifest exact count key 缺失。
26. manifest exact count 数值矛盾。
27. manifest exact count 为 bool。
28. manifest exact count 为 string。
29. manifest aggregate count binding 缺失。
30. manifest aggregate count 数值矛盾。
31. manifest aggregate count 为 bool。
32. manifest aggregate count 为 string。

另有 Round 3 QA 最小伪造专项 regression，双侧均 fail closed。

## Research Data Immutability

重建前正式包：
- files：`107`
- bytes：`1561252064`
- full inventory：
  `1f3ddd61fa0bbbf3268ae6f09b303d4876e83f456d9e6ce63956b1306707dc72`

重建后共有且仅有 `8` 个文件变化：

1. `runtime_source/cross_exchange_trigger_aligned_episodes.py`
2. `runtime_source/cross_exchange_jul30_episode_v3_admission.py`
3. `runtime_tests/test_cross_exchange_trigger_aligned_episodes.py`
4. `runtime_tests/test_cross_exchange_jul30_episode_v3_admission.py`
5. `frozen_episode_v3_contract.json`
6. `episode_v3_manifest.json`
7. `input_bindings.csv`
8. `reports/jul30_episode_v3.md`

第 8 项只因 canonical report 嵌入新的 runtime/test input-inventory SHA
而变化；其研究计数和结论文本未改变。

Research CSV/GZ raw-byte comparison：
- files：`99`
- bytes：`1560514934`
- changed files：`0`
- before/after 每个文件 bytes 和 raw SHA256 完全一致。

因此本轮没有改变 anchor、Family A/B view、feature、sparse path、
fixed-grid、event-count、outcome、quality、catalog 或 segment-summary
数据。

## Formal Package Identities

- package：
  `local_live_analysis/skhynix_trigger_aligned_episode_research_v1_stage04_jul30_episode_v3/`
- schema：
  `episode_v3_jul30_v1`
- outer contract version：
  `skhynix_jul30_episode_v3_contract_v3`
- aggregate attestation contract version：
  `skhynix_jul30_source_semantic_aggregate_evidence_v1`
- core package SHA256：
  `95dac8f94b6d8223d51b91c95cdea74963e62b9a940526f740fd104b062b9f7b`
- full inventory SHA256：
  `49cab05e39a74e6fd120088d19fcbf657739e64f8dadba33b267edb4c8d9272c`
- manifest SHA256：
  `7b165ff841d3829b88362bd7ea36e6546d415ec4b9e4b2637b4489cd9ee3b30b`
- contract SHA256：
  `ff2b80e627415b8fc88d05a1f08c7d740c612cfdf13a26915293bc15e862b867`
- artifacts / files / bytes：
  `106 / 107 / 1561285739`

Runtime/test bindings：

- builder：
  `003c73c7e924a11da64f6cf69881e942dd2875e60def61bb6068328b9e0ad3be`
- standalone admission：
  `95b2b7e459344b7411e22cea76c82b8631277dac7c21cf9abe4c3a7ea87b6b12`
- builder focused tests：
  `aa9fffdb5a14e418c146fe55967b80d2d5fc07f9035407de7680ec971b5595e8`
- admission focused tests：
  `17b00545c223cf96c3c38cc7b7af3fdc29e6838ebe2d000125e9493e84e6cba7`

## Determinism

- Formal：
  `local_live_analysis/skhynix_trigger_aligned_episode_research_v1_stage04_jul30_episode_v3/`
- isolated Build A：
  `/tmp/0815T003-r3-build-a.PuWtYN/package`
- isolated Build B：
  `/tmp/0815T003-r3-build-b.Q42o8J/package`
- 三者各自完成 8 segments、complete source-semantic admission 和原子发布。
- 三者共同：
  - files：`107`
  - bytes：`1561285739`
  - core：
    `95dac8f94b6d8223d51b91c95cdea74963e62b9a940526f740fd104b062b9f7b`
  - full：
    `49cab05e39a74e6fd120088d19fcbf657739e64f8dadba33b267edb4c8d9272c`
- Formal/A/B relative path、size、raw SHA 和 raw bytes 全量一致：
  `true`。
- residual formal staging directories：`0`。

Build performance：

| build | wall seconds | max RSS bytes |
| --- | ---: | ---: |
| Formal | 1316.45 | 981254144 |
| Build A | 1342.09 | 933773312 |
| Build B | 1345.84 | 956628992 |

## Positive Admission And Aggregate Evidence

默认 Python、无 `-B`：

| verifier | rc | wall seconds | max RSS bytes |
| --- | ---: | ---: | ---: |
| current source | 0 | 909.48 | 850149376 |
| archived runtime | 0 | 908.33 | 867975168 |

两侧共同返回：
- `verified=true`
- `source_semantic_verified=true`
- exact 12-projection scope；
- exact aggregate key set；
- 所有 projection expected/observed rows 相等；
- 所有 expected/observed SHA256 相等；
- 所有 mismatch rows 为 `0`；
- core/full、A/B/rejected counts 与正式包一致。

## Production-Size Hostile Regression

每个 attack：
- 从正式 `1.56 GB` package 创建 APFS clone；
- 修改 load-bearing artifact；
- 同步刷新 artifact record、core SHA 和 canonical manifest；
- 分别运行 current source 与 clone 内 archived verifier。

结果：

| attack | source | archived | failure boundary |
| --- | ---: | ---: | --- |
| Family A `pre_binance_bid_px` value | 2 | 2 | exact feature value |
| frozen contract scope shrink | 2 | 2 | frozen contract drift |
| manifest exact count drift | 2 | 2 | manifest exact count drift |
| manifest aggregate count drift | 2 | 2 | manifest aggregate output count drift |

- source/archive calls：`8/8` fail closed。
- `feature_value` wall：约 `71s`。
- contract/manifest attacks wall：约 `1203-1212s`。
- 前轮 `32` 类 production package attacks 的 research artifacts、
  source-semantic construction 和 data rows均未改变；本轮通过：
  - `99` research CSV/GZ raw-byte unchanged；
  - `379` focused/inherited tests；
  - representative feature coherent-rehash；
  - contract/manifest production attacks；
  - 完整 positive source/archive replay；
  形成最小充分不退化证据。

## Tests And Static Verification

Pytest：
- Stage 4 focused：`80 passed in 0.10s`
- Stage 1/2/3 inherited：`216 passed in 128.78s`
- R0/alignment/recovery：`83 passed in 0.44s`
- total unique related tests：`379 passed`

其他：
- Ruff：`All checks passed!`
- external `PYTHONPYCACHEPREFIX` compileall：
  `48` bytecode files，全部写入 `/tmp`
- scoped `git diff --check`：passed
- deterministic gzip integrity：`96/96`
- builder help：`18` lines
- source/archive admission help：`12 / 12` lines

Atomic publication failure injection：
- injected failure observed：`true`
- output visible：`false`
- staging directories：`0`
- partial markers：`0`

## Zero-Write And Input Immutability

正式 package 在 source/archive 默认 Python admission 前后：
- entries：`107 / 107`
- bytes：`1561285739 / 1561285739`
- path/type/size/raw SHA/mtime_ns/ctime_ns exact identical：`true`
- snapshot SHA256：
  `73829500831ef353f72fdf25a48ae8b83c1df7e9995bfabee77acf94b675fff5`
- package `.pyc/__pycache__`：`0`

Accepted input inventory 在 build/admission/tests/attacks 前后：
- rows：`394`
- bytes：`1412635729`
- identity：
  `1473c267c06c7c13c82298c11a92a4a3427353c9db6e37b1271e7792deab4348`
- exact identical：`true`

## Code Changes

- `examples/hyperliquid/cross_exchange_trigger_aligned_episodes.py`
  - source-owned aggregate evidence contract；
  - frozen expected manifest exact counts helper；
  - exact evidence validator；
  - source-semantic generator/count/fields 从同一 contract 派生；
  - frozen contract attestation rules。
- `examples/hyperliquid/cross_exchange_jul30_episode_v3_admission.py`
  - 成功前调用 source-owned exact evidence validator；
  - 删除 CLI 自有的不完整 aggregate 类型判断。
- `examples/hyperliquid/test_cross_exchange_trigger_aligned_episodes.py`
  - 12-projection aggregate contract coverage test。
- `examples/hyperliquid/test_cross_exchange_jul30_episode_v3_admission.py`
  - 完整 positive fixture；
  - 32 类参数化 negative matrix；
  - Round 3 QA 最小伪造 regression。
- 正式 package 通过完整 staging 原子重建，未手改发布包。

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
  QA report/mirror、accepted Stage 1/2/3 或 Jul30 immutable inputs。
- Stage 5、later-session future rows、case/model/score/actionability、
  own-order lifecycle 和 randomized EV 继续锁定。

## Residual Risk

- 未发现剩余 P0/P1/P2/P3 correctness defect。
- Complete source-semantic admission 仍是重型离线验证，单次约
  `15` 分钟；production contract/manifest attacks 因失败边界在完整
  replay 后，单次约 `20` 分钟。这是运行成本，不是 fail-open。
- 本线程不宣告任务 `已通过`。必须由第四轮全新独立 QA 验收。

## Done

- 第三轮 QA 唯一 P1 已完成有界返修。
- Research data rows 和所有 CSV/GZ artifacts 保持 raw-byte 不变。
- Formal、Build A、Build B 全量一致。
- source/archive positive admission、QA 最小伪造、32 类 negative matrix、
  production coherent-rehash、379 tests、gzip、zero-write、atomic failure
  和 input immutability 全部闭合。
- 本业务线程状态仅为 `待验收`；Stage 5 继续锁定。
