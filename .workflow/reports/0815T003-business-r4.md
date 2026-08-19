# 业务线程返修报告

执行线程：
- 业务线程-python/research

任务ID：
- `0815T003`

标题：
- SKHYNIX-JUL30-FAMILY-A-B-EPISODE-V3-BUILD 第四轮有界返修

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

## Scope

本轮只关闭第四轮独立 QA 的唯一 P1：

- `manifest["aggregate_output_counts"]` 必须拒绝未冻结 extra key；
- expected exact key universe 必须唯一从 frozen aggregate contract 的
  `projections[*].manifest_count_bindings` 派生；
- source 与 archived-runtime permanent matrix 必须覆盖
  `aggregate_count_extra`。

未修改 Episode v3 research rows、feature/view/outcome semantics，也未
进入 Stage 5。

## Implementation

### Exact Aggregate Count Contract

`_source_semantic_aggregate_contract()` 的 attestation contract version
升级为：

```text
skhynix_jul30_source_semantic_aggregate_evidence_v2
```

每个 `manifest_count_bindings` entry 现在精确冻结：

```text
section
field
expected_value
```

`aggregate_output_counts` 的 exact key policy 明确冻结为：

```text
policy=exact
derivation=
unique binding.field values from
projections[*].manifest_count_bindings
where binding.section=aggregate_output_counts
```

最终 frozen contract 中：

- aggregate-output bindings：`28`
- unique aggregate-output fields：`28`
- manifest observed fields：`28`
- expected mapping == observed mapping：`true`

没有第二份手工 key allowlist。字段 universe 和对应 frozen expected
values 只定义在 projection bindings 中。

### Fail-Closed Validator

`_manifest_count_binding_expected_values()`：

- 要求 binding 是 exact 三字段 dict；
- 要求 field 是非空 string；
- 要求 expected value 是 canonical Python `int`；
- 拒绝 duplicate binding field；
- 从指定 section 的 bindings 唯一派生 expected mapping。

`_validate_source_semantic_verification_evidence()` 在成功前要求：

- `type(aggregate_output_counts) is dict`；
- observed key set 与派生 expected key set exact equality；
- 每个 observed value 是 canonical Python `int`；
- 每个 observed value 等于 frozen binding expected value；
- missing、extra、bool、string、value contradiction 全部 fail closed。

永久参数化 negative matrix 新增：

```text
aggregate_count_extra
```

攻击 payload：

```text
manifest["aggregate_output_counts"]["unexpected"] = 0
```

source 与 archived-runtime targeted unit admission 均拒绝该 payload。

## Changed Files

Source：

- `examples/hyperliquid/cross_exchange_trigger_aligned_episodes.py`
- `examples/hyperliquid/test_cross_exchange_jul30_episode_v3_admission.py`

Deterministically rebuilt package：

- `local_live_analysis/skhynix_trigger_aligned_episode_research_v1_stage04_jul30_episode_v3/`

相对第三轮 candidate package，正式包中实际发生 raw-byte 变化的文件为：

- `runtime_source/cross_exchange_trigger_aligned_episodes.py`
- `runtime_tests/test_cross_exchange_jul30_episode_v3_admission.py`
- `frozen_episode_v3_contract.json`
- `episode_v3_manifest.json`
- `input_bindings.csv`
- `reports/jul30_episode_v3.md`

其余 `101` 个 package files raw-byte 不变；其中包含全部 `99` 个
research CSV/GZ。

本报告：

- `.workflow/reports/0815T003-business-r4.md`

未修改：

- `.workflow/reports/0815T003-round4-pre-repair-research-inventory.csv`
- task、task_plan、progress、findings
- 既有 QA reports 和 QA mirror
- standalone admission source
- builder focused test source

## Pre-Repair Durable Evidence

Inventory artifact：

```text
.workflow/reports/0815T003-round4-pre-repair-research-inventory.csv
```

Evidence：

- inventory file SHA256：
  `07521dbc2c8bfe5f41a3ab14ac5493591dad878d52053b2d74aad251b9f022df`
- research files：`99`
- research bytes：`1560514934`
- canonical inventory SHA256：
  `bb5aed2099b1a97da5b476d4b09dfe4a7bac06f0bf331f8ca864a88daf5c9232`

## Research Data Immutability

同口径：

- 包内所有 `.csv` / `.csv.gz`
- 排除 `input_bindings.csv`
- 比较 relative path、bytes、raw SHA256

Formal、Build A、Build B 均得到：

- research files：`99`
- research bytes：`1560514934`
- canonical inventory SHA256：
  `bb5aed2099b1a97da5b476d4b09dfe4a7bac06f0bf331f8ca864a88daf5c9232`

比较结果：

- Formal == pre-repair inventory：`true`
- Build A == pre-repair inventory：`true`
- Build B == pre-repair inventory：`true`
- Formal == Build A == Build B：`true`
- changed research files：`0`

因此 anchors、Family A/B views、features、sparse ranges、fixed grids、
event-count views、outcomes、quality、catalog 和 segment summaries均未
发生 row 或 raw-byte 变化。

## Formal Package Identity

Package：

```text
local_live_analysis/skhynix_trigger_aligned_episode_research_v1_stage04_jul30_episode_v3/
```

Identity：

- artifacts：`106`
- files：`107`
- bytes：`1561299482`
- core package SHA256：
  `77ee81b62458a0885521e42eea7db66042efa2e41e742cb943f24f2f11bb7cac`
- full inventory SHA256：
  `fda845f542668ffb593964dfdd2e199f37761fc4cae991d45b3a3dc2bcba983a`
- frozen contract SHA256：
  `2bc440f4db882c28ec95fccbb8f032da4d3a73214cad8763e25e57685a419d79`
- manifest raw SHA256：
  `03f73fd162bb8736f356ed60a499ed65d28af395e0e78318854f8cc462e3d8fe`

Population：

- Family A：`268522`
- Family B：`141768`
- rejected：`126754`

Runtime/test bindings：

- builder source：
  `df9201bca9fe52d86896b825e679538eb034ed93e179eec8ddba4f6f14174b98`
- standalone admission source：
  `95b2b7e459344b7411e22cea76c82b8631277dac7c21cf9abe4c3a7ea87b6b12`
- builder focused tests：
  `aa9fffdb5a14e418c146fe55967b80d2d5fc07f9035407de7680ec971b5595e8`
- admission focused tests：
  `6e4aa01220eaf9bb775ede586faac14d9843ed8ea824eada497512d3d11f74be`

Current source/test bytes 与 package archived runtime/test bytes逐项相等。

## Determinism

Build roots：

- Formal：
  `local_live_analysis/skhynix_trigger_aligned_episode_research_v1_stage04_jul30_episode_v3/`
- Build A：
  `/tmp/0815T003-r4-build-a.O4iJbL/package`
- Build B：
  `/tmp/0815T003-r4-build-b.4tQYqP/package`

三次构建都独立完成：

- 8 segment generation；
- complete source-semantic replay；
- package verification；
- staging fsync；
- atomic publication。

三方共同：

- files：`107`
- bytes：`1561299482`
- core：
  `77ee81b62458a0885521e42eea7db66042efa2e41e742cb943f24f2f11bb7cac`
- full：
  `fda845f542668ffb593964dfdd2e199f37761fc4cae991d45b3a3dc2bcba983a`
- contract：
  `2bc440f4db882c28ec95fccbb8f032da4d3a73214cad8763e25e57685a419d79`
- manifest：
  `03f73fd162bb8736f356ed60a499ed65d28af395e0e78318854f8cc462e3d8fe`

三方 relative path、bytes、raw SHA inventory exact equal：`true`。

Performance：

| build | wall seconds | max RSS bytes |
| --- | ---: | ---: |
| Formal | 1313.65 | 919207936 |
| Build A | 1327.21 | 998359040 |
| Build B | 1331.10 | 906182656 |

Residual staging directories：`0`。

Package `.pyc` / `__pycache__`：`0`。

## Tests And Verification

### Pytest

Current source：

- Stage 4 focused：
  `81 passed in 0.10s`
- Stage 1/2/3 inherited：
  `216 passed in 128.14s`
- R0/alignment/recovery inherited：
  `83 passed in 0.42s`
- total unique related tests：
  `380 passed`

Archived runtime/tests：

- Stage 4 archived focused：
  `81 passed in 0.38s`

Targeted standalone attestation：

- current source positive + `aggregate_count_extra`：
  `2 passed`
- archived runtime positive + `aggregate_count_extra`：
  `2 passed`

这些 targeted tests 使用任务允许的 monkeypatched complete positive
source-semantic evidence；三次完整 package builds 分别执行了真实 accepted
Jul30 source-semantic replay。

### Static And Integrity

- Ruff：`All checks passed!`
- external `PYTHONPYCACHEPREFIX` compileall：
  `48` bytecode files，全部写入 `/tmp`
- deterministic gzip integrity：`96/96`
- scoped `git diff --check`：passed
- pre-repair inventory file SHA unchanged：`true`

## Hard Boundary

Formal、Build A、Build B manifest 均声明并由构建路径保持：

```text
jul30_legacy_episode_rows_read=false
aug03_aug04_future_event_rows_read=false
aug07_event_rows_read=false
model_or_score_run=false
case_retrieval_run=false
actionability_run=false
network_accessed=false
private_or_order_endpoint_accessed=false
new_collection=false
own_order_fill_pnl_fields=false
```

本轮未运行 case/model/score/actionability/orders/network/live config。
Stage 5 继续锁定。

## Remaining Caveat And QA Request

- 本报告是业务线程证据，不构成 QA 接受。
- 第五轮全新独立 QA 应：
  - 直接使用 durable pre-repair inventory 比较 Formal 和 fresh rebuild；
  - 独立复现 `aggregate_count_extra` 对 current/archived admission 的拒绝；
  - 检查 28-key universe 只能从 frozen projection bindings 派生；
  - 复核 Formal/fresh core/full/contract/manifest identity；
  - 保持 Stage 5 锁定，直到 QA 状态为 `已通过`。

blockers：
- 无。
