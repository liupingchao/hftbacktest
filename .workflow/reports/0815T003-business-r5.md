# 业务线程返修报告

执行线程：
- 业务线程-python/research

任务ID：
- `0815T003`

标题：
- SKHYNIX-JUL30-FAMILY-A-B-EPISODE-V3-BUILD 第五轮有界返修

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

本轮只关闭第五轮独立 QA 的唯一 P1：

- package tree universe 不得忽略 dangling symlink；
- package root 必须是真实 directory，不得是 symlink；
- descendant 只能是真实 regular file 或 directory；
- symlink、FIFO、socket、device 和其他 special entry 必须在 manifest
  读取、artifact trust 或 full inventory identity 计算前 fail closed；
- `_directory_inventory()`、`_artifact_records()` 和
  `_verify_artifact_closure()` 必须共享同一个 exact `lstat` universe。

本轮未修改任何 Episode v3 research rows、feature/view/outcome semantics，
也未进入 Stage 5。

## Implementation

### Source-Owned Exact Tree Contract

新增 source-owned frozen helper contract：

```text
root_type=real_directory
classification=lstat with stat.S_ISREG/stat.S_ISDIR
descendant_allowed_types=[regular_file,directory]
forbidden_types=[
  symlink,
  fifo,
  socket,
  character_device,
  block_device,
  other_special
]
symlink_target_following=false
closure_before_manifest_or_identity=true
```

实现由以下单一 helper family 承担：

- `ExactTreeEntry`
- `_tree_entry_type_contract()`
- `_lstat_entry_type()`
- `_exact_tree_entries()`

`_exact_tree_entries()`：

- 对 package root 使用 `lstat()`；
- root 不是实际 directory 时立即拒绝；
- 对每个 descendant 使用 `lstat()` 分类；
- 只接受 `stat.S_ISREG` 和 `stat.S_ISDIR`；
- 不调用跟随 target 的 `is_file()` / `is_dir()` 决定 entry 类型；
- 任一 symlink、FIFO、socket、device 或其他 special entry 立即拒绝；
- 返回一个排序后的 exact descendant universe。

正式 package 的 frozen contract artifact 已包含
`tree_entry_type_contract`，因此 entry-type policy 由 source、contract 和
package identity 共同绑定。

### Shared Load-Bearing Universe

以下 identity-bearing 路径现在全部使用 `_exact_tree_entries()`：

- `_directory_inventory()`
- `_artifact_records()`
- `_verify_artifact_closure()`
- `_fsync_tree()`

具体保证：

- `_directory_inventory()` 先完成 exact type closure，再只对已经确认的
  regular files 读取 bytes 和计算 SHA256；
- `_artifact_records()` 从同一个 exact universe 构造 file/directory
  sets，并同时执行 exact artifact allowlist；
- `_verify_artifact_closure()` 从同一个 exact universe 构造
  files/directories，再与 frozen allowlists exact 比较；
- package root 不再通过 `.resolve()` 提前消除 symlink 身份；
- build、verify 和 compare 路径均保留 root 自身的 `lstat` 语义。

正式 package：

- root：`1` 个真实 directory；
- descendants：`128` 个；
- descendant regular files：`107`；
- descendant directories：`21`；
- 包含 root 时 directories：`22`；
- symlink/special entries：`0`。

## Permanent Regressions

`test_cross_exchange_trigger_aligned_episodes.py` 新增统一
`TREE_BOUNDARIES`：

```text
_directory_inventory
_artifact_records
_verify_artifact_closure
```

每个 load-bearing boundary 都覆盖：

- dangling symlink；
- symlink -> existing regular file；
- symlink -> directory；
- FIFO；
- package root symlink；
- 正常 regular-file/directory tree。

新增测试数量：

- descendant attacks：`4 x 3 = 12`
- root symlink：`1 x 3 = 3`
- valid tree/contract：`1`
- total：`16`

Stage 4 focused suite 因此从 `81` 增至 `97` tests。正式 package archived
runtime/tests 已确定性重建，并运行同一组 `97` tests。

## Production Attack Matrix

使用正式 package 的全新 APFS clone，逐项增加单个攻击 entry，再分别运行
current-source 和 archived-runtime standalone admission CLI。

| attack | current | archived | frozen failure |
| --- | ---: | ---: | --- |
| dangling symlink | rc=2 | rc=2 | `type=symlink` |
| symlink -> manifest regular file | rc=2 | rc=2 | `type=symlink` |
| symlink -> `anchors/` directory | rc=2 | rc=2 | `type=symlink` |
| FIFO | rc=2 | rc=2 | `type=fifo` |
| package root symlink | rc=2 | rc=2 | root is not a real directory |

结果：

- current：`5/5` fail closed；
- archived：`5/5` fail closed；
- total：`10/10` fail closed；
- fail-open：`0`；
- 攻击均在完整 source-semantic replay 前被 exact type closure 拒绝。

## Aggregate Contract Non-Regression

第四轮已接受的 aggregate exact-key 修复保持不变：

- contract version：
  `skhynix_jul30_source_semantic_aggregate_evidence_v2`
- projections：`12`
- aggregate-output bindings：`28`
- unique aggregate-output fields：`28`
- derived mapping canonical SHA256：
  `00a23f5e1105978e27bf70796a39276fc07a50bf468c2ce0a51bd57157bdfc2d`

current/archive `97`-test focused suites保留全部既有 admission negative
matrix，包括 `aggregate_count_extra`，没有回退。

## Changed Files

Source：

- `examples/hyperliquid/cross_exchange_trigger_aligned_episodes.py`
- `examples/hyperliquid/test_cross_exchange_trigger_aligned_episodes.py`

Deterministically rebuilt package：

- `local_live_analysis/skhynix_trigger_aligned_episode_research_v1_stage04_jul30_episode_v3/`

相对第四轮正式 package，本轮 contract/runtime 变化传播到：

- `runtime_source/cross_exchange_trigger_aligned_episodes.py`
- `runtime_tests/test_cross_exchange_trigger_aligned_episodes.py`
- `frozen_episode_v3_contract.json`
- `episode_v3_manifest.json`
- `input_bindings.csv`
- `reports/jul30_episode_v3.md`

本轮未修改：

- `examples/hyperliquid/cross_exchange_jul30_episode_v3_admission.py`
- `examples/hyperliquid/test_cross_exchange_jul30_episode_v3_admission.py`
- task、task_plan、progress、findings
- 任何既有 QA report 或 QA mirror
- pre-repair durable inventory
- accepted inputs/dependencies

本报告：

- `.workflow/reports/0815T003-business-r5.md`

## Runtime And Test Bindings

Current/archive raw SHA exact：

- builder source：
  `e9ead424e44f4837d55011944b2f5499be31b09bd27b630ec02155711aa7b25a`
- builder focused tests：
  `e87925f5786fe2f866833683af54b3a5f75b6ecaf0ffa0058479a78014444f84`
- standalone admission source：
  `95b2b7e459344b7411e22cea76c82b8631277dac7c21cf9abe4c3a7ea87b6b12`
- admission focused tests：
  `6e4aa01220eaf9bb775ede586faac14d9843ed8ea824eada497512d3d11f74be`

## Deterministic Rebuild

Build roots：

- Formal：
  `local_live_analysis/skhynix_trigger_aligned_episode_research_v1_stage04_jul30_episode_v3/`
- Build A：
  `/tmp/0815T003-r5-build-a.49R5b8/package`
- Build B：
  `/tmp/0815T003-r5-build-b.7bRnQo/package`

Build command shape：

```text
/usr/bin/time -l python -B
examples/hyperliquid/cross_exchange_trigger_aligned_episodes.py
--output-dir <Formal-or-isolated-package>
```

三次构建均完成：

- 8 segment generation；
- complete source-semantic replay；
- package verification；
- staging fsync；
- atomic publication。

三方共同 identity：

- artifacts：`106`
- files：`107`
- bytes：`1561307420`
- Family A：`268522`
- Family B：`141768`
- rejected：`126754`
- core package SHA256：
  `78be6559c7ac5aaec4d5411b042f7ddcce522473f60f987237bcaa9d2dd5e157`
- full inventory SHA256：
  `669fb7d12f25cfa7828aec0fb1546398b1def754952de2290cd19784a477a433`
- frozen contract SHA256：
  `b80b9bae2d6cf18cb6e7be4f133467138527fec20eef7054ab38b7f8c70cefde`
- manifest raw SHA256：
  `2c802336c1446eaf50eaf5ef546b115046abe3a1b69fdd1fcaea31d22a8e64a6`

三方 relative path、bytes、raw SHA inventory exact equal：`true`。

Performance：

| build | wall seconds | max RSS bytes |
| --- | ---: | ---: |
| Formal | 1328.16 | 970031104 |
| Build A | 1322.52 | 1022820352 |
| Build B | 1323.89 | 887586816 |

Residual staging directories：`0`。

正式 package `.pyc` / `__pycache__`：`0`。

后台 build/admission sessions：`0`。

## Research Data Immutability

Durable pre-repair inventory：

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

同口径：

- package 内所有 `.csv` / `.csv.gz`
- 排除 `input_bindings.csv`
- 比较 relative path、bytes、raw SHA256

结果：

- Formal == pre-repair inventory：`true`
- Build A == pre-repair inventory：`true`
- Build B == pre-repair inventory：`true`
- Formal == Build A == Build B：`true`
- changed research files：`0`

因此 99 个 research CSV/GZ、Episode rows、features、views 和 outcomes
均保持 raw-byte 不变。

QA evidence immutability：

- Round5 QA report SHA256：
  `2978c9dd825d0b740a1416063b74f34695ee1ae379382eab61ea5f0b64b2600f`
- QA mirror SHA256：
  `2978c9dd825d0b740a1416063b74f34695ee1ae379382eab61ea5f0b64b2600f`
- 两者保持不变：`true`

## Tests And Verification

### Pytest

Current Stage 4：

```text
python -m pytest -q
examples/hyperliquid/test_cross_exchange_trigger_aligned_episodes.py
examples/hyperliquid/test_cross_exchange_jul30_episode_v3_admission.py
```

结果：
- `97 passed in 0.11s`

Archived Stage 4：

```text
PYTHONPATH=<Formal>/runtime_source python -m pytest -q
<Formal>/runtime_tests/test_cross_exchange_trigger_aligned_episodes.py
<Formal>/runtime_tests/test_cross_exchange_jul30_episode_v3_admission.py
```

结果：
- `97 passed in 0.12s`

Current Stage 1/2/3 inherited：

```text
python -m pytest -q
examples/hyperliquid/test_cross_exchange_trigger_aligned_episode_contract.py
examples/hyperliquid/test_cross_exchange_trigger_density_core.py
examples/hyperliquid/test_cross_exchange_candidate_episode_merging.py
examples/hyperliquid/test_cross_exchange_trigger_density_inputs.py
examples/hyperliquid/test_cross_exchange_trigger_density_admission.py
examples/hyperliquid/test_cross_exchange_liquidity_response_trigger.py
examples/hyperliquid/test_cross_exchange_trigger_parity_admission.py
```

结果：
- `216 passed in 127.72s`

R0/alignment/recovery inherited：

```text
python -m pytest -q
examples/hyperliquid/test_cross_exchange_l2_timeline.py
examples/hyperliquid/test_cross_exchange_alignment_acceptance.py
examples/hyperliquid/test_cross_exchange_research_dataset.py
examples/hyperliquid/test_cross_exchange_liquidity_response_episodes.py
examples/hyperliquid/test_cross_exchange_liquidity_response_case_hierarchy.py
```

结果：
- `83 passed in 0.48s`

Current unique related tests：

- `97 + 216 + 83 = 396 passed`

### True Positive Admission

Current：

```text
/usr/bin/time -l python -B
examples/hyperliquid/cross_exchange_jul30_episode_v3_admission.py
--verify-only --package-dir <Formal>
```

结果：

- rc：`0`
- `verified=true`
- `source_semantic_verified=true`
- source-semantic mismatches：`0`
- wall：`918.21s`
- max RSS：`844972032 bytes`

Archived：

```text
PYTHONPATH=<Formal>/runtime_source /usr/bin/time -l python -B
<Formal>/runtime_source/cross_exchange_jul30_episode_v3_admission.py
--verify-only --package-dir <Formal>
```

结果：

- rc：`0`
- `verified=true`
- `source_semantic_verified=true`
- source-semantic mismatches：`0`
- wall：`918.86s`
- max RSS：`816267264 bytes`

### Static And Integrity

- Ruff：
  `All checks passed!`
- external `PYTHONPYCACHEPREFIX` compileall：
  `52` bytecode files，全部位于自动清理的 `/tmp`；
- deterministic gzip integrity：
  `96/96`；
- scoped `git diff --check`：
  passed；
- final package exact inventory：
  `107 files / 1561307420 bytes`；
- final Formal/A/B exact equality：
  `true`。

## Hard Boundary

Formal、Build A、Build B 均保持：

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

本轮未解析 legacy
`skhynix_liquidity_response_0730T017/episodes/*.csv.gz`，未读取
Aug03/Aug04 future rows 或 Aug07 event rows，未运行
case/model/score/actionability/orders/network/live config。

Stage 5 继续锁定。

## Remaining Caveat And QA Request

- 本报告是业务线程证据，不构成 QA 接受。
- aggregate contract v2、28-key exact mapping 和既有 negative matrix
  已保持通过。
- 请派发第六轮全新独立 QA，重点独立复现：
  - dangling/file/directory symlink；
  - FIFO 或其他 special entry；
  - package root symlink；
  - current/archive exact tree closure；
  - Formal/fresh identity；
  - durable 99-file research inventory equality。

files：
- `examples/hyperliquid/cross_exchange_trigger_aligned_episodes.py`
- `examples/hyperliquid/test_cross_exchange_trigger_aligned_episodes.py`
- `local_live_analysis/skhynix_trigger_aligned_episode_research_v1_stage04_jul30_episode_v3/`
- `.workflow/reports/0815T003-business-r5.md`

action：
- 实现并冻结统一 exact `lstat` tree-entry contract，重建正式 package，
  完成 Formal/A/B、current/archive admission 和攻击矩阵。

verify：
- current/archived Stage 4、Stage 1/2/3 inherited、
  R0/alignment/recovery、Ruff、compileall、96 gzip、三方 identity、
  99-file durable inventory 和 10-call production attack matrix。

done：
- 第五轮唯一 P1 已完成有界返修，业务状态为 `待验收`。

blockers：
- 无。
