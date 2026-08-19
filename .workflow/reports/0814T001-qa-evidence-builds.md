# 0814T001 Independent QA Evidence Builds

- 执行线程：`QA evidence subthread B`
- 任务ID：`0814T001`
- 记录时间：`2026-08-15 11:45 CST`
- 工作目录：`/Users/liu/Documents/hftbacktest-0814t001-skhynix-episode-research`
- 只读源工作区：`/Users/liu/Documents/hftbacktest`
- 正式包：`/Users/liu/Documents/hftbacktest-0814t001-skhynix-episode-research/local_live_analysis/skhynix_trigger_aligned_episode_research_v1`
- 临时构建 A：`/tmp/0814T001-qa-buildA.OtBb8y/package`
- 临时构建 B：`/tmp/0814T001-qa-buildB.e0Ak3t/package`

## Scope Guard

- 未修改代码或正式包内容。
- 未更新 `.workflow/tasks/`、`task_plan.md`、`progress.md`、`findings.md` 或 QA 结论文档。
- 未 content-open Aug07 event rows；仅使用生成器允许的 metadata allowlist 和 raw stat-only 路径。

## Exact Commands

```bash
python /Users/liu/Documents/hftbacktest-0814t001-skhynix-episode-research/examples/hyperliquid/cross_exchange_trigger_aligned_episode_contract.py \
  --source-root /Users/liu/Documents/hftbacktest \
  --output-dir /tmp/0814T001-qa-buildA.OtBb8y/package

python /Users/liu/Documents/hftbacktest-0814t001-skhynix-episode-research/examples/hyperliquid/cross_exchange_trigger_aligned_episode_contract.py \
  --source-root /Users/liu/Documents/hftbacktest \
  --output-dir /tmp/0814T001-qa-buildB.e0Ak3t/package \
  --compare-to /tmp/0814T001-qa-buildA.OtBb8y/package

python /Users/liu/Documents/hftbacktest-0814t001-skhynix-episode-research/examples/hyperliquid/cross_exchange_trigger_aligned_episode_contract.py \
  --output-dir /Users/liu/Documents/hftbacktest-0814t001-skhynix-episode-research/local_live_analysis/skhynix_trigger_aligned_episode_research_v1 \
  --verify-only \
  --compare-to /tmp/0814T001-qa-buildA.OtBb8y/package

python -m pytest -q /Users/liu/Documents/hftbacktest-0814t001-skhynix-episode-research/examples/hyperliquid/test_cross_exchange_trigger_aligned_episode_contract.py

python -m compileall -q \
  /Users/liu/Documents/hftbacktest-0814t001-skhynix-episode-research/examples/hyperliquid/cross_exchange_trigger_aligned_episode_contract.py \
  /Users/liu/Documents/hftbacktest-0814t001-skhynix-episode-research/examples/hyperliquid/test_cross_exchange_trigger_aligned_episode_contract.py

python -m ruff check \
  /Users/liu/Documents/hftbacktest-0814t001-skhynix-episode-research/examples/hyperliquid/cross_exchange_trigger_aligned_episode_contract.py \
  /Users/liu/Documents/hftbacktest-0814t001-skhynix-episode-research/examples/hyperliquid/test_cross_exchange_trigger_aligned_episode_contract.py

git -C /Users/liu/Documents/hftbacktest-0814t001-skhynix-episode-research diff --check
```

## Build Results

### Build A stdout summary

```json
{
  "artifact_count": 13,
  "aug07_full_event_rows_opened": false,
  "core_package_sha256": "9c2756f966a1a05816f9e6a91d7361d76cfea10fc3c587944ca2f0c1d2eb7c96",
  "output_dir": "/private/tmp/0814T001-qa-buildA.OtBb8y/package",
  "source_inventory_unchanged": true
}
```

### Build B stdout summary

```json
{
  "artifact_count": 13,
  "aug07_full_event_rows_opened": false,
  "comparison": {
    "artifact_count": 13,
    "core_package_sha256": "9c2756f966a1a05816f9e6a91d7361d76cfea10fc3c587944ca2f0c1d2eb7c96",
    "identical": true
  },
  "core_package_sha256": "9c2756f966a1a05816f9e6a91d7361d76cfea10fc3c587944ca2f0c1d2eb7c96",
  "output_dir": "/private/tmp/0814T001-qa-buildB.e0Ak3t/package",
  "source_inventory_unchanged": true
}
```

### Formal package verify-only summary

```json
{
  "artifact_count": 13,
  "aug07_full_event_rows_opened": false,
  "comparison": {
    "artifact_count": 13,
    "core_package_sha256": "9c2756f966a1a05816f9e6a91d7361d76cfea10fc3c587944ca2f0c1d2eb7c96",
    "identical": true
  },
  "core_package_sha256": "9c2756f966a1a05816f9e6a91d7361d76cfea10fc3c587944ca2f0c1d2eb7c96",
  "output_dir": "/Users/liu/Documents/hftbacktest-0814t001-skhynix-episode-research/local_live_analysis/skhynix_trigger_aligned_episode_research_v1",
  "source_inventory_unchanged": true
}
```

## File / Byte / SHA Comparison

- formal、Build A、Build B 均为相同 `14` 个 package files：
  - `consumption_ledgers/aug07_access_ledger.json`
  - `consumption_ledgers/source_inventory_after.json`
  - `consumption_ledgers/source_inventory_before.json`
  - `data_admission/channel_inventory.csv`
  - `data_admission/data_admission.md`
  - `data_admission/hyperliquid_feed_cadence.csv`
  - `data_admission/input_manifest_bindings.csv`
  - `data_admission/session_topology.csv`
  - `data_admission/unavailable_fields.csv`
  - `data_admission/underlying_regime_coverage.csv`
  - `frozen_research_contract.json`
  - `input_inventory.csv`
  - `research_manifest.json`
  - `runtime_source/cross_exchange_trigger_aligned_episode_contract.py`

- 三者逐文件 SHA256 完全一致：

```text
40deed8ba0756e85697ef81e107a211ab2f4fabca638c1b42ae14f8314f7426d  consumption_ledgers/aug07_access_ledger.json
0a295a22dfb59e10f9895c14bd3f2bc394e75aa5e3febe0fd152e10a51417844  consumption_ledgers/source_inventory_after.json
0a295a22dfb59e10f9895c14bd3f2bc394e75aa5e3febe0fd152e10a51417844  consumption_ledgers/source_inventory_before.json
41a80972f0bf6407128bfcc89bc942a03e27aaeefc40b1f652d89dbd14eda333  data_admission/channel_inventory.csv
6c6ebc09ae550adf8af3a2737dcb8f1a566dcf0d70e94557731ea00916cd878a  data_admission/data_admission.md
2e4060036155f80e6c1cebddd81143edce3c3b534dbc7f5c129d6c3f4ed81c0a  data_admission/hyperliquid_feed_cadence.csv
c08c53f430398e8a68d6e2c8a83d32bd2a2fa889bc9a0fb5bc1c199b70fa59fc  data_admission/input_manifest_bindings.csv
b00f4eae599bba63099d3b8c629324c761ebdacd2481ac56ebd119b182016827  data_admission/session_topology.csv
8ba7c2061ac587adbe5fe79ecf5adef120869dd20894f675bc8d16f8232c03e5  data_admission/unavailable_fields.csv
ec33eeb2a1e7354d2efdbb4c08d683da2eb0dc1b6c8fc369935f8d2de06f0319  data_admission/underlying_regime_coverage.csv
8657c6a81cb541c8df1c7696f86b3b4cbfb0c2e01d0bb99a98bbecd158041a6e  frozen_research_contract.json
349fb8528437ad2beed538368173c0666f0e0be519a729a74c4b26727acfab42  input_inventory.csv
9d8bf64c1a95ea378e88fe8d23243ce2896d727661988402c4e5dfd8600d55c2  research_manifest.json
f675d8cff62432e4be7fb2560345213ac85bfd203c40be75a495ee0c7132a426  runtime_source/cross_exchange_trigger_aligned_episode_contract.py
```

- 三者全目录统计一致：

```text
/Users/liu/Documents/hftbacktest-0814t001-skhynix-episode-research/local_live_analysis/skhynix_trigger_aligned_episode_research_v1 count=14 bytes=1401387 digest=993bbc266d50e244af3bbeb4f9277e59efe437978d691d236d6bb034f7c4dd70
/tmp/0814T001-qa-buildA.OtBb8y/package count=14 bytes=1401387 digest=993bbc266d50e244af3bbeb4f9277e59efe437978d691d236d6bb034f7c4dd70
/tmp/0814T001-qa-buildB.e0Ak3t/package count=14 bytes=1401387 digest=993bbc266d50e244af3bbeb4f9277e59efe437978d691d236d6bb034f7c4dd70
```

- `research_manifest.json` 核心字段一致：
  - `core_package_sha256 = 9c2756f966a1a05816f9e6a91d7361d76cfea10fc3c587944ca2f0c1d2eb7c96`
  - `tracked_artifacts = 13`
  - `source_inventory_unchanged = true`
  - `aug07_full_event_rows_opened = false`

## Runtime-Source Binding

生成器源码与 formal、Build A、Build B 内嵌 runtime source SHA256 相同：

```text
f675d8cff62432e4be7fb2560345213ac85bfd203c40be75a495ee0c7132a426  /Users/liu/Documents/hftbacktest-0814t001-skhynix-episode-research/examples/hyperliquid/cross_exchange_trigger_aligned_episode_contract.py
f675d8cff62432e4be7fb2560345213ac85bfd203c40be75a495ee0c7132a426  /Users/liu/Documents/hftbacktest-0814t001-skhynix-episode-research/local_live_analysis/skhynix_trigger_aligned_episode_research_v1/runtime_source/cross_exchange_trigger_aligned_episode_contract.py
f675d8cff62432e4be7fb2560345213ac85bfd203c40be75a495ee0c7132a426  /tmp/0814T001-qa-buildA.OtBb8y/package/runtime_source/cross_exchange_trigger_aligned_episode_contract.py
f675d8cff62432e4be7fb2560345213ac85bfd203c40be75a495ee0c7132a426  /tmp/0814T001-qa-buildB.e0Ak3t/package/runtime_source/cross_exchange_trigger_aligned_episode_contract.py
```

## Source Inventories Before / After

formal、Build A、Build B 的 `source_inventory_before.json` 与 `source_inventory_after.json` 全部相同：

```text
file_count=890
total_bytes=2834242005
inventory_sha256=4b9c6f53ff43cefa3f606c8c8d78e403e5eda230d0044dc35e0b1ed1d40d8908
```

逐包检查结果：

```text
formal before 890 2834242005 4b9c6f53ff43cefa3f606c8c8d78e403e5eda230d0044dc35e0b1ed1d40d8908
formal after  890 2834242005 4b9c6f53ff43cefa3f606c8c8d78e403e5eda230d0044dc35e0b1ed1d40d8908
build_a before 890 2834242005 4b9c6f53ff43cefa3f606c8c8d78e403e5eda230d0044dc35e0b1ed1d40d8908
build_a after  890 2834242005 4b9c6f53ff43cefa3f606c8c8d78e403e5eda230d0044dc35e0b1ed1d40d8908
build_b before 890 2834242005 4b9c6f53ff43cefa3f606c8c8d78e403e5eda230d0044dc35e0b1ed1d40d8908
build_b after  890 2834242005 4b9c6f53ff43cefa3f606c8c8d78e403e5eda230d0044dc35e0b1ed1d40d8908
```

## Aug07 Access / Boundary Evidence

从 formal `consumption_ledgers/aug07_access_ledger.json` 读取：

```text
policy_version=aug07_exact_metadata_allowlist_v1
ledger_source=actual_successful_policy_content_reads
allowed_scope=exact_raw_and_compact_metadata_allowlists_only
event_rows_opened=false
event_row_read_count=0
content_read_count=51
unique_content_paths=13
raw_metadata_allowlist=['campaign_manifest.json']
compact_metadata_allowlist count=12
stat_only_paths count=39
forbidden_names=['basis_features.csv.gz', 'binance_hot_events.csv.gz', 'decision_labels.csv.gz', 'hyperliquid_auxiliary_events.csv.gz', 'hyperliquid_hot_events.csv.gz', 'raw.gz']
forbidden_suffixes=['raw.gz', '.csv.gz']
```

- `content_read_paths` 仅包含 `compact` metadata allowlist 与 `raw:campaign_manifest.json`。
- `stat_only_paths` 包含 Aug07 raw `raw.gz`、`raw.sha256`、runtime source、logs、snapshots 等路径，但这些路径未被 content-open。
- 未发现任何 `decision_labels.csv.gz`、`basis_features.csv.gz`、`raw.gz` 或其他 `.csv.gz` event artifact 的 content read。

## Manifest Closure

- `research_manifest.json` 记录 `artifacts` 数为 `13`，`core_package_sha256` 与 formal、Build A、Build B 一致。
- package 实际文件数为 `14`，其中第 `14` 个文件为 `runtime_source/cross_exchange_trigger_aligned_episode_contract.py`。
- `boundary` 字段保持：

```json
{
  "episode_v3_built": false,
  "trigger_density_run": false,
  "outcome_or_model_run": false,
  "new_collection": false,
  "private_order_cancel_access": false
}
```

## Atomic Publication Evidence

从生成器实现 [`examples/hyperliquid/cross_exchange_trigger_aligned_episode_contract.py`](/Users/liu/Documents/hftbacktest-0814t001-skhynix-episode-research/examples/hyperliquid/cross_exchange_trigger_aligned_episode_contract.py:2801) 读取到：

- `_single_writer_lock()` 使用 `output_dir.name + ".lock"`，通过 `os.O_CREAT | os.O_EXCL` 获取单写锁。
- `atomic_publish_directory()` 创建：
  - `.{output_dir.name}.tmp-{pid}-{uuid}`
  - `.{output_dir.name}.backup-{pid}-{uuid}`
- staging 内完整写入并 `verify_package(staging)` 成功后，才执行：

```python
os.replace(output_dir, backup)
os.replace(staging, output_dir)
```

- publish 失败时：
  - 删除 staging
  - 若 backup 存在且 output 不存在，则 `os.replace(backup, output_dir)` 回滚

这与“atomic publication / failure 不暴露部分目录”的实现要求一致。

## Zero Later-Stage Outputs

- formal、Build A、Build B 仅有上述 `14` 个 input-freeze / data-admission package files。
- 未发现 `episode_v3`、trigger density、detector parity、case retrieval、outcome/model/actionability 产物目录或文件。
- `research_manifest.json` 中后续阶段边界布尔值全部为 `false`。

## Verification Results

```text
python -m pytest -q .../test_cross_exchange_trigger_aligned_episode_contract.py
36 passed in 0.04s

python -m compileall -q ...
passed

python -m ruff check ...
All checks passed!

git -C /Users/liu/Documents/hftbacktest-0814t001-skhynix-episode-research diff --check
passed
```

## Notes

- 过程中曾临时生成 `.all_sha256.txt` 作为目录 digest 辅助文件，仅用于本地比较；已在 formal、Build A、Build B 中删除，最终 retained package surface 仍为 `14` 个正式文件。
- 本文档只记录独立 QA evidence builds 事实，不写最终 QA 结论，也不更新任务状态。
