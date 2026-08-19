# 0814T001 Independent QA Evidence Subthread C

执行线程：
- QA evidence subthread C

任务ID：
- 0814T001

状态：
- 待验收

是否进行QA验收：
- 是

QA说明：
- 本文档只记录独立证据、命令结果和 P0-P3 观察；不写最终 QA 结论，不更新任务文件。

files：
- `examples/hyperliquid/cross_exchange_trigger_aligned_episode_contract.py`
- `examples/hyperliquid/test_cross_exchange_trigger_aligned_episode_contract.py`
- `local_live_analysis/skhynix_trigger_aligned_episode_research_v1/`
- `.workflow/reports/0814T001-qa-evidence-contract.md`

action：
- 独立阅读 workflow、任务、业务/QA报告、实现与测试。
- 在不打开 Aug07 完整 event 内容的前提下， hostile-test `Aug07AccessPolicy` 的 exact compact allowlist、unknown/event filenames、symlink escape、resolved-path escape、ledger 派生纪律、topology fingerprint 重建，以及 frozen contract semantic freeze 覆盖。
- 对已发布 package 独立执行 verify-only、ledger/topology/cadence 验证与指纹重算。

## Commands And Results

1. 读取 workflow / task / existing reports / target source

```sh
sed -n '1,220p' .workflow/workflow-kit/workflow-manual.md
sed -n '1,220p' .workflow/workflow-kit/task-dispatch-template.md
sed -n '1,220p' .workflow/workflow-kit/thread-report-template.md
sed -n '1,220p' .workflow/workflow-kit/qa-acceptance-template.md
sed -n '1,260p' .workflow/tasks/0814T001.md
sed -n '1,260p' .workflow/reports/0814T001-business-r1.md
sed -n '1,260p' .workflow/reports/0814T001-qa.md
sed -n '560,840p' examples/hyperliquid/cross_exchange_trigger_aligned_episode_contract.py
sed -n '1840,2145p' examples/hyperliquid/cross_exchange_trigger_aligned_episode_contract.py
sed -n '2460,2775p' examples/hyperliquid/cross_exchange_trigger_aligned_episode_contract.py
sed -n '1,760p' examples/hyperliquid/test_cross_exchange_trigger_aligned_episode_contract.py
```

结果：
- `Aug07AccessPolicy` 现先检查请求路径和 resolved path 的 forbidden names/suffixes，再按 exact raw/compact metadata allowlist 判定。
- `ledger()` 从 `content_reads` 派生 `event_rows_opened` / `event_row_read_count`。
- `validate_aug07_access_ledger()` 会重算 allowlist/path/event-row 事实，不接受自由文本式 ledger。
- `validate_cadence_rows()` 现在精确核对 `missing_fields` 与空值分区。
- `validate_topology_rows()` 从完整 canonical acquisition payload 重建并重算 physical/collection fingerprints。
- `validate_frozen_contract()` 现在对 `_canonical_frozen_contract()` 做递归 canonical exact equality，而不是只挑几项语义断言。

2. Focused regression / lint / compile / CLI help

```sh
python -m pytest -q examples/hyperliquid/test_cross_exchange_trigger_aligned_episode_contract.py
python -m ruff check examples/hyperliquid/cross_exchange_trigger_aligned_episode_contract.py examples/hyperliquid/test_cross_exchange_trigger_aligned_episode_contract.py
python -m compileall -q examples/hyperliquid/cross_exchange_trigger_aligned_episode_contract.py examples/hyperliquid/test_cross_exchange_trigger_aligned_episode_contract.py
python examples/hyperliquid/cross_exchange_trigger_aligned_episode_contract.py --help
```

结果：
- `pytest`: `36 passed in 0.04s`
- `ruff`: `All checks passed!`
- `compileall`: 成功，无输出
- `--help`: 成功，暴露 `--verify-only` 和 `--compare-to`

3. Published package verify-only

```sh
python examples/hyperliquid/cross_exchange_trigger_aligned_episode_contract.py --verify-only --output-dir local_live_analysis/skhynix_trigger_aligned_episode_research_v1
```

结果：
- 成功返回 JSON：
  - `artifact_count=13`
  - `aug07_full_event_rows_opened=false`
  - `core_package_sha256=9c2756f966a1a05816f9e6a91d7361d76cfea10fc3c587944ca2f0c1d2eb7c96`
  - `source_inventory_unchanged=true`

4. Hostile-test Aug07 policy, exact compact allowlist, symlink escape, resolved-path escape, and ledger anti-self-report

```sh
python - <<'PY'
import copy, importlib.util, json, sys, tempfile
from pathlib import Path
mod_path = Path('examples/hyperliquid/cross_exchange_trigger_aligned_episode_contract.py').resolve()
spec = importlib.util.spec_from_file_location('episode_contract', mod_path)
mod = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = mod
spec.loader.exec_module(mod)

out = []
with tempfile.TemporaryDirectory() as td:
    td = Path(td)
    raw = td / 'raw'; compact = td / 'compact'; outside = td / 'outside'
    raw.mkdir(); compact.mkdir(); outside.mkdir()
    (raw / 'campaign_manifest.json').write_text('{}\n', encoding='utf-8')
    allow = compact / 'evidence/final_acceptance_summary.json'
    allow.parent.mkdir(parents=True)
    allow.write_text('{}\n', encoding='utf-8')
    forbidden_target = outside / 'decision_labels.csv.gz'
    forbidden_target.write_bytes(b'x')
    allowed_target = outside / 'allowed_alias.json'
    allowed_target.write_text('{}\n', encoding='utf-8')
    (compact / 'decision_labels.csv.gz').symlink_to(forbidden_target)
    (compact / 'evidence/alias_to_outside.json').symlink_to(allowed_target)
    policy = mod.Aug07AccessPolicy(raw, compact)
    results = {}
    for rel in ['decision_labels.csv.gz', 'evidence/alias_to_outside.json']:
        try:
            policy.assert_content_read_allowed(compact / rel)
            results[rel] = 'ALLOWED'
        except Exception as exc:
            results[rel] = f'{type(exc).__name__}:{exc}'
    policy.read_json(raw / 'campaign_manifest.json')
    policy.read_json(allow)
    ledger = policy.ledger()
    results['ledger_event_rows_opened'] = ledger['event_rows_opened']
    results['ledger_event_row_read_count'] = ledger['event_row_read_count']
    results['ledger_source'] = ledger['ledger_source']
    out.append(results)

rows = [
    {
        'scope':'raw',
        'relative_path':'campaign_manifest.json',
        'bytes':2,
        'sha256':'0'*64,
        'content_class':'allowlisted_metadata',
    }
]
ledger = {
    'schema_version':'aug07_first_read_guard_ledger_v1_r1',
    'policy_version':'aug07_exact_metadata_allowlist_v1',
    'event_rows_opened':False,
    'event_row_read_count':0,
    'allowed_scope':'exact_raw_and_compact_metadata_allowlists_only',
    'raw_metadata_allowlist':sorted(mod.AUG07_RAW_METADATA_ALLOWLIST),
    'compact_metadata_allowlist':sorted(mod.AUG07_COMPACT_METADATA_ALLOWLIST),
    'content_read_count':1,
    'content_read_paths':['raw:campaign_manifest.json'],
    'content_read_records':rows,
    'stat_only_paths':[],
    'forbidden_names':sorted(mod.AUG07_FORBIDDEN_CONTENT_NAMES),
    'forbidden_suffixes':list(mod.AUG07_FORBIDDEN_CONTENT_SUFFIXES),
    'ledger_source':'self_reported_false'
}
try:
    mod.validate_aug07_access_ledger(ledger)
    tamper = 'ALLOWED'
except Exception as exc:
    tamper = f'{type(exc).__name__}:{exc}'
out.append({'tampered_ledger_source': tamper})
print(json.dumps(out, indent=2, ensure_ascii=True))
PY
```

结果：
- compact-root hostile filename `decision_labels.csv.gz` 被拒绝：
  `AdmissionError:aug07_full_event_read_forbidden:.../compact/decision_labels.csv.gz`
- compact 内 symlink 指向 policy roots 外文件被拒绝：
  `AdmissionError:aug07_content_read_outside_policy_roots:.../outside/allowed_alias.json`
- 在只读取 allowlisted metadata 的情况下，ledger 仍派生为：
  - `event_rows_opened=false`
  - `event_row_read_count=0`
  - `ledger_source=actual_successful_policy_content_reads`
- 伪造 `ledger_source='self_reported_false'` 的 ledger 被拒绝：
  `AdmissionError:Aug07 ledger ledger_source: expected 'actual_successful_policy_content_reads', got 'self_reported_false'`

5. Recompute topology fingerprints and inspect package-side ledger/cadence/topology facts

```sh
python - <<'PY'
import csv, importlib.util, json, sys
from pathlib import Path
root = Path('local_live_analysis/skhynix_trigger_aligned_episode_research_v1').resolve()
mod_path = Path('examples/hyperliquid/cross_exchange_trigger_aligned_episode_contract.py').resolve()
spec = importlib.util.spec_from_file_location('episode_contract', mod_path)
mod = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = mod
spec.loader.exec_module(mod)
manifest = json.loads((root/'research_manifest.json').read_text())
ledger = json.loads((root/'consumption_ledgers/aug07_access_ledger.json').read_text())
mod.validate_aug07_access_ledger(ledger)
with (root/'data_admission/session_topology.csv').open(newline='', encoding='utf-8') as fh:
    topo = list(csv.DictReader(fh))
mod.validate_topology_rows(topo)
with (root/'data_admission/hyperliquid_feed_cadence.csv').open(newline='', encoding='utf-8') as fh:
    cad = list(csv.DictReader(fh))
mod.validate_cadence_rows(cad)
recomputed = {}
for row in topo:
    payload = mod._topology_payload_from_row(row)
    recomputed[row['session_id']] = {
        'physical': mod.canonical_json_sha256({k: payload[k] for k in mod.PHYSICAL_TOPOLOGY}),
        'collection': mod.canonical_json_sha256(payload),
        'row_physical': row['physical_topology_fingerprint'],
        'row_collection': row['collection_topology_fingerprint'],
    }
print(json.dumps({
  'core_package_sha256': manifest['core_package_sha256'],
  'aug07_event_rows_opened': manifest['aug07_full_event_rows_opened'],
  'ledger_content_read_count': ledger['content_read_count'],
  'ledger_unique_paths': len(ledger['content_read_paths']),
  'ledger_stat_only_paths': len(ledger['stat_only_paths']),
  'topology_recomputed': recomputed,
  'cadence_rows': len(cad)
}, indent=2, ensure_ascii=True))
PY
```

结果：
- `core_package_sha256=9c2756f966a1a05816f9e6a91d7361d76cfea10fc3c587944ca2f0c1d2eb7c96`
- `aug07_event_rows_opened=false`
- `ledger_content_read_count=51`
- `ledger_unique_paths=13`
- `ledger_stat_only_paths=39`
- `cadence_rows=36`
- 四个 session 的重算指纹与 CSV 行内指纹逐一相等：
  - `jul30`: physical `1794fde7...bc8f7`, collection `c84d4bb1...d0698`
  - `aug03`: physical `1794fde7...bc8f7`, collection `cf740240...ede6e`
  - `aug04`: physical `1794fde7...bc8f7`, collection `642f48a5...6be0f`
  - `aug07`: physical `1794fde7...bc8f7`, collection `d00c643a...d7ed7`

6. Inspect semantic-freeze surface of canonical contract

```sh
python - <<'PY'
import importlib.util, json, sys
from pathlib import Path
mod_path = Path('examples/hyperliquid/cross_exchange_trigger_aligned_episode_contract.py').resolve()
spec = importlib.util.spec_from_file_location('episode_contract', mod_path)
mod = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = mod
spec.loader.exec_module(mod)
payload = mod.build_frozen_contract()
print(json.dumps({
  'top_level_keys': sorted(payload.keys()),
  'family_view_keys': sorted(payload['family_views'].keys()),
  'scoring_keys': sorted(payload['scoring_contract'].keys()),
  'aug07_contract_keys': sorted(payload['aug07_first_read_contract'].keys()),
}, indent=2, ensure_ascii=True))
PY
```

结果：
- top-level canonical freeze surface 包含：
  `aug07_first_read_contract`, `episode_horizon_contract`, `evidence_labels`,
  `family_views`, `feature_observation_contract`, `hypothesis_contract`,
  `interval_censoring_contract`, `landmarks`, `measurement_contract`,
  `outcome_contract`, `scoring_contract`, `underlying_calendar_contract`,
  以及 `schema_version` / `task_id` / `frozen_date` / `scope`
- `family_views` 覆盖 `family_a`、`family_b`、`shared_candidate_record_required`
- `scoring_contract` 覆盖 `CRPS` / `Brier` / `interval_log_loss` / baseline /
  normalized-loss / no-post-hoc-metric-selection 等完整键集
- `aug07_first_read_contract` 覆盖 allowlists、forbidden names/suffixes、
  `freeze_before_first_event_row`、`ledger_derived_from_actual_policy_reads`、
  `unknown_content_path_fails_closed`、`compact_metadata_must_not_substitute_for_omitted_event_rows`

## Observations

### P0
- 无。

### P1
- 无新 P1 发现。独立 hostile checks 未复现第一轮 QA 的四个 fail-open：
  compact event filename、resolved-path escape、ledger 自报、forged topology
  fingerprint、cadence fabricated missing values、以及 same-schema contract drift
  均已被实现或测试层 fail closed。

### P2
- `validate_aug07_access_ledger()` 能拒绝被修改的 `ledger_source`，也会重算
  `event_rows_opened` 与 `event_row_read_count`；这说明 manifest/ledger 侧
  不能仅靠自报布尔值蒙混过关。
- published package 的 `aug07_access_ledger.json` 与 `research_manifest.json`
  一致，且 verify-only 再次验证 `aug07_full_event_rows_opened=false`。
- topology fingerprints 现在不是“看起来像 SHA256”就放行；四个 session 都从
  canonical payload 重算后与 artifact 中指纹完全一致。

### P3
- frozen contract 的 exact-freeze 面从 selected semantic checks 扩大为完整
  canonical object equality，覆盖面比首轮 QA 时更彻底，也更容易在后续 drift
  时给出精确 nested path。
- 现有 focused tests 已直接包含 QA 对应 hostile families，并额外覆盖
  missing-key / extra-key / same-schema / combined mutation；对这一类研究冻结
  gate 来说，这比只证明 happy-path 更可信。

## Scope Guard

- 本次检查未修改代码或 formal package。
- 本次检查未写 task 文件、未写 final QA conclusion。
- 本次检查未打开 Aug07 完整 raw / R0 / R1 / basis event rows；仅检查代码、
  已发布 metadata/package、以及受控临时 hostile fixtures。

done：
- 已记录独立命令、结果和 P0-P3 观察到
  `.workflow/reports/0814T001-qa-evidence-contract.md`。

blockers：
- 无

commit：
- 无

提交信息：
- 无
