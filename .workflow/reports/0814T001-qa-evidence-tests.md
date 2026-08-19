# 0814T001 Independent QA Evidence Tests

执行线程：
- Independent QA evidence subthread A

任务ID：
- 0814T001

更新时间：
- 2026-08-15 11:42 CST

范围说明：
- 仅记录独立 QA 证据命令、结果和 P0-P3 观察。
- 不写最终 QA 结论。
- 不更新 task files。
- 未打开 Aug07 event rows；Aug07 相关 hostile 仅针对临时目录中的伪造 compact 路径和本地 validator/policy 调用。

## Focused verification commands

1. `python -m pytest examples/hyperliquid/test_cross_exchange_trigger_aligned_episode_contract.py`
   - exit: `0`
   - result:
     - `collected 36 items`
     - `36 passed in 0.04s`

2. `ruff check examples/hyperliquid/cross_exchange_trigger_aligned_episode_contract.py examples/hyperliquid/test_cross_exchange_trigger_aligned_episode_contract.py`
   - exit: `0`
   - result:
     - `All checks passed!`

3. `python -m compileall examples/hyperliquid/cross_exchange_trigger_aligned_episode_contract.py`
   - exit: `0`
   - result:
     - no output

4. `python examples/hyperliquid/cross_exchange_trigger_aligned_episode_contract.py --help`
   - exit: `0`
   - result:
     - parser rendered expected options:
       `--source-root`, `--output-dir`, `--clean-output`, `--verify-only`, `--compare-to`

5. `python examples/hyperliquid/cross_exchange_trigger_aligned_episode_contract.py --verify-only --output-dir local_live_analysis/skhynix_trigger_aligned_episode_research_v1`
   - exit: `0`
   - result:
     - `artifact_count=13`
     - `aug07_full_event_rows_opened=false`
     - `core_package_sha256=9c2756f966a1a05816f9e6a91d7361d76cfea10fc3c587944ca2f0c1d2eb7c96`
     - `source_inventory_unchanged=true`

6. `git diff --check`
   - exit: `0`
   - result:
     - no output

## Independent hostile injection command

7. `python - <<'PY' ... PY`
   - purpose:
     - import `examples/hyperliquid/cross_exchange_trigger_aligned_episode_contract.py` directly
     - build standalone cadence/topology/frozen-contract fixtures
     - inject hostile mutations for cadence fabricated/malformed/duplicate missing_fields
     - inject topology forged/rehashed/cardinality/schema drift
     - inject frozen-contract missing/extra/same-schema/combined drift
     - inject Aug07 compact forbidden filenames only inside a temporary directory
   - exit: `0`
   - exact JSON result:

```json
[
  {
    "case": "cadence_fabricated_missing_fields",
    "status": "PASS_BLOCKED",
    "exception": "AdmissionError",
    "message": "cadence partial metrics must exactly complement missing_fields"
  },
  {
    "case": "cadence_malformed_missing_fields",
    "status": "PASS_BLOCKED",
    "exception": "AdmissionError",
    "message": "cadence missing_fields is invalid"
  },
  {
    "case": "cadence_duplicate_missing_fields",
    "status": "PASS_BLOCKED",
    "exception": "AdmissionError",
    "message": "cadence missing_fields is invalid"
  },
  {
    "case": "topology_forged_fingerprint",
    "status": "PASS_BLOCKED",
    "exception": "AdmissionError",
    "message": "collection topology fingerprint drift"
  },
  {
    "case": "topology_rehashed_payload_mutation",
    "status": "PASS_BLOCKED",
    "exception": "AdmissionError",
    "message": "topology.jul30.acquisition_payload.python_executable: exact freeze drift expected='/home/admin/0729T003-venv/bin/python' got='/tmp/forged-python'"
  },
  {
    "case": "topology_duplicate_session_cardinality",
    "status": "PASS_BLOCKED",
    "exception": "AdmissionError",
    "message": "topology session cardinality drift"
  },
  {
    "case": "topology_missing_field_schema",
    "status": "PASS_BLOCKED",
    "exception": "AdmissionError",
    "message": "topology row schema drift"
  },
  {
    "case": "frozen_contract_missing_key",
    "status": "PASS_BLOCKED",
    "exception": "AdmissionError",
    "message": "frozen_contract.family_views.family_b: key set drift missing=['decision_landmark'] extra=[]"
  },
  {
    "case": "frozen_contract_extra_key",
    "status": "PASS_BLOCKED",
    "exception": "AdmissionError",
    "message": "frozen_contract.family_views.family_b: key set drift missing=[] extra=['unexpected']"
  },
  {
    "case": "frozen_contract_same_schema_drift",
    "status": "PASS_BLOCKED",
    "exception": "AdmissionError",
    "message": "frozen_contract.interval_censoring_contract.point_coercion_forbidden: exact freeze drift expected=True got=False"
  },
  {
    "case": "frozen_contract_combined_drift",
    "status": "PASS_BLOCKED",
    "exception": "AdmissionError",
    "message": "frozen_contract.landmarks.rejected_t_confirm_nullable: exact freeze drift expected=True got=False"
  },
  {
    "case": "aug07_compact_forbidden_decision_labels.csv.gz",
    "status": "PASS_BLOCKED",
    "exception": "AdmissionError",
    "message": "aug07_full_event_read_forbidden:/var/folders/jw/py86l_bj3cd8y1wwnpwtbkkr0000gn/T/tmpqouwsv7m/compact/decision_labels.csv.gz"
  },
  {
    "case": "aug07_compact_forbidden_basis_features.csv.gz",
    "status": "PASS_BLOCKED",
    "exception": "AdmissionError",
    "message": "aug07_full_event_read_forbidden:/var/folders/jw/py86l_bj3cd8y1wwnpwtbkkr0000gn/T/tmpqouwsv7m/compact/basis_features.csv.gz"
  },
  {
    "case": "aug07_compact_forbidden_raw.gz",
    "status": "PASS_BLOCKED",
    "exception": "AdmissionError",
    "message": "aug07_full_event_read_forbidden:/var/folders/jw/py86l_bj3cd8y1wwnpwtbkkr0000gn/T/tmpqouwsv7m/compact/raw.gz"
  }
]
```

## P0-P3 observations

- P0:
  - 无。

- P1:
  - 无。

- P2:
  - `compileall` and `git diff --check` both returned clean but emitted no positive detail beyond exit status; evidence is sufficient for the requested checks, though these commands are inherently thin compared with the richer pytest/verify-only outputs.

- P3:
  - `python -m compileall` produced no stdout on this path under the current interpreter, so the evidence is command-success based rather than text-rich.

## Evidence summary

- Focused test suite passed at `36/36`.
- Ruff passed.
- `compileall` passed.
- CLI `--help` passed.
- CLI `--verify-only` passed against `local_live_analysis/skhynix_trigger_aligned_episode_research_v1` with core SHA256 `9c2756f966a1a05816f9e6a91d7361d76cfea10fc3c587944ca2f0c1d2eb7c96`.
- `git diff --check` passed.
- All requested independent hostile injections blocked fail-closed in standalone Python execution.
