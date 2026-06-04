```md
执行线程：
- 业务线程-research

任务ID：
- 0604T005

状态：
- 待验收

是否进行QA验收：
- 是

QA说明：
- 无

files：
- `.workflow/tasks/0604T005.md`
- `.workflow/reports/0604T005-business.md`
- `examples/hyperliquid/canonical_event_mode_evidence.py`
- `examples/hyperliquid/test_canonical_event_mode_evidence.py`
- `local_live_analysis/canonical_evidence_source_lock_0604T005/canonical_source_lock_manifest.json`
- `local_live_analysis/canonical_evidence_source_lock_0604T005/canonical_guard_check_report.md`
- `local_live_analysis/canonical_evidence_source_lock_0604T005/negative_guard_validation_report.csv`
- `task_plan.md`
- `progress.md`
- `findings.md`

action：
- Added reusable source-lock guard behavior on top of the `0604T004` canonical event-mode evidence loader foundation.
- Added `guard_canonical_event_mode_evidence`, `validate_canonical_source_lock_manifest`, and `build_canonical_source_lock_artifacts` for downstream worker use before signal ranking or horizon/regime diagnostics.
- Guard checks now fail clearly for diagnostic-only fixed-grid inputs used as formal evidence, zero-canonical formal evidence, missing/inconsistent source-lock metadata, and inconsistent foundation metadata.
- Wrote task-scoped source-lock artifacts under `local_live_analysis/canonical_evidence_source_lock_0604T005/`.
- Updated focused tests for accepted canonical source, rejected diagnostic source, missing source-lock metadata, zero-canonical formal-evidence failure, and downstream-worker-style guard consumption.

verify：
- `python examples/hyperliquid/canonical_event_mode_evidence.py --help`：通过。
- `python -m py_compile examples/hyperliquid/canonical_event_mode_evidence.py examples/hyperliquid/test_canonical_event_mode_evidence.py`：通过。
- `python -m pytest examples/hyperliquid/test_canonical_event_mode_evidence.py`：通过，`10 passed`.
- `python examples/hyperliquid/canonical_event_mode_evidence.py --input-dir local_live_analysis/event_mode_canonical_pricing_signal_0604T003 --output-dir local_live_analysis/canonical_evidence_source_lock_0604T005 --foundation-dir local_live_analysis/canonical_event_mode_evidence_0604T004`：通过，`canonical_sample_count=3 diagnostic_rejection_count=0`.
- Synthetic diagnostic negative validation：通过，`guard_status=diagnostic_only_validation canonical_sample_count=0 diagnostic_rejection_count=3`.
- `python -m json.tool local_live_analysis/canonical_evidence_source_lock_0604T005/canonical_source_lock_manifest.json`：通过。
- `git diff --check`：通过。

done：
- Canonical source guard accepts `0604T003` canonical event-mode aggregate with `canonical_sample_count=3`.
- Synthetic diagnostic comparison is rejected for formal evidence and diagnostic-only classified for negative validation with `canonical_sample_count=0` and `diagnostic_rejection_count=3`.
- Source-lock manifest explicitly names `0604T003` as the formal evidence source and `0604T004` as the foundation loader artifact source.
- Boundary flags remain read-only guard hardening only: no new collection, no signal ranking, no regime selection, no private/order endpoints, no order lifecycle, no strategy implementation, no live/default-on/tiny-live, no parameter search, and no promotion.

blockers：
- 无

commit：
- 提交后由最终回报给出 exact commit id

提交信息：
- 0604T005 canonical evidence source lock guard
```
