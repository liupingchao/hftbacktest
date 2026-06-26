```md
执行线程：
- 业务线程-research

任务ID：
- 0625T002

状态：
- 待验收

是否进行QA验收：
- 是

QA说明：
- 无

files：
- `.workflow/tasks/0625T002.md`
- `.workflow/reports/0625T002-repair-business.md`
- `examples/hyperliquid/cross_exchange_sample_expansion.py`
- `examples/hyperliquid/test_cross_exchange_sample_expansion.py`
- `local_live_analysis/cross_exchange_mvp_sample_expansion_0625T002/**`
- `docs/cross_exchange_maker_mvp_plan.md`
- `task_plan.md`
- `progress.md`
- `findings.md`

action：
- Repaired T002's sample expansion gate so effective horizon coverage is not merely reported but is fail-closed for `1000ms` signal acceptance.
- Kept the existing public raw/provenance/alignment/join evidence unchanged.
- Added `effective_horizon_validity_matrix.csv`.
- Split row semantics in `symmetric_edge_context_coverage.csv`:
  - `has_future_label`
  - `context_fields_complete`
  - `near_target_1000ms`
  - `effective_horizon_valid`
  - `valid_for_1000ms_signal_acceptance`
  - `effective_horizon_bucket`
- Defined near-target `1000ms` validity as `1000ms <= effective_future_age_ms <= 1250ms`.
- Preserved `complete_context` as field-completeness only, not signal-acceptance validity.
- Changed repaired recommendation from `sample_contract_ready_for_signal_acceptance` to `needs_more_public_samples`.
- Changed `t003_creation_unlocked` from `true` to `false`.
- Updated task/status/planning docs to mark the prior QA result as superseded by this repair and to block T003 creation from the current package.

evidence：
- Raw/provenance/sample validity remains intact:
  - three accepted public windows
  - overlap `1799.999859s`, `1800.036434s`, `1800.059208s`
  - complete symmetric context rows `668/666/665`, aggregate `1999`
  - future/missing/stale Binance join counts remain `0/0/0`
- Effective horizon validity gate now fails closed:
  - near-target bounds: `1000ms` to `1250ms`
  - near-target label counts in `effective_horizon_validity_matrix.csv`: `3/1/1`
  - valid complete `1000ms` signal-acceptance rows: `2/0/1`, aggregate `3`
  - required floor remains `20/window` and `100 aggregate`
  - gate reason: `1000ms_near_target_label_coverage_insufficient`
- Final repaired recommendation:
  - `needs_more_public_samples`
  - `t003_creation_unlocked=false`
- This repair directly addresses the critique that T002 recorded effective age but failed to turn it into a data contract and acceptance gate.

verify：
- `python -m pytest examples/hyperliquid/test_cross_exchange_sample_expansion.py -q` -> `2 passed`
- `python -m pytest examples/hyperliquid/test_synchronized_public_collection.py examples/hyperliquid/test_cross_exchange_lead_lag_join.py examples/hyperliquid/test_cross_exchange_lead_lag_analysis.py examples/hyperliquid/test_binance_led_pricing_signal_runner.py examples/hyperliquid/test_cross_exchange_sample_expansion.py -q -k 'not test_manifest_quality_and_artifacts_from_accepted_sample'` -> `18 passed, 1 deselected`
- Same full command without deselect -> `18 passed, 1 failed`; the failure remains the pre-existing local-environment gap where `cross_exchange_public_sample_0602T001` is absent.
- `python -m py_compile examples/hyperliquid/cross_exchange_sample_expansion.py examples/hyperliquid/test_cross_exchange_sample_expansion.py` -> passed
- `python examples/hyperliquid/cross_exchange_sample_expansion.py --help` -> passed
- All T002 JSON manifests parse with `python -m json.tool` -> passed
- Six copied raw SHA256 checks -> passed
- Repair contract validation script -> passed
- Deterministic rerun to `/tmp/0625T002-repair-repro.pgrqmE` -> passed after normalizing generated time/output paths
- `git diff --check` -> passed

done：
- T002 now fail-closes on insufficient near-target `1000ms` effective horizon coverage.
- Current T002 package must not be used to create T003.
- More public samples or a redesigned horizon contract are required before signal acceptance.
- No strategy/live behavior, watcher, private/order endpoint, credentials, side mapping, signal threshold, canary, or promotion behavior changed.

blockers：
- No repair implementation blocker.
- Forward blocker: near-target `1000ms` label coverage is insufficient in the current sample package.

commit：
- fe011b5

提交信息：
- 0625 repair sample expansion horizon gate
```
