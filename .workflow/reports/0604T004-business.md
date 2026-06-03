```md
执行线程：
- 业务线程-research

任务ID：
- 0604T004

状态：
- 待验收

是否进行QA验收：
- 是

QA说明：
- 无

files：
- `.workflow/tasks/0604T004.md`
- `.workflow/reports/0604T004-business.md`
- `examples/hyperliquid/canonical_event_mode_evidence.py`
- `examples/hyperliquid/test_canonical_event_mode_evidence.py`
- `local_live_analysis/canonical_event_mode_evidence_0604T004/**`
- `progress.md`
- `findings.md`

action：
- 实现 read-only canonical event-mode evidence loader / validator。
- Loader 读取 `0604T003` aggregate 所需文件：`multi_sample_manifest.json`、`sample_quality_matrix.csv`、`feature_horizon_stability_across_samples.csv`、`effective_horizon_aliasing_by_sample.csv`、`venue_state_conditioning_across_samples.csv`。
- Validator 强制 canonical 样本必须同时满足 `decision_mode=event` 和 `canonical_status=canonical_event_mode`。
- Validator 将 `diagnostic_only_synthetic_decision_grid` 样本排除出 canonical 输出，并写入 diagnostic rejection report。
- 生成 task-scoped artifacts：
  - `local_live_analysis/canonical_event_mode_evidence_0604T004/canonical_sample_manifest.json`
  - `local_live_analysis/canonical_event_mode_evidence_0604T004/canonical_sample_quality_summary.csv`
  - `local_live_analysis/canonical_event_mode_evidence_0604T004/diagnostic_rejection_report.csv`
  - `local_live_analysis/canonical_event_mode_evidence_0604T004/canonical_evidence_validation_report.md`
  - `local_live_analysis/canonical_event_mode_evidence_0604T004/synthetic_diagnostic_validation/**`
- API/CLI：
  - Importable API: `load_canonical_event_mode_evidence(...)`
  - Importable API: `build_canonical_event_mode_evidence_artifacts(...)`
  - CLI: `python examples/hyperliquid/canonical_event_mode_evidence.py --input-dir <dir> --output-dir <dir>`

verify：
- `python examples/hyperliquid/canonical_event_mode_evidence.py --help` -> passed
- `python -m py_compile examples/hyperliquid/canonical_event_mode_evidence.py examples/hyperliquid/test_canonical_event_mode_evidence.py` -> passed
- `python -m pytest examples/hyperliquid/test_canonical_event_mode_evidence.py` -> passed, `5 passed`
- `python examples/hyperliquid/canonical_event_mode_evidence.py --input-dir local_live_analysis/event_mode_canonical_pricing_signal_0604T003 --output-dir local_live_analysis/canonical_event_mode_evidence_0604T004` -> passed, `canonical_sample_count=3 diagnostic_rejection_count=0`
- `python examples/hyperliquid/canonical_event_mode_evidence.py --input-dir local_live_analysis/event_mode_canonical_pricing_signal_0604T003/synthetic_diagnostic_comparison --output-dir local_live_analysis/canonical_event_mode_evidence_0604T004/synthetic_diagnostic_validation` -> passed, `canonical_sample_count=0 diagnostic_rejection_count=3`
- `python -m json.tool local_live_analysis/canonical_event_mode_evidence_0604T004/canonical_sample_manifest.json` -> passed
- `python -m json.tool local_live_analysis/canonical_event_mode_evidence_0604T004/synthetic_diagnostic_validation/canonical_sample_manifest.json` -> passed
- `git diff --check` -> passed

done：
- Canonical event-mode aggregate validates and returns exactly 3 canonical samples: `cross_exchange_public_sample_xemm_0603_quiet_a_event`, `cross_exchange_public_sample_xemm_0603_quiet_b_event`, `cross_exchange_public_sample_xemm_0603_quiet_c_event`。
- Synthetic diagnostic comparison parses as diagnostic-only and returns `0` canonical samples with `3` diagnostic rejections。
- Missing required file and missing required column failure paths are covered by focused tests。
- Boundary flags remain read-only: no new collection, no signal ranking, no regime selection, no case-library construction, no shadow decision generation, no private/order endpoints, no order lifecycle, no strategy implementation, no live/default-on/tiny-live, no parameter search, no schema/API changes, no promotion。

blockers：
- 无

commit：
- 3e6a3d8

提交信息：
- 0604T004 canonical event evidence loader
```
