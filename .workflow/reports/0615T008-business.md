# 0615T008 Business Report

执行线程：
- 业务线程-python

任务ID：
- 0615T008

状态：
- 待验收

是否进行QA验收：
- 是

QA说明：
- 无

files：
- `.workflow/tasks/0615T008.md`
- `.workflow/reports/0615T008-business.md`
- `examples/binance_tick_mm/small_cap_live_test_protocol.py`
- `examples/binance_tick_mm/test_small_cap_live_test_protocol.py`
- `docs/small_cap_live_test_protocol.md`
- `local_live_analysis/small_cap_live_test_protocol_0615T008/**`
- `progress.md`
- `findings.md`

prerequisite status：
- `0615T007` QA is `已通过`; final recommendation is `proof_limited_read_only_runner_ready_for_qa`.

protocol summary：
- Defined small-cap protocol for `BTCUSDT`.
- Duration cap: `10` minutes.
- Max gross notional: `25 USDT`.
- Max single order notional: `5 USDT`.
- Max position notional: `10 USDT`.
- Max loss: `2 USDT`.
- Maker-only and post-only are required.
- Default-on is forbidden.
- Explicit total-control live-window approval is required before `0615T009`.

dry-run artifact summary：
- `small_cap_live_test_protocol.csv`: `1` row.
- `risk_gate_matrix.csv`: `6` rows.
- `kill_switch_rules.csv`: `4` rows.
- `required_live_artifacts.csv`: `10` rows.
- `dry_run_acceptance.csv`: `3` rows.
- `boundary_validation.csv`: `4` rows.
- Manifest final recommendation: `small_cap_live_test_protocol_ready_for_qa`.

final recommendation：
- `small_cap_live_test_protocol_ready_for_qa`
- This means only that the small-cap live-test protocol and dry-run gate are ready for QA/controller review.
- It does not open live, authorize credentials, connect endpoints, place/cancel orders, change strategy defaults, compute PnL proof, deploy, promote, or prove maker viability.

verify：
- `/home/liushuai/workspace/hftbacktest/.conda-envs/hft-py38/bin/python examples/binance_tick_mm/small_cap_live_test_protocol.py --help` passed.
- `/home/liushuai/workspace/hftbacktest/.conda-envs/hft-py38/bin/python -m pytest examples/binance_tick_mm/test_small_cap_live_test_protocol.py -q` passed: `3 passed`.
- `/home/liushuai/workspace/hftbacktest/.conda-envs/hft-py38/bin/python examples/binance_tick_mm/small_cap_live_test_protocol.py generate-artifacts --output-dir local_live_analysis/small_cap_live_test_protocol_0615T008` passed.
- Parsed generated artifacts successfully.
- `git diff --check` passed.

blockers：
- 无 execution blocker.

commit：
- pending

提交信息：
- pending
