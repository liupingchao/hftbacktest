执行线程：
- 业务线程-python

任务ID：
- 0611T004

状态：
- 待验收

是否进行QA验收：
- 是

QA说明：
- 无

files：
- `.workflow/tasks/0611T004.md`
- `examples/binance_tick_mm/account_inventory_source.py`
- `examples/binance_tick_mm/test_account_inventory_source.py`
- `docs/basis_positive_account_inventory_source_artifact_skeleton.md`
- `local_live_analysis/basis_positive_account_inventory_source_artifact_skeleton_0611T004/**`
- `.workflow/reports/0611T004-business.md`
- `task_plan.md`
- `progress.md`
- `findings.md`

action：
- 将 `0611T003` tracking 状态更新为 QA `已通过`，并派发/执行 `0611T004`。
- 实现本地-only account/inventory artifact skeleton / validator。
- 实现 disk-only CSV/JSON parser、schema/enum/timestamp/snapshot/transition/conservation/reconciliation/overclaim fail-closed validation。
- 生成 synthetic local fixtures、validator summary/details、conservation policy、reconciliation policy、boundary validation 和 manifest。
- 编写设计说明和 focused pytest。

verify：
- `python examples/binance_tick_mm/account_inventory_source.py --help`
- `python examples/binance_tick_mm/account_inventory_source.py generate-artifacts --output-dir local_live_analysis/basis_positive_account_inventory_source_artifact_skeleton_0611T004`
- `python examples/binance_tick_mm/account_inventory_source.py validate --input local_live_analysis/basis_positive_account_inventory_source_artifact_skeleton_0611T004/fixtures/valid_account_observed_transition.csv`
- `python examples/binance_tick_mm/account_inventory_source.py validate --input local_live_analysis/basis_positive_account_inventory_source_artifact_skeleton_0611T004/fixtures/order_fills_alone_inventory_proof_overclaim.csv` returned expected fail-closed exit code `2`
- `python -m pytest examples/binance_tick_mm/test_account_inventory_source.py -q`
- `git diff --check`

done：
- Local account/inventory artifact skeleton / validator is ready for QA review.
- Final recommendation: `account_inventory_artifact_skeleton_ready_for_qa`.
- This means only that a local synthetic artifact skeleton / validator is ready for QA/controller review.
- It does not authorize endpoint/source collector/runner implementation, account/private/order/live data use, user stream, signing/nonce handling, inventory lifecycle proof, realized inventory/exposure proof, economics metrics, PnL proof, strategy/live/default-on/tiny-live, case-library/shadow decisions, parameter search, deployment, promotion, or execution-layer maker viability proof.

blockers：
- 无

commit：
- 待提交

提交信息：
- 待提交
