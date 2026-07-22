# 0722T063 Business Report

执行线程：
- 总控 / 业务执行线程

任务ID：
- 0722T063

状态：
- 待验收

更新时间：
- 2026-07-22 16:40 Asia/Shanghai

是否进行QA验收：
- 是

QA说明：
- 无

files：
- `examples/hyperliquid/cross_exchange_shared_signal_kernel.py`
- `examples/hyperliquid/test_cross_exchange_shared_signal_kernel.py`
- `examples/hyperliquid/cross_exchange_basis_regression_production_shadow.py`
- `examples/hyperliquid/test_cross_exchange_basis_regression_production_shadow.py`
- `local_live_analysis/cross_exchange_basis_regression_production_shadow_0722T062/boundary_manifest.json`
- `.workflow/tasks/0722T063.md`
- workflow tracking files

action：
- Basis contract validator 现在要求 exact top-level field set、exact task/source
  identity、normalization label、effective-horizon condition、basis definition、
  formula、caveat、training row count/sample IDs 和 exact nested field sets。
- Normalization 每个 feature 的 `source_row_count` 必须等于 frozen
  `10704`；raw/derived coefficient schema 和 intercept 均 fail closed。
- Basis-enabled shared-kernel call 新增显式
  `basis_regression_expected_contract_hash`；缺失或不匹配直接 block。
- Official runner 同时冻结 contract file SHA 和 canonical hash
  `675c86d43625713bec580e777d2c1e83471b26d958660af26e9689b709baacef`。
- Legacy/default-off path 恢复读取 legacy contract 的 `side_mapping`；
  invalid mapping 继续抛 `unsupported_side_mapping`，missing mapping 继续
  抛 `KeyError`，与 T062 父提交行为一致。
- Boundary 顶层新增 `shared_kernel_changed_in_task=true` 和明确 change
  scope；T061 boundary 改名为 task-scoped
  `source_t061_boundary_snapshot`。

verify：
- Focused strict-contract/kernel/shadow：
  `28 passed in 0.12s`。
- Signal/kernel/legacy shadow/basis acceptance/new shadow：
  `41 passed in 0.41s`。
- Full Hyperliquid：
  `1283 passed, 2 skipped in 59.02s`。
- Conda `py_compile`、absolute-path scan 和 `git diff --check`：通过。
- Official T062 runner exit `0`。
- Official numerical artifacts 相对 T062 implementation commit：
  decision rows、per-window summary 和 warning propagation 均无 diff。
- Recommendation 仍为
  `basis_regression_public_shadow_accepted_with_warnings`；
  `10704` would-submit、mean adjusted edge `7.048206278`、max contribution
  `0.3351083707`、六项 warning 全部不变。
- Official artifact 仅 `boundary_manifest.json` 发生预期语义修复；无本机
  绝对路径。

done：
- T062 三项 P2 均已有 fail-closed regression coverage。
- Forecast arithmetic、valid legacy contract behavior、no-submit boundary 和
  official shadow numerical evidence保持不变。
- 实现已准备进入独立 QA。

blockers：
- 独立 QA 验收。

commit：
- 待提交

提交信息：
- `Repair T062 basis contract truthfulness`

