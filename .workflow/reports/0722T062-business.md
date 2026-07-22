# 0722T062 Business Report

执行线程：
- 总控 / 业务执行线程

任务ID：
- 0722T062

状态：
- 待验收

更新时间：
- 2026-07-22 16:18 Asia/Shanghai

是否进行QA验收：
- 是

QA说明：
- 无

files：
- `examples/hyperliquid/cross_exchange_shared_signal_kernel.py`
- `examples/hyperliquid/test_cross_exchange_shared_signal_kernel.py`
- `examples/hyperliquid/cross_exchange_basis_regression_production_shadow.py`
- `examples/hyperliquid/test_cross_exchange_basis_regression_production_shadow.py`
- `local_live_analysis/cross_exchange_basis_regression_production_shadow_0722T062/`
- `.workflow/tasks/0722T062.md`
- `.workflow/reports/0722T062-business.md`
- `task_plan.md`
- `progress.md`
- `findings.md`

action：
- Shared kernel 新增 strict basis contract validator、canonical hash 和
  regression signal evaluator。
- `evaluate_shared_kernel()` 新增显式 optional
  `basis_regression_contract`；默认 `None`，legacy path 不变。
- Basis mode 使用 frozen contract normalization/intercept/coefficient，
  regression output 直接作为 forecast move ticks，然后继续复用现有
  forecast/reservation/two-sided post-only quote 路径。
- 新 T062 runner 冻结 T061 contract file SHA，调用 shared kernel，输出
  decision/per-window/source-age/basis/regime/counterfactual artifacts。
- T061 六项 warning 为 required set，少一项 shadow recommendation 即失败。
- 所有 official artifact 路径为 repo-relative。

verify：
- Kernel + basis shadow focused：
  `20 passed in 0.11s`。
- Signal/kernel/legacy shadow/basis acceptance/new shadow focused：
  `33 passed in 0.39s`。
- Official artifact deterministic rerun：
  全部文件 SHA-256 一致。
- Official contract/kernel/default-off/warning/no-submit `jq` gates：通过。
- Full Hyperliquid：
  `1275 passed, 2 skipped in 57.91s`。
- Conda `py_compile`、absolute-path scan 和 `git diff --check` 通过。

done：
- Official recommendation：
  `basis_regression_public_shadow_accepted_with_warnings`。
- Contract file SHA：
  `a8d372e9108dbe921aecfa44d81e33968f4a154f11cdc44f86cd6ae8925f6190`。
- Contract canonical hash：
  `675c86d43625713bec580e777d2c1e83471b26d958660af26e9689b709baacef`。
- `10,704 / 10,704` valid rows 产生 would-submit intent；无 kernel block。
- Per-window mean adjusted counterfactual edge ticks：
  - `utc16_a`: `15.8933649289`
  - `utc17_b`: `3.0836837679`
  - `utc17_c`: `2.0943661972`
- Aggregate mean adjusted counterfactual edge：
  `7.048206278` ticks。
- Max window contribution：
  `0.3351083707`。
- T061 六项 warning 全部传播。
- Counterfactual scope 明确为：
  same accepted training package mechanism shadow，不是新 OOS、live
  economics、promotion 或 default-on proof。
- 未修改 watcher/live config，未读取 credentials，未调用 private/order/
  cancel endpoint。

blockers：
- 独立 QA 验收。

commit：
- `da0a6198f8186df410f22b2eea641ce5abafe3bb`

提交信息：
- `Run basis regression production shadow`
