# 线程回报

执行线程：
- 总控 auto-loop / 业务实现线程

任务ID：
- 0719T003

状态：
- 待验收

是否进行QA验收：
- 是

QA说明：
- 无

files：
- `.workflow/tasks/0719T003.md`
- `examples/hyperliquid/hyperliquid_tiny_live_m2_fill_window.py`
- `examples/hyperliquid/cross_exchange_t024_same_window_acceptance.py`
- `examples/hyperliquid/test_hyperliquid_tiny_live_m2_fill_attribution.py`
- `examples/hyperliquid/test_cross_exchange_t024_same_window_acceptance.py`

action：
- Producer cancel target matching 改为 all-token consistency：
  - 每个非空 oid/cloid token 分别解析；
  - 每个 token 必须唯一命中；
  - 所有 token 必须命中同一个 attempt/reference；
  - unknown、duplicate、cross-reference conflict 均 fail-closed。
- Oid-only、cloid-only 和一致的 oid+cloid 仍可正常映射。
- Acceptance 新增独立 raw-proof reconstruction：
  - 只以 `cancel_shutdown_proof.tracked_refs/cancel_results` 为原始输入；
  - 自行规范化 attempt/oid/cloid 和 deterministic reference key；
  - 自行解析 raw exchange cancel response 的 authoritative success；
  - 自行识别 generic `already canceled, or filled`；
  - 自行应用 all-token unique same-reference matching；
  - 自行计算每个 reference 的 authoritative terminal status。
- Acceptance 不导入、不调用 producer reconciliation helper。
- 独立重建结果必须同时与 fill manifest summary、cancel shutdown proof summary 完全一致。
- 缺失或非 list 的 raw tracked refs/cancel results 直接 fail-closed。
- 未修改 quote selection、strategy thresholds、live envelope、activation flags 或 actual quote behavior。
- 本任务未调用 live/private/account/order/cancel/network/remote/service。

verify：
- T026 QA producer 复现 `oid=101 + cloid=unknown` 返回 `fail_closed`。
- 正确 cloid + unknown oid 返回 `fail_closed`。
- oid/cloid 分别指向两个已知 reference 返回 `cancel_result_conflicting_target`。
- 同步保留两个 pass summary、但 raw target unrelated 时 acceptance blocked。
- 同步保留两个 pass summary、但 raw response ambiguous-only 时 acceptance blocked。
- 缺失 raw proof inputs 时 acceptance blocked。
- Valid raw proof 的 independent reconstruction 与 producer output 逐字段相等。
- 原 attempt 1 success + attempt 2 ambiguous-only、same-reference redundant generic、standalone multi-attempt 和 two-sided manager regressions继续通过。
- focused regression：`130 passed in 22.62s`。
- full `python -m pytest examples/hyperliquid -q`：`483 passed in 32.81s`。
- modified Python `py_compile`、三个 CLI `--help`、`git diff --check` 通过。

done：
- T026 QA 的 producer partial-conflicting-token P1 已修复。
- T026 QA 的 synchronized forged-summary/raw-proof contradiction P1 已由独立 reconstruction gate 修复。
- 离线实现和回归完成，等待独立 QA。

blockers：
- 独立 QA 通过前，不得创建或启动新的 live task。
- Principal Task 12 和 Task 10 single-level two-sided lifecycle gate 尚未关闭。

commit：
- `ba220c52a4e0cfa6f045cabcbc1ab034acd3f2f9`

提交信息：
- `Rebuild cancel proof from raw evidence`
