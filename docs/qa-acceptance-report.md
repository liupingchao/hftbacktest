# QA Acceptance Report

## 0722T049

状态：`阻塞`

T049 的 bounded fill-feedback implementation、focused/full tests、runtime source provenance、terminal checksum、estimator replay 和 fill-feedback replay 均通过。但 immutable same-window live evidence 未通过：

- attempt 1 的 cancel 只有 ambiguous generic match；
- 5 轮 direct `oid/cloid` query 和 1 次 historical query 均为 `unknown`；
- 因此缺少 reference-bound authoritative terminal proof，producer 保持 `no_fill_unproven` / `fill_reconciliation_required_no_fill_unproven`；
- exact fill profile 的 preflight 还错误记录了 `lead_source=unspecified`，应为 `binance_public_book_ticker`。

验收计数：provenance `113/0`、config `82/1`、decision `43/0`、lifecycle `66/12`、economics `6/0`、optimism `6/0`。不得把最终空订单、空仓或 tiny-live 结果解释为稳定 PnL、fill-rate、maker viability、promotion 或 multi-level 证据。T049 不得开启第二个 live window；需新建正式修复任务。

## 0722T050

状态：`已通过`

T050 完成了 fill-feedback exact profile 的最小 offline provenance 修复：
`two-sided-fill-feedback-manager` preflight 现在严格记录
`lead_source=binance_public_book_ticker`，focused orchestrator `38 passed`，
py_compile 和 diff-check 通过。T049 sealed run、原始
`lead_source=unspecified`、`110/110` checksum 以及 ambiguous
cancel/unknown terminal query blocker 均保持不变；T050 不关闭或重解释
T049 same-window evidence gate。
