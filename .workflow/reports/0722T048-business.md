# 0722T048 Business Report

执行线程：
- 总控 auto-loop / 业务执行线程

任务ID：
- 0722T048

状态：
- 待验收

更新时间：
- 2026-07-22 09:08 Asia/Shanghai

是否进行QA验收：
- 是

QA说明：
- 无

files：
- `examples/hyperliquid/cross_exchange_t024_same_window_acceptance.py`
- `examples/hyperliquid/test_cross_exchange_t024_same_window_acceptance.py`
- `.workflow/tasks/0722T048.md`

action：
- 修复 dynamic profile acceptance 把 exact profile 错误写死为 `two-sided-manager` 的问题。
- canonical watcher grammar 现在按 expected dynamic activation 条件允许且只允许末尾的 `--enable-dynamic-spread`。
- fixed profile 默认仍拒绝 dynamic flag，保留 fail-closed regression。
- T047 post-live account proof 从 sealed run root 移到同级附加证据路径；sealed run root 恢复为 terminal manifest 的原始文件集合。

verify：
- `python -m pytest examples/hyperliquid/test_cross_exchange_t024_same_window_acceptance.py -q`：`245 passed`。
- dynamic/fixed canonical grammar direct check：通过。
- `py_compile`：通过。
- `git diff --check`：通过。
- offline T047 same-window acceptance：exit `0`；provenance `113/113`、config `76/76`、decision `43/43`、lifecycle `78/78`、economics `6/6`、optimism `6/6`。
- 远端 sealed run root 独立 checksum：`110/110` pass。
- 本任务未执行 live/private/account/network/remote/service/order/cancel。

done：
- T047 dynamic-profile acceptance verifier contract 已修复，T047 可进入独立 QA。
- 旧 fixed profile 的 canonical grammar 和 activation-off 边界保持。

blockers：
- 独立 QA 验收。

commit：
- `357efc1cabe040118c3ed880fd8de3889d2ff57f`

提交信息：
- `Repair dynamic-profile same-window acceptance contract`
