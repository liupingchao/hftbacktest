# 业务线程执行回报

执行线程：
- 总控 auto-loop / 业务执行线程

任务ID：
- 0719T012

状态：
- 待验收

更新时间：
- 2026-07-19 23:59 CST

是否进行QA验收：
- 是

QA说明：
- 无

files：
- `.workflow/tasks/0719T012.md`
- `examples/hyperliquid/hyperliquid_tiny_live_m2_public_watcher.py`
- `examples/hyperliquid/cross_exchange_t024_same_window_acceptance.py`
- `examples/hyperliquid/test_hyperliquid_tiny_live_m2_event_driven_watcher.py`
- `examples/hyperliquid/test_cross_exchange_t024_same_window_acceptance.py`

action：
- Producer 新增 versioned `event_driven_decision_evidence_summary_v1`：
  - exact candidate evaluation、trigger、anti-drift、immediate guard、edge gate 和 order-authorized counts；
  - anti-drift、guard 和 edge reason maps；
  - 复合 immediate-guard reason 拆为四个独立 reason atoms；
  - candidate evidence rows、distinct manager-attempt identities、submitted attempts 和 cancelled attempts 分离；
  - private read、order 和 cancel endpoint 分字段记录；
  - summary 同时写入 watcher manifest、inline manifest 和独立 JSON。
- `trigger_count` 不再布尔化为 first hit；inline 路径按完整 trigger matrix 精确计数。
- `requote_attempts_completed` 只计实际到达 order endpoint 的 canonical attempts；skipped candidate rows 单独保留为 candidate evidence cardinality。
- Acceptance 升级为 v7 并独立重建 summary：
  - 不导入 producer helper；
  - 严格解析 CSV boolean、status、event sequence、attempt identity 和 cross-file joins；
  - 独立比较 top-level/nested quote attempt matrices；
  - producer summary、独立 summary JSON、inline summary 均必须与 raw reconstruction 精确一致。
- Acceptance 新增外部 `--expected-remote-run-root`：
  - preflight、runtime source、run status、run complete、window status 和 command output 必须匹配该外部 anchor；
  - remote path 必须是 canonical absolute POSIX path；
  - local physical pullback root 仅用于本地 source/checksum/current-byte 验证，不再与 remote absolute path 字符串比较。
- 新增 synchronized remote-root rewrite、noncanonical path、stale summary、private-read conflict、top/nested attempt drift、malformed boolean、anti-drift join drift 和 candidate/manager/submission cardinality 对抗。

verify：
- Focused acceptance/watcher/orchestrator/manager：
  - `240 passed in 16.36s`
- Full Hyperliquid：
  - `699 passed in 35.21s`
- Modified modules/tests `py_compile`：pass。
- Acceptance CLI `--help`：pass，并要求 `--expected-remote-run-root`。
- Watcher CLI `--help`：pass。
- `git diff --check`：pass。
- T011 byte-exact offline re-acceptance：
  - provenance `112 pass / 0 fail`；
  - config `70 pass / 1 fail`；
  - decision `16 pass / 27 fail`；
  - lifecycle `34 pass / 27 fail`；
  - independent summary validation reasons `[]`；
  - path-only failures 全部消失；
  - final recommendation 继续 blocked。
- T011 独立 row reconstruction：
  - candidate evaluations `2564`；
  - trigger rows `26`；
  - anti-drift gate `42 pass / 5 block`；
  - immediate guard `10 pass / 11 fail`；
  - edge gate `0 pass / 10 block`；
  - edge reasons `7 fair_mid_source_stale / 3 edge_below_required_buffer`；
  - 四个 immediate-guard reason atoms 各 `11`；
  - candidate attempt rows `21`；
  - distinct manager-attempt identity `1`；
  - submissions/cancels `0/0`；
  - private read true、order false、cancel false。
- 未执行 live、private、account、order、cancel、network、remote 或 service 操作。

done：
- T011 QA 的 evidence-summary、endpoint-classification 和 path-portability defects 已完成离线实现与业务验证。
- Remote canonical provenance 由外部 expected root 约束；local pullback bytes 独立严格验证。
- T011 不再因 local/remote path 字符串不同而失败。
- T011 仍因零 submissions、edge/guard blockers 和缺失 two-sided lifecycle 正确 fail-closed。
- Multi-level、dynamic spread、fill feedback、inventory skew 和 actual quote behavior activation 未改变。

blockers：
- 当前唯一流程节点是独立 QA 验收 T012。
- T012 QA 通过前不得启动新的 bounded live window。
- Principal Task 7/12 lifecycle 和 multi-level unlock 仍未完成。

commit：
- `ab8ce0e6e803fd82ded91d054371bd0a2c3f6905`

提交信息：
- `Repair live evidence summary path portability`
