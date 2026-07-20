执行线程：
- 总控 auto-loop / 业务执行线程

任务ID：
- 0720T021

状态：
- 待验收

是否进行QA验收：
- 是

QA说明：
- 无

files：
- `examples/hyperliquid/hyperliquid_tiny_live_real_order_executor.py`
- `examples/hyperliquid/test_hyperliquid_tiny_live_real_order_executor.py`
- `examples/hyperliquid/test_hyperliquid_maker_order_manager.py`
- `examples/hyperliquid/test_hyperliquid_tiny_live_m2_event_driven_watcher.py`
- `.workflow/tasks/0720T021.md`
- `.workflow/reports/0720T021-business.md`
- `task_plan.md`
- `progress.md`
- `findings.md`

action：
- Position parser 在 symbol filtering 前要求每个 row 有 canonical、非空、无首尾空白的 string coin identity。
- Direct 和 nested coin identity 同时存在时必须一致；missing、boolean、non-string 或 conflict 均 fail closed。
- 所有 row 的 `szi` 在 numeric conversion 前拒绝 boolean，并要求可转换且 finite。
- Valid empty list 仍返回 verified `0.0`；explicit well-formed foreign-only rows仍有效并返回 BTC `0.0`。
- One finite string/numeric BTC row保持正确，duplicate BTC rejection不变。
- Manager/finalizer 对 empty row、empty nested row、boolean BTC `szi` 和 coin conflict 的 fail-closed status/blocker/working-order 独立性已覆盖。
- 未改变 strategy、quote、order、cancel、risk cap、activation 或 endpoint 行为。

verify：
- Focused executor/manager/watcher/fill-attribution/acceptance：`451 passed in 16.60s`。
- Full `python -m pytest -q -p no:cacheprovider examples/hyperliquid`：`790 passed in 37.01s`。
- Modified modules/tests `py_compile`、acceptance CLI `--help`、`git diff --check`、implementation `git show --check`：通过。
- T016 exact replay：预期 exit `2`；decision `43/43`、lifecycle `49 pass / 12 fail`、provenance `112/112`、config `72/72`，历史 attempt 2 blocker保持。
- T016 replay boundary：offline-only，network/private/order/cancel/remote/new-live 均为 false。
- T016 输入 path+bytes aggregate hash 回放前后均为 `a5443f8f9cb777509ffde21f9caed3837a9ef58f31b25cd82acef2d6a3991398`。

done：
- Missing/ambiguous coin rows不能再被当作 foreign 静默跳过。
- Boolean coin 和 boolean `szi` 不能成为有效 position evidence。
- Valid empty、foreign-only 和 finite BTC semantics保持。
- T020 status propagation、T017-T019 行为和 T016 historical blocker保持。

blockers：
- 业务实现无已知阻塞；独立 QA 尚未完成，QA 通过前不启动新 bounded live、Task 8 或 adaptive/multi-level activation。

commit：
- `fa1095af12415d46dda80894e78a5e7f266281b4`

提交信息：
- `Harden position row evidence parsing`
