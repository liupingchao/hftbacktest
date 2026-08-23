# 0822T002 Business Report

执行线程：
- 业务线程-python/research

任务ID：
- 0822T002

状态：
- 执行中

更新时间：
- 2026-08-23（星期日）

是否进行QA验收：
- 否

QA说明：
- 当前任务结果暂不进入QA验收，待总控确认后再决定是否派发QA验收。

files：
- `.workflow/tasks/0822T002.md`
- `.workflow/runners/0822T002_run_c6in_latency.sh`
- `examples/hyperliquid/test_skhynix_c6in_latency_v2.py`
- `.workflow/reports/0822T002-c6in-gate2-700bbec038e6/`
- `.workflow/reports/0822T002-c6in-gate2-0640c1502527/`
- `.workflow/reports/0822T002-c6in-account-probe-46ac34e4/`
- `.workflow/reports/0822T002-c6in-inspect-blocker-98bc8ded/`
- `.workflow/reports/0822T002-c6in-active-blocker-121d1050/`
- `.workflow/reports/0822T002-c6in-final-reconciliation-blocker-92534a32/`
- `.workflow/reports/0822T002-business.md`
- `task_plan.md`
- `progress.md`
- `findings.md`

action：
- 在 c6in 的 isolated `/tmp` runtime 上执行 source commit
  `0640c1502527a369e82987e81cefbeda8101f3dd`。
- 保留 source commit `700bbec038e6224cc3106af64baa5e5b773c1e62`
  的早期 notional-only subgate evidence；该轮未读取 credential 或调用
  private/order/cancel endpoint。
- 完成 Gate 0、Gate 1、current/frozen hostile preflight、notional
  subgate、future-window freeze 和完整 900 秒 public quote-safety
  collection。
- 在 public safety 通过后读取生产 credential source，并执行仅私有读取的
  open-orders、position 和 margin baseline。
- 账户 margin Gate fail closed 后停止，没有进入 post-only submit、
  cancel、flatten、L1、formal package 或 archive。
- 生成 redacted blocker receipt，拉回全部证据并逐文件比对远端 SHA256。
- 阻塞后接入 `/home/admin/trading/inspect --json`：runner 以 detached
  clean task repo、pinned task venv 和
  `/home/admin/trading/credentials.env` 为显式 override，在
  Gate 0/Gate 1/hostile/notional/schedule freeze 之后、full Gate 2
  之前验证 runtime/credential/interface 边界。
- `2026-08-23` 按用户提示检查统一账户历史，确认 production
  `HL_WALLET` 是 approved `hp1` agent。旧 runner 把 agent 当成 account，
  因此产生空历史和零 margin 的错误解释。
- 修复 account binding：从 configured/signer agent 通过
  `Info.user_role` 派生 unified master，使用 `extra_agents` 验证批准与
  有效期，使用 master 做 Info 查询和
  `Exchange(account_address=master)`，并把统一账户 spot USDC 纳入
  collateral Gate。
- 在 c6in exact commit `46ac34e4` 的干净 checkout 上执行只读 recovery
  probe；没有下单、撤单或成交。
- 从 source commit `98bc8ded` 启动首次正式恢复 runner。Gate 0、
  Gate 1、hostile、notional 和 schedule freeze 通过，随后 inspect
  因 venv Python symlink 被解析到系统解释器而 fail closed。
- 将任务 venv 改为 `python3 -m venv --copies`，并增加 inspect Python
  exact-path/non-symlink 断言；c6in inspect-only probe 通过。
- 从 source commit `121d1050` 完成 fresh full Gate 2 并进入首个 active
  window。attempt 1 的 post-only SKHX buy 被 accepted，无 fill、无仓位，
  但 exact `orderStatus=open` 在 5 秒内未可见，runner 在 attempt 2 前
  fail closed。
- 使用同一 deterministic cloid 完成 emergency cancel，最终
  open orders=`0`、SKHX position=`0`，未调用 flatten。
- 修复 resting admission：exact `orderStatus` 不可见时，仅接受
  target-DEX open orders 中 oid+cloid 双匹配；无法确认时自动执行
  cancel-by-cloid rescue 和最终 reconciliation。
- 从 source commit `92534a32` 再次完成 fresh full Gate 2。attempt 1-8
  全部 primary eligible；attempt 9 已 exact cancel confirmed，但一次
  immediate open-orders snapshot 仍显示 canceled order，因而 safety
  stop。
- collection final reconciliation 随后确认 open orders=`0`、
  SKHX position=`0`。修复 final safety proof 为 exact terminal 后
  bounded 5 秒 / 50ms polling。

verify：
- Gate 0：
  `15 surfaces / 15 negative mutations / 7 exit criteria`，verified；
  Surface Matrix SHA256：
  `6bff9a34963b1ad68d1f9bbea3d1e43836ffab3e22e6350b381a9a816f674510`。
- Gate 1：`217 passed in 8.67s`。
- Hostile preflight：`35` cases、current/frozen `70` executions、
  fail-open `0`；receipt SHA256：
  `0359fda8cf5c64ed113704060240d7588296db432d856aaa7726f0f42420ca74`。
- Notional subgate：tick `0.1`、lot `0.001`、minimum valid size
  `0.009 SKHX`、minimum executable notional about `11.2 USDC`，低于
  `15 USDC` per-order cap 和 `30 USDC` aggregate cap。
- Public quote safety：`900.004857329s`、`1658` samples、`1657` valid
  pairs、nearest-rank p99 `0.8033096356983497 bps`、minimum safe distance
  `1.6066192713966994 bps`、10 ticks
  `8.030515960650472 bps`，status `pass`。
- Credential source：owner match、mode `0600`、required keys present。
- Private account baseline：open orders `0`、SKHX position zero、available
  margin positive `false`；stable error
  `LATENCY_AUTHORIZATION_MISMATCH` at `account_baseline`。
- Safety boundary：
  `credential_file_read=true`、`private_endpoint_called=true`、
  `order_endpoint_called=false`、`cancel_endpoint_called=false`、
  `fill_observed=false`、H0-B access `false`、H0-A mutation `false`。
- Evidence pull：远端/本地 `21` 文件路径和逐文件 SHA256 完全一致；
  credential/address/private-response 泄露扫描通过。
- Full-run remote inventory SHA256：
  `3af2cbfe07f255f1ed2bcf41dfeb6655912ffbdecf2b2a16ccd9c713e98f1112`。
- Blocker receipt SHA256：
  `a1d58b5a08c44670e7735bddf6aebd1a81d6cad662d7726d8eed94423f72696e`。
- Public quote-safety receipt SHA256：
  `ae426a2327dce281afdad75ea3eae6cc5b013432399098c0648d82014369fb3a`。
- Prior notional-only inventory SHA256：
  `69dd0255d3e8975599390ad55834b69cd13c51aad2c4ddb9562084a43e80f3a8`。
- Post-blocker hardening verification：
  local focused/inherited suite `218 passed`；Ruff、compileall、
  `bash -n`、Gate 0 validator 和 `git diff --check` 通过。
- c6in clean-repo inspect override probe：
  `execution_runtime_ready=true`、blockers `[]`、credential alias symlink
  secure、SDK `0.24.0`、order/cancel/query surface ready；account/private/
  order/cancel endpoint calls 均为 `false`，credential value emit/copy
  均为 `false`。
- Unified-account recovery probe：
  master identity token
  `be1875f83ebd41ae1198f77955152546087725e16556d138f9a7cbf8b5c9d889`
  与 8 月 13 日 accepted preflight 一致；configured/signer agent token
  `923b12b24a0bfd09bc9e0ecd3e715bd2295817ce2726dec8c9c4bbafb5bb619d`
  与 accepted `hp1` 一致。
- Account facts：configured/signer role `agent`、master role `user`、
  abstraction `unifiedAccount`、agent approved 且未过期、open orders
  `0`、SKHX position zero、`30 USDC` aggregate Gate pass，collateral
  source=`unified_spot_usdc_available`。
- History facts：`682` historical orders、`348` fills；
  `xyz:SKHX=12` orders / `11` fills。History receipt SHA256：
  `c9b2313831186302e2c75895292087795322112fee56ee68e7602aa45ce401b2`。
- Recovery account baseline SHA256：
  `eb25131a17292d2f41e5376670cce22fb0e4c701af7da8f1b4c7a1f23cc47c19`。
- Post-fix local verification：`221 passed`；Ruff、compileall、`bash -n`、
  Gate 0 validator、current/frozen hostile `70` executions、
  `git diff --check` 均通过。Surface Matrix SHA256：
  `4e6a357adb9b0d1d70e03644a674646be2f66d702143eda5a19dc728f0dcaf5c`。
- First resumed formal attempt at `98bc8ded`：
  Gate 0=`15/15/7`、Gate 1=`221 passed`、hostile=`35 cases /
  70 executions / fail-open 0`、notional subgate pass。Future schedule
  froze at `2026-08-23T01:48:59Z` for windows
  `02:06:00Z..02:53:00Z`，但在 private access 前停止。
- Inspect blocker：selected Python path 指向 task venv，但 resolved path
  为 `/usr/bin/python3.13`；SDK metadata 不可见，stable blocker
  `hyperliquid_sdk_surface_incomplete`。
- Boundary：credential file 仅由 inspect 读取 key names；
  `private_endpoint_called=false`、`account_endpoint_called=false`、
  `order_endpoint_called=false`、`cancel_endpoint_called=false`、
  `wallet_client_constructed=false`。
- `--copies` c6in probe：selected/resolved Python 均为 copied venv
  executable，`is_symlink=false`、SDK=`0.24.0`、
  order/cancel surface ready、execution runtime ready、blockers=`[]`。
- `121d1050` fresh Gate 2：`1662` public quote samples、full receipt
  `status=pass`、统一账户 collateral source=
  `unified_spot_usdc_available`、首笔下单前 open orders/position 为零。
- First active attempt：submit accepted、notional=`10.0056 USDC`、
  fill=`0`、position delta=`0`；fail-closed receipt 为
  `LATENCY_UNRESOLVED_EXPOSURE`，后续 emergency reconciliation
  `reconciled=true`、final open orders zero、final position zero、
  reduce-only flatten attempted=`false`。
- Active-blocker evidence：`37` files，inventory SHA256
  `5fe6e85fcca5bca34deb5b425f20669308efe627ed3ae64685db4e5d82d93cb5`；
  emergency reconciliation SHA256
  `439022aa43fc4c3b93c7f91bf1b8d6fb94827b3124a1bd52384cccea2b883fe8`。
- First resting repair：`223 passed`；Gate 0=`15/15/7`、Ruff、
  compileall、bash syntax、diff check、current/frozen hostile
  `35 cases / 70 executions / fail-open 0` 全部通过。
- `92534a32` active facts：`9` attempts、`8` eligible、fills=`0`；
  attempt 9 terminal=`cancel_confirmed`、collection final open orders=`0`、
  final position zero。L1 因不足完整 sample/window population fail
  closed，未生成 formal package。
- Final-reconciliation blocker evidence：`39` files，inventory SHA256
  `cf717321c8190fbeb624242394d3579a61788fa9493c1fc5d94b91486cdd3f34`。
- Second repair：`224 passed`；plan SHA256
  `084a8d5edc2b06366f5071eda5b78f01a13aa12f6b34e26886aa163149db1a7c`；
  Surface Matrix SHA256
  `8c1cad9654868888c9baebaeb4c714d2f4ddec4458ee0a5304527f2a8a9b30a8`。

done：
- 合同 2 已证明 notional 和 10-tick public safety 前置条件可执行。
- 已证明生产 credential source 可安全读取，且执行开始时没有挂单或
  SKHX 仓位。
- 已在首笔订单前 fail closed，并保留后来用于定位 agent/master
  identity 错误的 redacted durable evidence。
- 后续恢复 runner 已绑定统一 trading credential alias 和 clean task
  runtime discovery，不会使用 dirty shared trading checkout。
- 已证明旧 margin blocker 是 agent/account identity binding 错误，不是
  预算不足；`15 / 30 / 3 USDC` 不需要再次扩大。
- 已证明正确 unified master 满足当前 collateral Gate，并保留无原始地址、
  无订单引用的 read-only recovery evidence。
- 已修复 inspect 对 venv Python symlink 的解析歧义，且独立 probe 证明
  新 runtime discovery 路径可执行。
- 已证明首笔 active order 的 fail-closed 路径没有成交或残余仓位，并已
  完成零挂单/零仓位 reconciliation。
- 已把 HIP-3 resting 可见性延迟收口为 exact oid+cloid fallback，并为
  resting 未确认分支加入自动撤单与最终安全证明。
- 已把 terminal confirmation 后的 open-orders 可见性延迟收口为
  frozen bounded polling，不再用单次 immediate snapshot 误判。

blockers：
- 资金 blocker 已解除。
- 仍需重新冻结 future UTC windows，并从新的 committed source 执行
  fresh full Gate 2；既有窗口和 900 秒结果不能替代新的
  as-of-time evidence。
- 用户对 private read、post-only submit、cancel 和 reduce-only flatten
  的授权继续有效，无需再次取得授权。
- Active attempts、L1 recommendation、formal package、amdserver archive
  和 QA 尚未执行；H0-B 继续锁定。

commit：
- `6db3de5a0520d36efe09de055baa5edd8e876100`

提交信息：
- `fix: keep c6in inspect inside copied venv`
