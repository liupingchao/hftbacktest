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

blockers：
- 资金 blocker 已解除。
- 仍需重新冻结 future UTC windows，并从新的 committed source 执行
  fresh full Gate 2；8 月 22 日已过期窗口和 900 秒结果不能替代新的
  as-of-time evidence。
- 用户对 private read、post-only submit、cancel 和 reduce-only flatten
  的授权继续有效，无需再次取得授权。
- Active attempts、L1 recommendation、formal package、amdserver archive
  和 QA 尚未执行；H0-B 继续锁定。

commit：
- `46ac34e4048e1c88f5d648dc9249dd6ed2ddabbf`

提交信息：
- `fix: resolve c6in unified account from agent`
