# 0726T068 Business Execution Report

执行线程：
- 业务执行线程

任务ID：
- 0726T068

状态：
- 执行中

是否进行QA验收：
- 否

QA说明：
- 当前只完成 no-order preflight；真实 live evidence 尚未执行，暂不进入 QA。

files：
- `examples/hyperliquid/cross_exchange_live_remote_orchestrator.py`
- `examples/hyperliquid/test_cross_exchange_live_remote_orchestrator.py`
- `.workflow/tasks/0726T068.md`
- `local_live_analysis/seeded_dynamic_live_preflight_0726T068/`
- `local_live_analysis/seeded_dynamic_live_preflight_0726T068_r1/`
- Workflow tracking files

action：
- Rendered the exact seeded-dynamic one-window command locally without reading
  credentials or starting the watcher.
- Verified EC2 and SSM health.
- Used SSM to inspect clock, disk, memory, service/process and lock state
  without calling any trading endpoint.
- Archived exact commit
  `8736efb36a0866f2c62c51e4afe4fb590c9ac72b`, transferred it to awsserver1 and
  extracted it into the isolated
  `/home/admin/hftbacktest-cross-exchange-0726T068` source root.
- Verified remote source hashes match local source/seed bytes.
- Changed only the isolated source/preflight ownership to `admin:admin`.
- Ran the exact orchestrator with `--preflight-only` as `admin` on Debian and
  pulled the artifact/receipts back locally.
- Found that three separately sealed `windows=1` invocations would all reuse
  artifact window ID and run ID `window_01`.
- Added a default-off `--window-id-offset` mapping. Local invocation
  directories remain `window_01`, while watcher artifact/run identity can be
  globally sealed as `01/02/03`.
- Added negative-offset, exact seeded preflight and runtime status regressions.
- Archived exact repair commit
  `a0bc92898ecea43cbdc4219efc1acbcd69580969`, transferred it with archive
  SHA-256
  `2e5619d7910a7c8ff18970519974e14b2439225ca677b4387f8bee57cd96e8a7`
  and extracted it into the isolated R1 source root.
- Ran three independent Linux-admin `--preflight-only` invocations with
  offsets `0/1/2`, then pulled and independently verified every preflight and
  critical source hash.
- Received fresh exact authorization for an earliest start of
  `2026-07-26T09:00:00Z`.
- Added an external read-only account identity guard and detached three-window
  controller. These do not alter the exact strategy source commit.
- The guard pins an opaque account-scope token, signer token and authorized
  identity artifact hash. Every pre/post window query must match all pins.

verify：
- EC2:
  `running / system ok / instance ok`.
- SSM:
  `Online`.
- Clock:
  `NTPSynchronized=yes`.
- Resources:
  root disk `75%`, `/tmp` `68%`, available memory about `1.8 GiB`, no swap.
- Runtime:
  `xemm.service=inactive`, live processes `0`, lock holders `0`.
- Remote preflight:
  `status=pass`, source commit exact, Linux `/home/admin/...` paths exact.
- Envelope:
  `two-sided-seeded-dynamic-manager`, one `1800s` window, `0.005 BTC`,
  `2` submissions, `0.01 BTC` position, `1 USDC` loss, `Alo`, strict gate.
- Seed:
  exact
  `e35c7fd8f3ec8268e5d50c7963889b73409470c9f01f92ca3a0d598a96562be9`.
- Source hashes:
  remote watcher, loader, contract and exposure bytes match local.
- Execution boundary:
  watcher/credential/private/account/order/cancel are all false.
- Focused tests:
  `42 passed`.
- Full Hyperliquid:
  `1309 passed, 2 skipped`.
- R1 artifact identity:
  exact window IDs and run IDs `01/02/03`; no cross-window identity reuse.
- R1 preflight artifacts:
  `f43e1170de717a21c8fc68ae2309c55b92639626023de2b2516982f943b8bd96`,
  `a85672dd7ea3ad4f21c3117da2f0356af22edbece2d80363663bb129f2a9d9c1`,
  `36b678480f3f7edf01d5cdc7ccafcfa7526b03cbd465c1bae9e194be458d0b89`.
- Final SSM assertion:
  `2026-07-26T07:56:15Z`, `xemm.service=inactive`, live Python processes `0`,
  lock holders `0`, private/account/order/cancel actions `0`.
- Authorized private gate:
  open orders `0`, BTC position `0.0`, source exact and opaque account/signer
  identities pinned; raw credentials written `false`.
- One SSM post-assertion attempt failed only because shell quoting stripped
  Python string quotes after the preflight JSON had been generated. A separate
  successful command verified that exact artifact, ownership, service/process
  state and hash; both receipts are retained.
- The first R1 SSM command failed before extraction because Linux
  `fs.protected_regular` prevented root from opening the admin-owned stale lock
  file. The retained replacement checks the same lock as `admin` and passed;
  no source, watcher or account action occurred in the failed command.

done：
- Phase 0 no-order preflight is complete.
- The initial preflight is superseded by R1.
- Exact R1 source and three Linux commands are ready for sequential,
  separately sealed one-window live invocations.

blockers：
- Fresh exact authorization was received for an earliest start of
  `2026-07-26T09:00:00Z`.
- Live execution remains fail-closed on the private/account/infra gate and
  exact account-identity continuity.

commit：
- Implementation:
  `a0bc92898ecea43cbdc4219efc1acbcd69580969`.
- R1 evidence:
  `54ec316c921c26d427fdb0729ea6cdb472a62eea`.

提交信息：
- Implementation:
  `Separate live artifact window identities`.
- R1 evidence:
  `Record T068 distinct-window live preflights`.
