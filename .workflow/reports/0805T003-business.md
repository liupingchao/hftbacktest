# 线程回报

执行线程：
- 业务线程-python/research-postprocess + cross-exchange-signals

任务ID：
- 0805T003

状态：
- 待验收

是否进行QA验收：
- 是

QA说明：
- 无

files：
- `examples/hyperliquid/cross_exchange_postprocess/basis_dislocation.py`
- `examples/hyperliquid/cross_exchange_postprocess/pipeline.py`
- `examples/hyperliquid/cross_exchange_postprocess/profiles.py`
- `examples/hyperliquid/cross_exchange_postprocess/reporting.py`
- `examples/hyperliquid/test_cross_exchange_basis_dislocation.py`
- `examples/hyperliquid/test_cross_exchange_postprocess.py`
- `.agents/skills/cross-exchange-postprocess/`
- `docs/cross_exchange_postprocess_pipeline.md`
- amdserver output:
  `/home/molly/project/hftbacktest/local_live_analysis/skhynix_cross_exchange_basis_postprocess_0805T003`
- deterministic isolated outputs:
  `/home/molly/project/hftbacktest/local_live_analysis/skhynix_basis_dislocation_0805T003_build_a`
  and
  `/home/molly/project/hftbacktest/local_live_analysis/skhynix_basis_dislocation_0805T003_build_b`

action：
- 新增单数据集 point-in-time basis/dislocation stage，使用 Binance
  `bookTicker` 与 Hyperliquid BBO 的同机 local receipt-time union。
- 同 timestamp 顺序固定为 reconnect reset、Binance、Hyperliquid；
  strict as-of 保证 source timestamp 不晚于 decision timestamp。
- fast-L2 core reconnect 起点显式清空 Hyperliquid BBO，新 epoch 首条 BBO
  前禁止旧状态 forward-fill。
- 输出 midpoint basis、`d_bh`、`d_hb`、两边 spread、100ms change、
  15min trailing median/MAD/z、60s Binance volatility、source age 和
  core/fast/auxiliary masks。
- rolling 统计固定 `closed=left`；mask 固定 inclusive start/end。
- 每行验证 directional-spread identity；future join、crossed book、
  epoch regression 或 identity error 均 fail closed。
- 每个 consumed R0 mask/hot artifact 在计算前必须匹配 manifest 声明的
  SHA 和 row count；缺失或不一致均 fail closed。
- 100ms change、15min rolling state 和 feature warmup 在 fast reconnect
  后按新 state group 重启，不跨 connection epoch 借用历史。
- 使用固定 gzip header 构建 deterministic state，并绑定 R0/R1 manifest
  和所有实际输入 SHA。
- 新增可执行 `basis-research` profile；`signal-research` 仍因 lead-lag、
  maker diagnostics 未实现而拒绝执行。
- Atom/Episode/Motif/Regime 保持 deferred/not_run，未接入当前 pipeline。

verify：
- 本机 focused + adjacent regression：`61 passed in 0.66s`。
- 新 stage/pipeline 聚焦测试：`19 passed`。
- `py_compile`、CLI help/profile、Skill quick validation 和 scoped
  `git diff --check` 均通过。
- amdserver 安装并确认 `polars 1.39.3`；远端聚焦测试 `19/19` 通过。
- Aug05 accepted R0/R1 上两次隔离 basis build 均约 `21.9s`，包含
  manifest SHA/row-count admission audit。
- 两次 build 的 `6/6` 输出 byte-identical，完整 inventory SHA 均为
  `ad424495322006063ae43199e911d40167c67bed4a435909dda8ce8431319f77`。
- 真实 `basis-research` pipeline 用时约 `155.24s`，R0 `4/4`、R1
  `5/5` 路径无关核心 artifacts 与 golden 一致。
- 真实 resume 复用 `5/5` stages；validator 返回 `passes=true`、
  `failures=[]`。

done：
- Aug05 输出 `2,641,172` 行 state；`2,634,210` 行 book eligible，
  `2,250,643` 行 feature eligible。
- 输入包含 `2,521,391` 条 Binance BBO 和 `119,986` 条 Hyperliquid
  BBO；检测并执行 `1` 次 Hyperliquid fast reconnect reset，并在新
  epoch BBO 前抑制 `3` 条旧 epoch BBO。
- future joins、crossed books、epoch regressions 均为 `0`；
  old-epoch state leaks 为 `0`，spread identity 最大误差为 `0.0`。
- reconnect 后抑制 `16` 个跨 epoch 100ms change 连接，并重新执行
  15min feature warmup。
- source campaign、既有 R0、既有 R1 的完整 inventory fingerprint 在
  双构建和完整 pipeline 前后保持不变。
- 输出已可作为后续 lead-lag stage 的唯一规范 point-in-time 输入。

blockers：
- 无。
- 本任务不包含 future outcomes、lead-lag inference、maker diagnostics、
  exact fill、可执行套利、PnL 或因果 Binance lead 结论。

commit：
- 无

提交信息：
- 无
