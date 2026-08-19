执行线程：
- 业务线程-python/cross-exchange-research

任务ID：
- 0801T009

状态：
- 待验收

是否进行QA验收：
- 是

QA说明：
- 第一轮独立 QA 为 `未通过`；本报告已更新为两项 P1 validator closure
  修复后的第二轮验收输入。

files：
- `examples/hyperliquid/cross_exchange_liquidity_response_regime_v2.py`
- `examples/hyperliquid/test_cross_exchange_liquidity_response_regime_v2.py`
- `local_live_analysis/skhynix_liquidity_response_case_hierarchy/regime_v2/`
- `.workflow/tasks/0801T009.md`
- `.workflow/reports/0801T009-business.md`
- `task_plan.md`
- `progress.md`
- `findings.md`

action：
- 新增独立 `regime_v2/` builder，不修改历史 `regime/`。
- 严格验证已验收 T006 Episode v2、T007 baseline v2 R2 和 T008 motif v2
  的 11 项 exact role/path/row-count/SHA provenance。
- 从 discovery/post-selection episode pre-state features 构造每段独立的
  non-overlapping `1min` context windows。
- Context 使用 Binance/Hyperliquid spread/depth、Binance volatility、
  basis level/change、episode/atom arrival、signed flow/directionality、
  source age 和 degraded-state rate 共 13 个字段。
- Median/IQR transform 只在 discovery `0001-0003` 拟合；real 和
  surrogate detector 均绑定同一 transform SHA 和同一 pre-3/post-3
  standardized Euclidean score。
- Primary surrogate 运行 `999` 次 within-segment contiguous `5min` block
  shuffle，seed `0`；`3min/7min` 作为冻结 sensitivity diagnostics。
- 每个 surrogate 重放 discovery p95 threshold estimator、`3min` spacing
  和短区间合并。正式 empirical p 使用全扫描 maximum accepted score
  null，segment null 只作诊断。
- 仅 spacing candidate 输出 family-wise empirical p；非候选 audit 行
  留空，避免无效 p 值解释。
- Boundary publication 完成后才按 `episode_decision_ts_ns` 链接 episode
  和 motif_v2 membership。
- Motif-by-regime 仅输出 prevalence 和 prototype distance；source 与
  linked classification 均强制保持 `not_supported`。
- Frozen contract 记录 context transform、detector、surrogate、label、
  motif linkage 和历史 `regime/` 九文件 SHA。
- Manifest/contract/input/output exact closure、transform content SHA
  recomputation、legacy package drift 和 hostile source/output metadata
  均 fail closed。
- 第一轮 QA 后，detector、surrogate 和 motif-linkage nested contract
  改为 shared canonical factory + exact equality validation。
- Existing-package validator 重新读取 accepted discovery/post-selection
  feature inputs，重建全部 `240` 个 context rows，并重新计算 discovery
  medians/IQR、transform SHA、p95 threshold 和 label thresholds。
- 正式 `one_minute_context.csv.gz` 与 source-derived reconstruction
  逐行 exact 比较；协调修改 median/transform/output SHA 不能再通过。
- 新增 coordinated hostile tests：关闭 short-interval merge；同时修改
  primary block/diagnostics/seed/threshold replay；修改 median 并重算
  transform SHA。三类均 fail closed。

verify：
- focused：
  `python -m pytest examples/hyperliquid/test_cross_exchange_liquidity_response_regime_v2.py -q`
  -> `10 passed`。
- combined：
  `python -m pytest examples/hyperliquid/test_cross_exchange_liquidity_response_case_hierarchy.py examples/hyperliquid/test_cross_exchange_liquidity_response_baseline_v2.py examples/hyperliquid/test_cross_exchange_liquidity_response_motif_v2.py examples/hyperliquid/test_cross_exchange_liquidity_response_regime_v2.py -q`
  -> `51 passed in 143.02s`。
- `ruff`、`py_compile`、CLI help、formal validator 和
  `git diff --check` -> pass。
- 真实输出：
  - context windows `240`；
  - scanned boundary rows `200`；
  - discovery p95 threshold `4.887172347199368`；
  - threshold candidates `13`；
  - spacing candidates `7`；
  - primary surrogate runs `999`；
  - candidate global family-wise p-values `0.259-1.0`；
  - global boundary-count p-value `0.756`；
  - published data-driven boundaries `0`。
- Formal intervals 为八个机械 segment intervals，classification 全为
  `context_only`。
- Episode-regime membership `12,677`；motif-by-regime rows `64`；
  motif classification 全为 `not_supported`。
- 两次完整真实构建通过 deterministic existing-package gate，正式十个
  output SHA 均不变。
- 历史 `regime/` 九个文件 SHA 与 frozen contract 完全一致。
- Exact `held_out` split label 数为 `0`；正式 supported motif 数为 `0`。

done：
- regime manifest SHA-256：
  `6c5d72f411eae71c738dacef21bb3bd71a133d6c443e9027294c1c6c20c4366c`。
- frozen regime contract SHA-256：
  `43510485f9d6300edc58427613e42d04e291b7d3cf817e9d8bd08b00b7c2e4a2`。
- context/audit/boundary/surrogate SHA：
  - `511d87098f212a99f259d9847fffbc79f0a3145883b44bc9137da2ffd97af77d`
  - `82a394ecddec784bef609ca0f5c307719fd83649d1fc89f0cdc9db34b498593a`
  - `4a735cf54ca9d24f4e99983cc1fd768e0aa6317f213413b6b61c02c82026106d`
  - `91303bd517f3bd2d1f05a45516163c2ca210d657e4c632c467b310feea507b0a`
- interval/membership/motif/transition SHA：
  - `538a41f142d75f506ab1eaca9c1dedf39f1812f02e56b0a657ad148dd4c35fcd`
  - `abe2d8afdcc63fd052a67d712a6e6e5583fe69d3725835ba45e654b9693d542d`
  - `9b50070dc37e84c503a995bfdb3f443b8b45ee3b4716db9a357a17fba09cf6d4`
  - `a88732aecc8c8ba5dc95bf04a35f5c90ce21f5a18e0cc7b851a20096b939f2bb`
- 不声明 permanent market ontology、formal held-out motif support、
  tradable signal、maker identity、exact fill 或 maker PnL。

blockers：
- 无；等待独立 QA。

commit：
- 无

提交信息：
- 无
