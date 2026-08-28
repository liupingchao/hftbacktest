# 业务执行回报

执行线程：
- SKHYNIX Liquidity Break Onset A0 Execution 业务线程

任务ID：
- 0828T008

状态：
- 待验收

是否进行QA验收：
- 是

QA说明：
- 无

files：
- `.workflow/tasks/0828T008.md`
- `examples/hyperliquid/skhynix_liquidity_break_onset_a0.py`
- `examples/hyperliquid/test_skhynix_liquidity_break_onset_a0.py`
- `.workflow/reports/0828T008-business.md`
- `local_live_analysis/skhynix_liquidity_break_onset_a0_0828T008/`

action：
- 按已验收 Revision 2 contract 实现 raw-message event-driven
  `LIQUIDITY_BREAK_ONSET_V1` detector。
- 对 29 个 admitted Binance SKHYNIX captures 执行 size/SHA closure、
  causal L1-L5 reconstruction、20ms checkpoint normalization 和第二遍
  exact-event onset replay。
- normalization 只使用 2026-07-29 calibration role 拟合 denominator
  floors 和 global IQR floors，后续日期不重拟合。
- 实现 two-of-three coherence、direction ambiguity、precursor、
  active-lock release、opposite-onset switch 和 reset censoring。
- 构造 250ms outcome-blind controls，并按 frozen relaxation order 做
  no-reuse common-support matching。
- 仅使用 capture boundary/quality geometry 选择 `tau_max=10000ms`。
- 补齐合同要求的 session role、component support、normalization scale、
  precursor delay 和 control overlap compact ledgers。
- 修正 failure classification 映射：超过 300 anchors/hour 时返回
  `A0_anchor_near_continuous`，而不是泛化的 support-insufficient。
- 全程未读取 future midpoint/best-price target，未物化 outcome，未拟合
  H0/H1，未访问 private API、订单或新采集。

verify：
- `python -m pytest
  examples/hyperliquid/test_skhynix_liquidity_break_onset_a0.py -q`：
  15 passed。
- Synthetic tests 覆盖 equal-timestamp ordering、depth sequence
  fail-closed、active lock/release、opposite switch、reset censoring、
  future-anchor control boundary、no-reuse matching 和 zero-target ledger。
- `ruff check`：通过。
- `python -m py_compile`：通过。
- `git diff --check`：通过。
- Markdown fence parity：contract 134 fences，配对完整。
- 正式完整运行：
  `python -m examples.hyperliquid.skhynix_liquidity_break_onset_a0
  --verify-hashes --rebuild-cache`。
- 第二 output root 使用 byte-identical 29 raw checkpoint caches，独立重建
  final normalization、detector、controls、matching 和全部 compact
  artifacts。
- 两个 output roots 均有 28 个 manifest artifacts，路径、size、SHA
  零差异。
- 两个 `run_manifest.json` SHA256 均为
  `1f5f875d882e21aacd1be14793c3d3bd8e35f96ee8a0921290b3e6c8f774cdd0`。
- 两个 manifests 自身 size/SHA closure 均零 mismatch。

done：
- Source closure：29 captures、9 research dates、35.9172008142 hours、
  raw hashes 现场复核通过。
- Denominator floors：
  - depth `0.4906666576862335`
  - trade `0.029999999329447746`
- Primary anchors：340,068；anchor rate `9468.1098/hour`，超过冻结上限
  `300/hour` 约 31.6 倍。
- Direction counts：up 178,717；down 161,351。
- Precursor delay：p50 `6.6765ms`；p90 `28.2847ms`。
- Same-direction inter-anchor p50：`474.8195ms`。
- Active interval p50：`176.3133ms`；p90 `405.8209ms`；
  occupancy `0.556564`。
- 204,918/340,068 intervals 由 `opposite_onset` 结束，表明 detector
  主要在连续 flow 中高频翻向。
- Component presence：dep `0.9311`、trade `0.5458`、ofi `0.9169`。
- `dep+ofi` pair share `0.84794`；至少两个 component 使用 non-floor
  local scale 的 anchor share 仅 `0.64298`，低于冻结要求 `0.95`。
- Controls：443,298 candidates；250,556 matched pairs；overall common
  support `0.73678`；minimum date support `0.59027`。
- A0 gates：
  - A0-0 pass
  - A0-1 pass
  - A0-2 fail
  - A0-3 pass
  - A0-4 fail
  - A0-5 fail
  - A0-6 pass
- Canonical classification：
  `A0_anchor_near_continuous`。
- `A1_authorized=false`。按冻结合同不得开始 target materialization 或
  timeliness outcome audit。

blockers：
- `LIQUIDITY_BREAK_ONSET_V1` A0 未通过；A1 target support audit 被
  A0 passing gate 阻塞。
- 失败不是 anchor 稀少，而是 onset 定义没有把连续背景压缩成稀缺过程
  起点，同时 normalization/component family 和 control overlap 不闭合。
- 若继续研究，必须注册新的 hypothesis/version；不得在当前版本上提高
  threshold、改变 release 或查看 target 后 rescue。

commit：
- c30cad90
- cfd9ef54

提交信息：
- research: execute liquidity break onset A0
