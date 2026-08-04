# 线程回报

执行线程：
- 业务线程-python/multi-market-queue-notebook-copies

任务ID：
- 0804T005

状态：
- 待验收

是否进行QA验收：
- 是

QA说明：
- 无

files：
- `.workflow/tasks/0804T005.md`
- `.workflow/reports/0804T005-business.md`
- `examples/tutorial_reproduction/market_queue_experiments.py`
- `examples/tutorial_reproduction/notebook_support.py`
- `examples/tutorial_reproduction/refresh_notebooks.py`
- `examples/tutorial_reproduction/test_market_queue_experiments.py`
- `examples/tutorial_reproduction/test_notebook_support.py`
- `examples/tutorial_reproduction/test_advanced_experiments.py`
- `examples/tutorial_reproduction/README.md`
- `examples/tutorial_reproduction/notebooks/0804T005/` 下五个副本
- amdserver：
  `local_live_analysis/tutorial_reproduction_0804T005/executed_notebooks/`
- amdserver：
  `local_live_analysis/tutorial_reproduction_0804T005/20250801_300s/`

action：
- 保持用户列出的五个 `examples/*.ipynb` 原文件不变，在独立目录生成
  五个可 Run All 副本。
- Making Multiple Markets 使用 Binance、Bybit、OKX、Bitget、Gate
  五个真实 Tardis BTC 永续市场运行相同 Grid/PowerProbQueueModel3，
  按统一名义资金组合权益。
- Introduction 保留原教程的合成组合实验，但固定随机种子以保证确定性。
- Probability Queue Models 在相同 Binance 输入、策略、延迟、费用和参数下
  只切换 Square/Log2/Power3 三种概率队列模型。
- Queue-Based Market Making 比较 mid、book pressure、causal trade
  impulse 和 thin-queue backoff；明确 BTCUSDT 不是原 CRVUSDT
  large-tick 等价输入。
- Exchange Comparison 在同日 300 秒窗口使用五交易所真实深度、成交和
  BBO，比较 5/10/20 tick 三组统一报价深度。
- 外部交易所数据使用 task-level cache；输入 manifest 记录源路径、
  size/mtime、bounded staged SHA、行数、tick/lot/multiplier 和 fused SHA。
- 首次服务器执行发现 `convert_fuse` 不接受普通 `buffer_size`，修复为仅
  使用 `ss_buffer_size` 并新增 AST 接口回归测试。
- 结果审计发现 Bitget 前导少量盘口尚未初始化，统计层改为只使用完整
  BBO/有限权益行，并重新执行多市场和交易所对比副本。

verify：
- 本机 focused pytest：`18 passed`。
- amdserver focused pytest：`18 passed`。
- 本机/服务器 `py_compile`、`refresh_notebooks.py --task-id 0804T005
  --check` 和 `git diff --check` 通过。
- 五个副本连续再生成 SHA256 完全一致。
- 五个原始 notebook 本机与 amdserver Git diff 均为空，SHA256 为：
  - Making Multiple Markets：
    `fd01ea701d4268b0f3a8dcba4cb1cf7294e40705d326514d9f83d5f6bda2924e`
  - Introduction：
    `01c73fa5c72c9a52be727520e1b53e2e5f9d89e09bab396df61bd8a42cdf31a1`
  - Probability Queue Models：
    `38d48c7a14c27452957974ea6b47ec889f2c60c5a62106bee1ec30e5188abe04`
  - Queue-Based：
    `dd9e1166cc4fe99c587c21c252ebd3dfdfddb57e352176adbcf962029d1161d6`
  - Exchange Comparison：
    `f81146c529728104812dc8c6a754166ba800419c427832b459fe7817652390ac`
- 本机 2025-01-01、30 秒 sample 五项按顺序执行：
  `adapted / passed / adapted / adapted / adapted`，无 failed。
- amdserver 2025-08-01、300 秒按顺序执行五次 `nbconvert --execute`；
  五本均有四个活动 code cell、execution count `[1,2,3,4]`、error
  output `0`。
- amdserver manifests：`1 passed / 4 adapted / 0 failed`。
- 五市场实际可用：
  `binance-futures / bybit / okex-swap / bitget-futures /
  gate-io-futures`。
- 外部 fused 行数：
  Bybit `120133`、OKX `536906`、Bitget `7325`、Gate `20959`；
  Binance 共享 prepared data 为 `563454` 行。
- 多市场 300 秒组合 diversification ratio 为 `1.7403267550`，
  final normalized return 为 `-0.0003273191`；仅为短窗教学诊断。
- 三种 queue model 成交数：
  Square `616`、Log2 `628`、Power3 `613`；相同输入下结果有差异。
- Queue-Based 四组成交数：
  mid `536`、book pressure `510`、book+impulse `512`、
  thin-queue backoff `536`。
- 修复后五交易所 flow volatility/spread/BBO quantity 均为有限值；
  Bitget 完整 BBO 样本 `2995`。
- amdserver 最终执行 HEAD：
  `baedf7cf56e15c8311bdf58701fc297ba96cd538`。

done：
- 五个独立 notebook 副本已在本机和 amdserver 可 Run All。
- amdserver 多市场和交易所对比读取五个真实 Tardis 挂载。
- 原始 notebook 未修改。
- 所有结果含输入身份、策略边界、适配说明和短窗风险说明。

blockers：
- 无。

残余风险：
- 当前正式执行窗口只有 2025-08-01 前 300 秒，不支持长期 Sharpe、
  稳定参数或生产收益结论。
- 多市场实验是同一 BTC 永续跨 venue 分散，不是原教程的多资产横截面。
- Gate/OKX 的 contract multiplier 用于把原始合约数量换算为线性资产
  头寸；跨 venue 的原始 quantity 不应直接比较。
- Queue-Based 使用 BTCUSDT，只验证队列信号机制，不证明 CRVUSDT
  large-tick 行为。

commit：
- `c44b024b534b763f65d1809afb962fcf80533426`
- `d27dfdc22e3cb6e2bacd777098276c13b54b64bc`
- `baedf7cf56e15c8311bdf58701fc297ba96cd538`

提交信息：
- `add multi-market queue notebook copies`
- `fix external market Tardis conversion`
- `ignore uninitialized venue book rows`
