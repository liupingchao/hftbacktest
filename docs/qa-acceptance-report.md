# QA 验收结果

执行线程：
- QA验收线程

任务ID：
- 0529T003

状态：
- 已通过

更新时间：
- 2026-05-29 16:22 CST

验收线程：
- QA验收线程

验收对象：
- 业务线程-python 0529T003

验收结论：
- 已通过
- 结论说明：
  - `0529T003` 按合同完成了 Hyperliquid 第一阶段本地只读 market-data alignment 证据链：raw parse、converter、npz、raw provenance、raw-to-npz mapping、top-N sidecar、synthetic as-of join 和 classification 均可复现。
  - 任务正确将本地样本分类为 `limited_pricing_research`，没有把缺失 subscription/session/recovery evidence 的样本过度声明为 `passes_pricing_research_market_view`。

关键验收结果：
- raw messages: `55`
- `l2Book` messages: `46`
- `trades` messages: `9`
- trade events: `64`
- converted npz rows: `413`
- raw parse errors: `0`
- top-N coverage: `1.0`
- synthetic join coverage: `1.0`
- future joins: `0`
- missing joins: `0`
- event-order validation: `passed`
- sample classification: `limited_pricing_research`

QA 复跑：
- `python examples/hyperliquid/hyperliquid_raw_alignment.py --help` 通过。
- `python -m pytest examples/hyperliquid/test_hyperliquid_raw_alignment.py -q` 通过，`3 passed`。
- `python -m py_compile py-hftbacktest/hftbacktest/data/utils/hyperliquid.py` 通过。
- runner 复跑到 `/tmp/qa_0529T003_alignment` 通过，输出 `classification=limited_pricing_research`、`l2Book=46`、`trade_events=64`、`npz_rows=413`、`join_coverage=1.000000`。
- `python3 .workflow/build_dashboard.py` 通过，loaded `75` tasks and `147` reports。

残余风险：
- 本任务使用既有本地样本，不包含 subscription ack、session id、connection attempt、reconnect/recovery snapshot 证据，因此不能作为 Hyperliquid fresh public collection 完整验收。
- 后续应执行 `0529T004`，只读补齐 fresh public sample / session / recovery evidence。

提交信息：
- business/artifact commit：`8dfff8b` `Add Hyperliquid raw alignment artifacts`
- business report commit：`ff1ca4b` `Record 0529T003 business report`
- QA report commit：待提交
