# 线程回报

执行线程：
- 总控 auto-loop / 业务实现线程

任务ID：
- 0718T016

状态：
- 待验收

是否进行QA验收：
- 是

QA说明：
- 无

files：
- `examples/hyperliquid/cross_exchange_shared_signal_kernel.py`
- `examples/hyperliquid/test_cross_exchange_shared_signal_kernel.py`
- `examples/hyperliquid/cross_exchange_skew_alpha_interaction_acceptance.py`
- `examples/hyperliquid/test_cross_exchange_skew_alpha_interaction_acceptance.py`
- `.workflow/tasks/0718T016.md`
- `progress.md`
- `findings.md`

action：
- 新增 typed `ReservationResult`、`TwoSidedQuotes`、`compute_reservation_price()`、`compute_two_sided_quotes()` 和 `inventory_quote_sides()`。
- Reservation audit 包含 raw/bounded position ratio、position notional、inventory penalty ticks/price、hard-cap breach 和 reservation price。
- Long inventory 降低 reservation，short inventory 提高 reservation；penalty 在 max position ratio 上有界。
- `NEAR_POSITION_CAP_RATIO=0.8` 起进入 reduce-only mode：long 保留 sell，short 保留 buy；hard cap 独立记录，skew 不替代 runtime cap。
- Quote 先由 reservation +/- half-spread 产生，再调用唯一 `post_only_price()`；记录 desired/final price、clamp reason、edge change 和 post-only invariant。
- 当 alpha 方向被 inventory gate 去掉时，新增 `signal_side` 保留预测方向，legacy `side/quote_px` 指向实际 eligible quote，避免 markout consumer 错误解释。
- 新增 C12 acceptance runner，以相同 decision universe 比较 alpha+zero-skew 与 alpha+bounded-skew，分别报告 observed fill、conservative proxy fill 和 censored no-fill。
- Runner 当前输入是 deterministic fixture structural evidence；`real_lifecycle_fill_evidence_count=0`，所有 observed-fill-shaped rows 标记为 `fixture_label_not_live_lifecycle`，不得解释为真实成交证据。
- C12 summary 包含 quote distance、observed/proxy fill coverage、1s/5s observed markout、spread retention、peak inventory、recovery duration、add/reduce opportunity。
- 结构验收通过，但 skew enablement recommendation 保持 `remain_disabled_pending_real_c12_evidence`。

verify：
- `python -m pytest examples/hyperliquid/test_cross_exchange_shared_signal_kernel.py examples/hyperliquid/test_cross_exchange_skew_alpha_interaction_acceptance.py -q`
  - `13 passed`
- 完整相关 pricing/skew/shadow/replay/price-math regression：
  - `33 passed in 0.15s`
- `python examples/hyperliquid/cross_exchange_skew_alpha_interaction_acceptance.py --help`
  - pass
- `python examples/hyperliquid/cross_exchange_skew_alpha_interaction_acceptance.py --output-dir /tmp/0718T016-c12-acceptance`
  - final recommendation: `offline_c12_structural_acceptance_pass_keep_skew_disabled`
  - observed fill coverage: `0.5` per case
  - censored no-fill: `3/6` per case
  - all structural gates: pass
- `python -m py_compile ...`
  - pass
- `git diff --check`
  - pass
- 未执行 live、credential、private、order、cancel、network、remote 或 service action。

done：
- Principal Alignment Task 5 的 reservation/skew contract 与 C12 offline pre-acceptance 已完成；skew 仍默认关闭，等待 QA。

blockers：
- 无

commit：
- pending implementation commit

提交信息：
- pending implementation commit
