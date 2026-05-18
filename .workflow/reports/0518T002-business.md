```md
执行线程：
- 业务线程-python

任务ID：
- 0518T002

状态：
- 待验收

是否进行QA验收：
- 是

QA说明：
- 无

files：
- `.workflow/tasks/0518T002.md`
- `.workflow/reports/0518T002-business.md`
- `task_plan.md`
- `progress.md`
- `findings.md`
- `.workflow/dashboard.html`
- `.workflow/dispatch_suggestions.md`

action：
- 完成 Step 5A BBO quote-anchor / post-only design contract。
- 本任务保持 design-only：
  - 未实现 quote-anchor / post-only 代码。
  - 未修改 strategy behavior、fair/reservation、quote placement、risk guard 或 live scripts。
  - 未运行 live、未补样本、未运行 replay experiment。
- 本任务按 allowed files 约束，把要求的 5 个输出作为本业务回报中的具名设计产物记录：
  - `BBO_QUOTE_ANCHOR_POST_ONLY_DESIGN.md`
  - `quote_anchor_decision_matrix.csv`
  - `post_only_protection_contract.md`
  - `step5b_diagnostic_requirements.md`
  - `run_manifest.json`

evidence basis：
- `0514T001` QA 已通过：
  - `5-13-day-control-30min` classification = `passes_pricing_research_market_view`
  - future/missing/gap-crossed join = `0`
  - stale join rate = `0.0090949283142803`
  - BBO mismatch rate = `0.00016340354734246414`
  - top5 tick match = `0.8232061647296615`
  - top5 qty match = `0.8014359103924541`
- `0514T003` QA 已通过：
  - strongest pricing candidates include top5/top1 imbalance and microprice-family signals.
  - result is pricing research only; it does not prove strategy PnL, side-adjusted execution outcome, full L2, exact queue/fill, or live readiness.
- `0514T005` QA 已通过：
  - execution outcome label layer exists on `5-13-day-control-30min`.
  - post-only/reject/throttle/churn, queue/priority, missed opportunity, and PnL decomposition remain observed-only proxy or low-sample where appropriate.
- `0518T001` QA 已通过：
  - queue-ahead proxy design remains repair-design-only.
  - top5/trade proxy must not be described as exact queue position or exact fill proof.

`BBO_QUOTE_ANCHOR_POST_ONLY_DESIGN.md`：
- 推荐设计：
  - Quote hard anchor should be a layered fast-BBO design.
  - Primary hard protection source should be the freshest decision-time BBO/bookTicker-equivalent source available to the strategy path.
  - Depth/top5 should be used for fair-price, microprice, imbalance, liquidity, and risk context, but not as the sole final hard post-only quote anchor.
  - If a depth/top5 source disagrees with the fast BBO source, the final post-only clamp should prefer the fast BBO protection source and classify the disagreement for Step 5B diagnostics.
- Source priority:
  1. `fast_bbo_or_bookticker`: primary hard quote anchor for `bid <= best_bid` / `ask >= best_ask`.
  2. `depth_bbo`: secondary consistency check and fallback only when fast BBO is missing but depth is fresh and accepted.
  3. `top5_reconstructed_bbo`: research and diagnostic source; fallback only under strict freshness/data-quality gates and not yet authorized for implementation.
  4. `last_good_anchor`: cancel/hold-down safety fallback, not a license to place fresh add-side quotes.
- Live/replay equivalence principle:
  - Live implementation should use the same source family the exchange path can observe with the lowest stale risk.
  - Replay diagnostics should compare fast BBO, depth BBO, and top5 reconstructed BBO before any implementation.
  - Step 5B must quantify drift and stale regimes before deciding exact thresholds.

top5 role：
- top5 is a pricing input:
  - top5 imbalance, top5 microprice edge, top5 depth imbalance, and liquidity concentration are fair-price / reservation candidates from Stage 4.
- top5 is a risk/context input:
  - top5 size, age, imbalance, and queue-ahead proxies can stratify quote risk, fillability, and adverse-selection diagnostics.
- top5 is not the current final quote-anchor input:
  - Stage 3 top5 tick/qty matches are above research thresholds but not exact.
  - top5 sidecar is not full L2 or exact queue evidence.
  - Therefore top5 should not be the hard source for final `bid <= best_bid` / `ask >= best_ask` until Step 5B proves it is at least as safe as fast BBO for post-only protection.

`quote_anchor_decision_matrix.csv`：
- csv columns: `source,role,allowed_for_hard_anchor_now,allowed_for_pricing_now,allowed_for_risk_context_now,required_guard,step5b_question`
- `fast_bbo_or_bookticker,primary hard quote anchor,yes,yes,yes,freshness and latency guard,How often does fast BBO drift from depth/top5 at decision time and near rejects?`
- `depth_bbo,secondary check and guarded fallback,conditional,yes,yes,depth freshness and accepted join quality,When fast BBO is absent or stale does depth BBO remain safer than hold/cancel?`
- `top5_reconstructed_bbo,research and diagnostic context,no,yes,yes,Stage 3 market-view accepted rows only,Does top5 BBO ever lag or mismatch enough to create crossed/post-only-risk candidates?`
- `last_good_anchor,defensive hold/cancel reference,no,no,yes,max age and no add-side submit,Can hold/cancel reduce churn without hiding stale quote risk?`
- `none_or_uncertain,no fresh anchor,no,no,no,block add-side submit,What suppression/cancel behavior is needed when all anchors fail?`

`post_only_protection_contract.md`：
- Hard protection:
  - Compute candidate fair/reservation quote first.
  - Convert candidate quote to tick units before final exchange price emission.
  - For bid:
    - round candidate bid down/floor to tick.
    - clamp final bid to `min(candidate_bid_tick, anchor_best_bid_tick)`.
    - never round up through the current best ask.
  - For ask:
    - round candidate ask up/ceil to tick.
    - clamp final ask to `max(candidate_ask_tick, anchor_best_ask_tick)`.
    - never round down through the current best bid.
  - Re-check after clamp:
    - bid must be `<= anchor_best_bid_tick`.
    - ask must be `>= anchor_best_ask_tick`.
    - bid must be `< anchor_best_ask_tick`.
    - ask must be `> anchor_best_bid_tick`.
    - if the source is stale/missing/gap-crossed, do not create a fresh add-side submit.
- GTX / post-only:
  - GTX/post-only remains the exchange-enforced backstop.
  - Strategy should not rely on post-only rejects as normal quote-control flow.
  - A post-only reject is evidence of stale anchor, latency, rounding, or source drift and should be counted as a protection failure bucket.
- Stale quote prevention:
  - New add-side submit/re-add should be suppressed when anchor age, join age, or feed latency breaches the accepted threshold.
  - Existing stale quotes should be eligible for cancel/hold-down decisions rather than immediate re-add.
  - Reduce-side safety behavior may remain separate, but Step 5B must quantify it before implementation.
- Latency guard:
  - Quote placement should carry `anchor_age_ms`, `join_age_ms`, `decision_to_submit_ms`, `submit_to_ack_ms`, and latency bucket labels.
  - High latency should not directly rewrite fair price; it should gate freshness, suppress re-add churn, or force more conservative placement.
- Join-age guard:
  - Stage 3 quality categories should be reused:
    - no future join
    - no missing join
    - no gap-crossed join
    - bounded stale rate
  - Rows outside accepted quality should be diagnostic-only and not treated as quote-anchor proof.
- Reject/throttle/drop path:
  - `post_only_reject`: classify as `post_only_price_protection_failure_or_latency`; enter short cooldown/backoff for same-side re-add until a fresh anchor is observed.
  - `api_reject`: classify by reason; do not immediately re-submit at same price without a fresh anchor.
  - `throttle`: classify as execution hygiene failure; backoff quote churn and report throttle bucket.
  - `drop_or_missing_ack`: classify as lifecycle uncertainty; avoid assuming the order is safely working unless live state proves it.
  - `frequent_churn`: classify separately because churn can increase reject/throttle and adverse-selection exposure.

`step5b_diagnostic_requirements.md`：
- Step 5B must stay read-only and use `5-13-day-control-30min` first.
- Required diagnostic families:
  1. BBO source drift:
     - compare fast BBO/bookTicker, depth BBO, and top5 reconstructed BBO at decision time.
     - report tick drift, side drift, crossed-risk cases, stale/source-missing buckets.
  2. Quote distance / placement:
     - bucket candidate and actual quote distance to anchor BBO.
     - bucket touch, inside-risk, one-tick-back, multi-tick-back, and invalid/crossed candidates.
  3. Post-only risk candidates:
     - count quotes that would cross after rounding/clamp under each anchor source.
     - separate candidate formula risk from stale-anchor risk.
  4. Reject/throttle/churn:
     - summarize GTX/post-only rejects, API rejects, throttles, drops, cancel/replace churn, and fast re-add loops by anchor state.
  5. Stale / latency / join-age:
     - report anchor age, join age, feed latency, decision-to-submit latency, submit-to-ack latency, and their quantiles.
     - stratify by stale/non-stale and source mismatch buckets.
  6. Execution tradeoff:
     - use existing Stage 5 labels to summarize fill probability, time-to-fill, fill-after-cancel, markout, and spread capture by anchor/placement bucket.
     - mark queue, opportunity cost, and PnL decomposition as observed-only proxy when exact counterfactuals are unavailable.
  7. Implementation readiness:
     - produce pass/fail or diagnostic-only conclusion for whether Step 5C default-off implementation is justified.
- Step 5B must not:
  - modify quote placement.
  - change fair/reservation formulas.
  - implement post-only protection.
  - run live or replay experiment.
  - claim exact queue or production readiness.

`run_manifest.json`：
- task_id: `0518T002`
- mode: `design_only`
- dataset: `5-13-day-control-30min`
- code_changes: `workflow_docs_only`
- live_started: `false`
- replay_experiment_started: `false`
- strategy_behavior_changed: `false`
- recommended_anchor: `layered_fast_bbo_primary_depth_guarded_fallback_top5_research_context`
- top5_role: `pricing_and_risk_context_not_final_hard_anchor`
- next_task: `0518T003`
- next_task_mode: `read_only_diagnostic`

current decision：
- Quote-anchor 推荐设计：
  - hard protection anchor = fast BBO/bookTicker-equivalent source first.
  - depth BBO = consistency check / guarded fallback.
  - top5 reconstructed BBO = pricing/risk/diagnostic input, not current hard quote anchor.
- `bid <= best_bid` / `ask >= best_ask` hard protection should be based on the primary fast BBO anchor where available.
- Tick rounding should be side-conservative first, then final post-only clamp, then validity re-check.
- Stale quote prevention, latency guard, and join-age guard should suppress fresh add-side submits/re-add churn when anchor quality is not accepted.
- GTX/post-only remains exchange backstop; reject/throttle/drop paths are evidence buckets and should trigger cooldown/backoff or lifecycle uncertainty handling.
- Step 5B is required before implementation because current evidence still needs source drift, post-only risk, stale/latency, and execution-tradeoff quantification.

verify：
- 人工检查 `.workflow/tasks/0518T002.md` 和 `.workflow/reports/0518T002-business.md`
- `python3 .workflow/build_dashboard.py`
- `git diff --check`

done：
- 已明确 quote-anchor 推荐设计。
- 已明确 top5 的角色定位。
- 已明确 post-only / rounding / stale / latency / reject path 设计。
- 已明确 Step 5B 需要验证的指标。
- 已明确未实现策略、未改代码、未启动 live。

blockers：
- 无

commit：
- e380a3c

提交信息：
- docs(workflow): complete step5 quote anchor design
```
