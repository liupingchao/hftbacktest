```md
执行线程：
- 业务线程-python

任务ID：
- 0526T006

状态：
- 待验收

是否进行QA验收：
- 是

QA说明：
- 无

files：
- `.workflow/tasks/0526T006.md`
- `.workflow/reports/0526T006-business.md`
- `task_plan.md`
- `progress.md`
- `findings.md`
- `.workflow/dashboard.html`
- `.workflow/dispatch_suggestions.md`

action：
- 阅读 `0526T006` 任务合同、`0526T005` 业务/QA 结果、Stage 9H maker-edge triage artifacts，以及 `0526T004` min-move sweep 负面结论。
- 产出 focused maker-edge design contract：`inventory_aware_quote_placement_request`。
- 本轮只做 design/plan，没有实现 runner、没有改策略代码、没有跑 parameter search、没有采集 live、没有 default-on 或 promotion。

input evidence：
- `0526T005` triage 结论：
  - `inventory` score `7.0`，clean fills `365`，clean 5s markout range about `92.44` ticks。
  - `quote_distance` score `7.0`，clean fills `365`，clean 5s markout range about `303.60` ticks。
  - `size_side` score `7.0`，clean fills `365`，clean 5s markout range about `283.60` ticks。
  - `fair_price` and `reservation` score `6.0`，both promising but similar in current Stage 5 label view。
- `0526T004` min-move projected-suppression grid 结论：
  - `sweep_seed_promising=0`
  - `reject=80`
  - `not_decisionable=676`
  - 当前不应继续把研发主线放在 `min_move_quote_age_churn_guard` projected-suppression grid 上。

final design objective：
- Design one focused default-off request layer for later read-only evaluation:
  - name: `inventory_aware_quote_placement_request`
  - purpose: use inventory state to decide side preference and size pressure, use fair/reservation edge to decide whether a quote is worth keeping/placing, and use quote-distance buckets to define participation frontier.
- The design should improve maker execution quality by:
  - reducing toxic inventory-increasing add-side fills
  - preserving or improving recovery-side participation
  - avoiding pure fill suppression being mistaken as improvement
  - keeping stale/latency/post-only context as guard/safety context, not as main alpha

state variables：
- `inventory_bucket`:
  - `flat`
  - `mild_skew`
  - `large_skew_or_low_score`
- `side_class`:
  - `add_side`
  - `reduce_side`
  - `flat_side`
- `edge_bucket`:
  - derived from fair/reservation side-adjusted edge at decision time
  - `edge_strong_favorable`
  - `edge_weak_or_neutral`
  - `edge_adverse`
- `quote_distance_bucket`:
  - `touch`
  - `one_tick_tight`
  - `step_back_gt1`
  - `outside_or_no_quote`
- `size_side_bucket`:
  - existing side/size context grouped into add-side / reduce-side / flat-side request behavior
- `safety_context`:
  - `post_only_safe`
  - `post_only_risk`
  - `anchor_fresh`
  - `anchor_stale`
  - `latency_normal`
  - `latency_stale`
  - These are gating/context fields only; they cannot become the main alpha in this design.

candidate policy skeleton：
- Flat inventory:
  - allow both sides to follow baseline unless fair/reservation edge is clearly adverse.
  - if `edge_strong_favorable` and quote is at `touch` or `one_tick_tight`, keep/place request may be emitted.
  - if `edge_weak_or_neutral`, produce `no_change` unless quote-distance is already safe and baseline would quote.
  - if `edge_adverse`, request widen/step-back or no-change suppression only for the adverse side.
- Mild skew:
  - reduce-side should be preserved when edge is not adverse.
  - add-side should require stronger fair/reservation edge than reduce-side.
  - add-side at `touch` is allowed only with `edge_strong_favorable` and clean safety context.
  - add-side with `edge_weak_or_neutral` should request smaller size or one-step wider placement, not forced cancellation by default.
  - reduce-side with `edge_weak_or_neutral` may remain baseline to avoid blocking recovery.
- Large skew or low score:
  - reduce-side gets priority if edge is not adverse.
  - add-side requires `edge_strong_favorable`; otherwise request size reduction, wider quote, or no-change fallback.
  - add-side at `touch` should be avoided unless fair/reservation edge is strongly favorable and post-only/anchor/latency context is clean.
  - recovery-side quotes should not be suppressed merely because quote age, min move, or churn context looks unfavorable.

quote-distance mapping：
- `touch`:
  - highest participation and fill-risk region.
  - allowed for reduce-side when edge is not adverse.
  - allowed for add-side only when inventory is flat or mild skew with strong favorable edge; stricter for large skew.
- `one_tick_tight`:
  - preferred compromise for mild/large skew add-side when edge is favorable but not enough for touch.
  - may be used to keep participation while reducing adverse selection.
- `step_back_gt1`:
  - default defensive placement for weak add-side edge under inventory pressure.
  - should not become a blanket suppression bucket because T005 showed quote-distance has large separation but not enough to prove a one-dimensional rule.
- `outside_or_no_quote`:
  - only for adverse edge plus inventory pressure or unsafe post-only/stale context.
  - must be measured as participation loss separately from quality improvement.

size-side mapping：
- Add-side:
  - flat inventory: baseline size unless edge adverse.
  - mild skew: reduce size when edge weak/neutral; preserve only when edge strong.
  - large skew/low score: suppress or sharply reduce size unless edge strong favorable.
- Reduce-side:
  - preserve size or priority when edge not adverse.
  - do not use min-move/churn-only suppression against recovery-side quoting.
  - if edge adverse, fall back to baseline/no-change rather than forcing a quote update.
- Flat-side:
  - use as neutral control bucket; no aggressive request unless edge is strong and quote-distance is safe.

no-change fallback：
- Emit no request when:
  - required decision-time-visible fields are missing.
  - inventory bucket cannot be classified.
  - fair/reservation edge cannot be computed.
  - safety context says post-only or anchor state is unsafe.
  - the candidate would only suppress fills without preserving a recovery-side or edge-quality rationale.
  - the side is reduce-side and edge is weak/neutral but not adverse.
  - caveated-sample-only evidence is the only support for the request shape.

decision-input policy：
- Allowed decision inputs:
  - current inventory/position and inventory score/bucket
  - side class derived from current position and quote side
  - decision-time fair/reservation edge
  - decision-time quote distance to BBO/anchor
  - decision-time size/side context
  - spread/post-only/anchor-age/join-age/latency context as safety gates only
- Forbidden decision inputs:
  - future fill outcome
  - future markout
  - future spread capture
  - fill-after-cancel outcome
  - same-sample PnL optimization feedback
  - exact queue position or hidden queue assumptions

proposed action outputs：
- `side_preference_request`:
  - `prefer_reduce_side`
  - `allow_both_sides`
  - `discourage_add_side`
- `quote_distance_request`:
  - `keep_baseline`
  - `allow_touch`
  - `prefer_one_tick_tight`
  - `prefer_step_back_gt1`
  - `outside_or_no_quote`
- `size_request`:
  - `baseline_size`
  - `reduce_add_side_size`
  - `suppress_add_side`
  - `preserve_reduce_side_size`
- `candidate_action`:
  - `no_change`
  - `request_quote_adjustment`
  - `request_size_adjustment`
  - `request_side_priority`
- This task does not implement these outputs; it only defines them for a later read-only runner.

evaluation metrics for later runner：
- Participation / suppression:
  - decision rows covered
  - submit count delta
  - fill count delta
  - fill rate by bucket
- Fill quality:
  - 5s side-adjusted markout
  - spread capture
  - fill-after-cancel rate
  - time-to-fill buckets
- Inventory quality:
  - inventory-increasing fills
  - inventory-reducing fills
  - recovery-side fill quality
  - inventory cycle exposure
- Quote mechanics / safety:
  - quote-distance x inventory frontier
  - post-only risk after recheck
  - reject/throttle/drop exposure
  - churn and cancel/readd exposure
- Stability:
  - clean-only primary summary
  - caveated-sample sensitivity summary
  - per-sample direction consistency
  - fill-mass sufficiency

distinguishing rules：
- True toxic-fill reduction:
  - add-side or inventory-increasing fills decrease while 5s markout/spread capture improves, and reduce-side participation is preserved.
- Improved recovery-side participation:
  - inventory-reducing fills or recovery-side fill quality improves without higher post-only/reject/throttle risk.
- Simple fill suppression:
  - total fills fall materially but markout/spread/inventory quality do not improve.
- Caveated-only effect:
  - effect appears only when caveated samples are included.
- Insufficient-fill uncertainty:
  - bucket has too few clean fills or too few sample-level observations to distinguish quality from noise.

later read-only runner contract：
- Proposed task:
  - `0527T002` or next available task id
  - title: `Read-only inventory-aware quote placement runner implementation`
- Runner mode:
  - read-only, default-off, offline evaluation only
  - no live, no production behavior change, no parameter search on first implementation
- Expected inputs:
  - `local_live_analysis/stage9h_maker_edge_triage_0526T005/**`
  - Stage 5 execution label directories for current-format samples
  - Stage 5C quote-anchor/post-only diagnostics
  - Step 9B/9C/9D artifacts where available
  - current-format sample directories:
    - `5-19-day-control-30min`
    - `5-19-night-active-30min-a`
    - `5-19-night-active-30min-b`
    - `5-19-night-active-30min-c`
    - `5-21-day-control-60min`
    - `5-26-active-minmove-control-30min-a`
    - `5-26-active-minmove-control-60min-a`
    - optional sensitivity inputs after QA: `5-26-active-makeredge-control-180min-a`, `5-26-active-minmove-control-30min-b`
- Required outputs:
  - `run_manifest.json`
  - `candidate_decision_rows.csv`
  - `bucket_metrics_by_inventory_side_edge_distance.csv`
  - `clean_only_stability_summary.csv`
  - `caveated_sample_sensitivity.csv`
  - `participation_and_fill_loss.csv`
  - `inventory_recovery_quality.csv`
  - `quote_mechanics_safety.csv`
  - `candidate_recommendation.md`
- Verdict taxonomy:
  - `focused_design_promising`
  - `promising_but_needs_parameter_sweep`
  - `needs_more_clean_fills`
  - `too_conservative_fill_loss`
  - `caveated_only`
  - `reject`
  - `not_decisionable`

verdict rules for later runner：
- `focused_design_promising`:
  - clean-only multi-sample evidence improves fill quality or inventory recovery without material participation collapse or safety regression.
- `promising_but_needs_parameter_sweep`:
  - direction is positive but threshold strength or size multiplier requires bounded parameter design.
- `needs_more_clean_fills`:
  - direction is plausible but clean fill mass is insufficient.
- `too_conservative_fill_loss`:
  - fill loss dominates and quality metrics do not compensate.
- `caveated_only`:
  - conclusion depends on caveated samples.
- `reject`:
  - clean-only evidence worsens markout/spread/inventory or safety.
- `not_decisionable`:
  - coverage or label availability is insufficient.

non-goals：
- No strategy implementation.
- No live run.
- No default-on behavior.
- No production config change.
- No parameter search.
- No queue/touch repair.
- No replay audit bloat repair.
- No `ready_for_tiny_live_design` claim.

verify：
- `python3 .workflow/build_dashboard.py`
- `git diff --check`
- Manual consistency check:
  - no code/runner/live/parameter-search files were modified
  - design is one focused track: `inventory_aware_quote_placement_request`
  - min-move/churn grid is not promoted

done：
- `0526T006` design contract is complete and supports a later read-only implementation task.
- Task status moved to `待验收`.
- This task does not authorize tiny-live, default-on, live promotion, guard relaxation, or `ready_for_tiny_live_design`.

blockers：
- 无

commit：
- 25c45b5

提交信息：
- docs(workflow): complete 0526T006 design contract
```
