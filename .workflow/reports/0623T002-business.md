执行线程：
- 业务线程-live

任务ID：
- 0623T002

状态：
- 待验收

是否进行QA验收：
- 是

QA说明：
- 无

files：
- `examples/hyperliquid/hyperliquid_tiny_live_m2_public_watcher.py`
- `examples/hyperliquid/hyperliquid_tiny_live_m2_fill_window.py`
- `examples/hyperliquid/test_hyperliquid_tiny_live_m2_event_driven_watcher.py`
- `examples/hyperliquid/test_hyperliquid_tiny_live_m2_fill_loop.py`
- `.workflow/tasks/0623T002.md`
- `.workflow/reports/0623T002-business.md`
- `local_live_analysis/hyperliquid_tiny_live_m2_fresh_touch_evidence_0623T002/**`
- `task_plan.md`
- `progress.md`
- `findings.md`

action：
- Removed event-driven synthetic `quote_aging_status=stayed_touch` as sufficient fresh-touch proof.
- Added strict event-driven freshness handling in `candidate_freshness_status()`: when `event_driven_inline_candidate=true`, freshness must come from non-synthetic BBO-history fields with `fresh_touch_evidence_status=pass`.
- Added event-driven BBO-history evidence fields: `freshness_source`, `touch_stability_ms`, `last_touch_change_ms`, `top_reset_status`, `top_reset_reason`, `fresh_touch_evidence_status`, and `fresh_touch_evidence_reason`.
- Added watcher-side BBO-history evidence derivation:
  - `real_bbo_history_touch_stability` when current touch has persisted at least `250ms`.
  - `real_bbo_history_top_reset` when same-touch top size/order-count has materially reduced.
  - `synthetic_current_event_only` / `real_bbo_history_insufficient` fail closed.
- Extended `current_candidate_audit.csv` with freshness evidence fields.
- Added focused tests for synthetic-only block, real BBO touch-stability pass, real BBO top-reset pass, and fill-window strict-mode rejection of synthetic `stayed_touch`.
- Generated local non-live artifacts under `local_live_analysis/hyperliquid_tiny_live_m2_fresh_touch_evidence_0623T002/`:
  - `synthetic_only_block`: `trigger_found=false`, `live_submissions_count=0`, mock order intents `0`, and candidate audit records `freshness_source=synthetic_current_event_only`.
  - `real_bbo_history_pass`: `trigger_found=true`, `live_submissions_count=1`, mock order intents `1`, and candidate audit records `freshness_source=real_bbo_history_touch_stability`.

verify：
- `python -m pytest examples/hyperliquid/test_hyperliquid_tiny_live_m2_event_driven_watcher.py -q` -> `13 passed`
- `python -m pytest examples/hyperliquid/test_hyperliquid_tiny_live_m2_fill_loop.py -q` -> `22 passed`
- `python -m py_compile examples/hyperliquid/hyperliquid_tiny_live_m2_public_watcher.py examples/hyperliquid/hyperliquid_tiny_live_m2_fill_window.py` -> passed
- `python examples/hyperliquid/hyperliquid_tiny_live_m2_public_watcher.py --help` -> passed
- `python examples/hyperliquid/hyperliquid_tiny_live_m2_fill_window.py --help` -> passed
- `git diff --check` -> passed
- Local artifact health: `106` files, `0` empty files.

done：
- T002 completed fresh-touch evidence hardening only.
- Event-driven current-candidate generation can no longer pass fresh-touch solely because it synthesized `stayed_touch`; it must prove real BBO-history touch stability or top reset.
- Historical public-flow CSV evidence remains compatible when it has real public-window touch / strict-through timing fields.
- No live order window was run for this task, no credentials were read, no remote checkout was refreshed, no final gate was rerun, and no real order endpoint was called by this task.
- This does not complete M2, does not prove stable PnL, and does not authorize M3, taker/crossing, one-tick-back, cap relaxation, or default-on behavior.

blockers：
- 无 task-scoped blocker.
- M2 remains blocked on missing live maker fill / fee / inventory / realized PnL proof.

commit：
- `0cffa9b`

提交信息：
- `0623 harden fresh touch evidence`
