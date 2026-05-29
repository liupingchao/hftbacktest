```md
执行线程：
- 业务线程-python

任务ID：
- 0528T003

状态：
- 待验收

是否进行QA验收：
- 是

QA说明：
- 无

files：
- `.workflow/tasks/0528T003.md`
- `.workflow/reports/0528T003-business.md`
- `docs/hyperliquid_live_replay_alignment_design.md`
- `.workflow/dashboard.html`
- `.workflow/dispatch_suggestions.md`

action：
- Read the workflow rules, current project plan/progress/findings, and `0528T003` task scope.
- Reviewed the current Binance live/replay alignment path in `examples/binance_tick_mm/align_live_run.py`, raw converter wrapper, market-view sidecar, maker acceptance, audit comparison, preflight, and deploy launcher.
- Reviewed the existing Hyperliquid raw converter at `py-hftbacktest/hftbacktest/data/utils/hyperliquid.py`.
- Reviewed the existing Hyperliquid collector primitives under `collector/src/hyperliquid/`.
- Reviewed official Hyperliquid API docs and official `hyperliquid-python-sdk` source for `l2Book`, `trades`, REST/info snapshot, WebSocket reconnect guidance, tick/lot precision, `Info` methods, and later private lifecycle surfaces.
- Reviewed local XEMM Hyperliquid files as optional engineering reference only.
- Wrote `docs/hyperliquid_live_replay_alignment_design.md`.
- Marked `0528T003` as `待验收`.
- Rebuilt workflow dashboard artifacts.

verify：
- `python -m py_compile py-hftbacktest/hftbacktest/data/utils/hyperliquid.py` -> passed.
- `python examples/binance_tick_mm/align_live_run.py --help` -> passed.
- `python examples/binance_tick_mm/maker_acceptance.py --help` -> passed.
- `python3 .workflow/build_dashboard.py` -> passed, loaded `73` tasks and `143` reports.
- `git diff --check` -> passed.

done：
- Produced the design document at `docs/hyperliquid_live_replay_alignment_design.md`.
- The design decomposes the current Binance pipeline into exchange-neutral pieces and Binance-specific assumptions.
- The design defines the Hyperliquid first-stage boundary as read-only `l2Book` + `trades` raw market-data alignment with raw provenance, raw-to-npz mapping, top-N sidecar, and Hyperliquid-specific market-view acceptance.
- The design maps official references for `l2Book`, `trades`, REST/info snapshot/bootstrap usage, precision/rounding, and later private lifecycle surfaces.
- The design states XEMM can be used only as engineering reference, not schema truth or acceptance evidence.
- Recommended next task: Hyperliquid read-only raw market-data sample / converter / sidecar validation.
- No Binance strategy behavior changed.
- No Hyperliquid connector was implemented.
- No live or remote deploy was run.
- No default-on, guard relaxation, parameter search, tiny-live, or promotion claim was made.

blockers：
- 无

commit：
- 待提交

提交信息：
- 待提交
```
