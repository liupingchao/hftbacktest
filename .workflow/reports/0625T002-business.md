```md
执行线程：
- 业务线程-research-aws

任务ID：
- 0625T002

状态：
- 待验收

是否进行QA验收：
- 是

QA说明：
- 无

files：
- `.workflow/tasks/0625T002.md`
- `.workflow/reports/0625T002-business.md`
- `examples/hyperliquid/cross_exchange_sample_expansion.py`
- `examples/hyperliquid/test_cross_exchange_sample_expansion.py`
- `local_live_analysis/cross_exchange_public_sample_xemm_0625_t002_*/**`（本地大体积原始/中间证据，未提交 Git）
- `local_live_analysis/cross_exchange_lead_lag_join_xemm_0625_t002_*/**`（本地中间证据，未提交 Git）
- `local_live_analysis/cross_exchange_lead_lag_analysis_xemm_0625_t002_*/**`（本地中间证据，未提交 Git）
- `local_live_analysis/binance_led_hyperliquid_pricing_signal_xemm_0625_t002_*/**`（本地中间证据，未提交 Git）
- `local_live_analysis/cross_exchange_mvp_sample_expansion_0625T002/**`
- `docs/cross_exchange_maker_mvp_plan.md`
- `task_plan.md`
- `progress.md`
- `findings.md`

action：
- 在 `awsserver1` 的隔离 checkout `/home/admin/hft_live/hftbacktest_0625T002` 上，以 commit `198ed46` 和 `/home/admin/hft_live/venv/bin/python` 收集了三个新的 Binance `BTCUSDT` + Hyperliquid `BTC` public-only 同步窗口。
- 远端只接受 public raw/provenance；远端自动 alignment 仅作为诊断。所有 accepted Binance/Hyperliquid alignment、as-of join、lead-lag analysis 和 pricing rows 均从 checksum-verified raw 在本地重跑。
- 本地 Hyperliquid alignment 使用兼容环境 `/Users/liu/.local/conda/envs/hftbacktest/bin/python`；默认 Python 3.13 环境的 numpy/numba 组合不可用，未修改依赖或伪造结果。
- 新增 task-scoped offline runner，先使用 decision-time/public 波动、交易事件强度和 top5 流动性字段划分相对 regime，再读取 pricing/future-label 文件。
- 生成 1000ms symmetric contexts；每行同时保留当前 HL bid 作为 buy-touch alternative、当前 HL ask 作为 sell-touch alternative，不选择 side。
- 生成全部八个 required task artifacts，并把最终 recommendation 限定为任务允许枚举值。

samples：
- `xemm_0625_t002_utc15_a`
  - start `2026-06-25T14:54:54.200279+00:00`
  - requested/actual/overlap `1800s / 1800.004869s / 1799.999859s`
  - intended session `UTC15-a`; observed regime `high_activity_liquidity`
  - Binance bookTicker/depth/trade `2450781/68052/378933`
  - Hyperliquid l2Book/trade messages/trade events `336/6870/33622`
  - reconnects Binance/HL `0/0`
  - raw SHA256 match Binance/HL `true/true`
- `xemm_0625_t002_utc15_b`
  - start `2026-06-25T15:28:40.471204+00:00`
  - requested/actual/overlap `1800s / 1800.036434s / 1800.036434s`
  - intended session `UTC15-b`; observed regime `normal_activity_liquidity`
  - Binance bookTicker/depth/trade `1994240/67643/248722`
  - Hyperliquid l2Book/trade messages/trade events `335/5387/21450`
  - reconnects Binance/HL `0/0`
  - raw SHA256 match Binance/HL `true/true`
- `xemm_0625_t002_utc16_c`
  - start `2026-06-25T16:02:40.149997+00:00`
  - requested/actual/overlap `1800s / 1800.059208s / 1800.059208s`
  - intended session `UTC16-c`; observed regime `low_activity_liquidity`
  - Binance bookTicker/depth/trade `1832122/67534/245957`
  - Hyperliquid l2Book/trade messages/trade events `335/5526/21636`
  - reconnects Binance/HL `0/0`
  - raw SHA256 match Binance/HL `true/true`
- Window start separations are `2026.270925s` and `2039.678793s`, both above 30 minutes.

evidence：
- Binance alignment npz/top5 rows:
  - A `13958425/68053`
  - B `10882191/67644`
  - C `10330829/67535`
- Hyperliquid alignment npz/top5/decision rows:
  - A `55989/336/3596`
  - B `42344/335/3596`
  - C `42462/335/3595`
- Join/primary/excluded rows:
  - A `3596/671/2925`
  - B `3596/669/2927`
  - C `3595/667/2928`
- Future/missing/stale Binance join counts are `0/0/0` for every window.
- Binance source-age p50/p90/p99 ms:
  - A `13.623150/24.025461/27.815766`
  - B `12.338718/23.990780/26.347699`
  - C `13.649481/24.057386/26.407653`
- Regime decision-time inputs:
  - A mean rolling RV `9.20857357`, combined public trade events `229.19660223/s`
  - B mean rolling RV `6.30155019`, combined public trade events `150.09251750/s`
  - C mean rolling RV `6.04992684`, combined public trade events `148.65788796/s`
- 1000ms context rows / complete rows:
  - A `669/668`
  - B `667/666`
  - C `665/665`
  - aggregate `2001/1999`
- The only incomplete contexts are the first-row missing `binance_mid_move_ticks_from_prev` values in A/B; they are reported and excluded, not imputed.
- Nominal 1000ms effective-age coverage:
  - A count/min/p50/mean/p90/max `669/1000/5000/5111.360239/5500/6000ms`
  - B `667/1000/5000/5123.688156/5500/6000ms`
  - C `665/1000/5000/5139.849624/5500/6000ms`
- The sparse effective horizon is explicitly reported for T003; nominal horizon is not treated as exact elapsed time.
- Exact final recommendation: `sample_contract_ready_for_signal_acceptance`.
- T003 may be created only after T002 QA acceptance and controller review.

verify：
- `python -m pytest examples/hyperliquid/test_cross_exchange_sample_expansion.py -q` -> `1 passed`
- Required combined suite -> `17 passed`, `1 failed`; the failure is the pre-existing accepted-sample integration test requiring missing local artifact `local_live_analysis/cross_exchange_public_sample_0602T001`.
- Same combined suite with only that unavailable historical-artifact test deselected -> `17 passed, 1 deselected`.
- `python -m py_compile examples/hyperliquid/cross_exchange_sample_expansion.py examples/hyperliquid/test_cross_exchange_sample_expansion.py` -> passed.
- `python examples/hyperliquid/synchronized_public_collection.py --help` -> passed.
- `python examples/hyperliquid/cross_exchange_sample_expansion.py --help` -> passed.
- All T002 JSON manifests parsed with `python -m json.tool`.
- All six copied raw files independently matched their `raw.sha256`.
- Required CSV schemas, non-empty files, row thresholds, regime count, boundary flags and recommendation enum -> passed.
- Independent rerun to `/tmp/0625T002-repro.n48JRO` -> all seven deterministic artifacts byte-identical; manifest identical after normalizing generated time/output paths.
- `git diff --check` -> passed.

done：
- Three new windows satisfy raw/provenance/checksum, duration/overlap/start separation, two-regime minimum, alignment/join and complete-context acceptance.
- The task package is committed under `local_live_analysis/cross_exchange_mvp_sample_expansion_0625T002/`.
- T002 did not freeze a signal/side contract, change strategy or watcher behavior, use credentials/private/account/order/cancel endpoints, initialize a trading client, place orders, change edge/quote/cap policy, or authorize canary/promotion.
- Business execution is complete and ready for QA.

blockers：
- No T002 execution blocker.
- One neighboring integration test cannot run because its separately accepted historical local sample `cross_exchange_public_sample_0602T001` is absent on this machine; task-scoped and remaining neighboring tests pass.

commit：
- cee2a63

提交信息：
- 0625 complete synchronized sample expansion
```
