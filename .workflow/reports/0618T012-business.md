```md
执行线程：
- 业务线程-research

任务ID：
- 0618T012

状态：
- 待验收

是否进行QA验收：
- 是

QA说明：
- 无

files：
- `.workflow/tasks/0618T012.md`
- `examples/hyperliquid/hyperliquid_tiny_live_m2_public_flow_diagnosis.py`
- `examples/hyperliquid/test_hyperliquid_tiny_live_m2_public_flow_diagnosis.py`
- `local_live_analysis/hyperliquid_tiny_live_m2_public_flow_diagnosis_0618T012/**`

action：
- Implemented a read-only Hyperliquid public L2/trades flow diagnosis runner.
- The runner can collect fresh public `l2Book` / `trades` data or analyze an existing public raw sample.
- It creates passive buy-at-bid and sell-at-ask touch-quote candidates under the existing M2 order-size proxy, then measures same-side top depth, touch/strict trade-through flow, public queue-depletion proxy, quote aging / BBO drift, and side-level opportunity asymmetry.
- Local direct Hyperliquid public collection failed with repeated SSL EOF before any subscription messages, so it was not used for diagnosis.
- Ran the public-only collector on `awsserver1` without git refresh, credential read, private/account/order endpoint calls, live orders, or strategy process. The remote sample was pulled back and analyzed locally.

verify：
- `python -m pytest examples/hyperliquid/test_hyperliquid_tiny_live_m2_public_flow_diagnosis.py examples/hyperliquid/test_hyperliquid_public_sample.py -q` -> `5 passed`
- `python -m py_compile examples/hyperliquid/hyperliquid_tiny_live_m2_public_flow_diagnosis.py`
- `python examples/hyperliquid/hyperliquid_tiny_live_m2_public_flow_diagnosis.py --help`
- Local fresh public collection attempt: produced no messages because Hyperliquid mainnet public REST/WebSocket returned SSL EOF from this machine.
- Remote public-only collection on `awsserver1`: `l2Book=56`, `trades=122`, `subscription_ack=2`, `reconnects=0`, `close_reason=duration_elapsed`.
- `python examples/hyperliquid/hyperliquid_tiny_live_m2_public_flow_diagnosis.py --raw-input local_live_analysis/hyperliquid_tiny_live_m2_public_flow_diagnosis_0618T012/remote_public_collection/raw.gz --output-dir local_live_analysis/hyperliquid_tiny_live_m2_public_flow_diagnosis_0618T012 --quote-hold-seconds 45 --candidate-stride-seconds 5`
- JSON/CSV validation passed for manifest and output matrices.
- Artifact non-empty check passed: `14` files / `0` empty.
- Boundary scan passed on diagnosis outputs for credential/private-key/signature/nonce/wallet-address values. Raw public trade payload is kept local/ignored and may contain exchange-provided public `users` fields.
- `git diff --check` passed.

done：
- Artifact path: `local_live_analysis/hyperliquid_tiny_live_m2_public_flow_diagnosis_0618T012/`.
- Manifest: `final_recommendation=m2_public_flow_diagnosis_ready_for_qa`.
- Public sample parsed into `56` book events, `356` individual trade events, and `38` passive touch-quote candidates.
- Hypothesis matrix:
  - `queue_too_deep=supported`: public top+order depletion proxy reached only `7/38` candidates; median top-depth multiple was `337.004x` for buy and `1910.611x` for sell.
  - `no_trade_through=rejected_for_sample`: strict trade-through appeared in `21/38` candidates and touch trades in `34/38`.
  - `wrong_time_of_day=inconclusive`: sample was only `117.948s` in UTC hour `9`.
  - `wrong_side=supported`: buy had `6/19` public-depletion candidates and `13/19` strict-through candidates, while sell had `1/19` and `8/19`.
  - `quote_aging_or_fast_drift=supported`: adverse lost-touch occurred in `21/38` candidates.
- Interpretation: in this public sample, the main blockers are queue depth plus quote aging/fast drift, with sell-side materially worse than buy. The sample rejects "no trade-through" as the primary explanation for this window.
- M2 remains blocked because this is public-flow proxy only; no live maker fill, fee/inventory proof, or realized PnL proof exists.

blockers：
- Local direct Hyperliquid public API/WebSocket access from this machine failed with SSL EOF, so fresh local collection was not usable.
- Time-of-day cannot be decided from one short public window.
- Exact queue position, private order lifecycle, and real fill/PnL remain unproven.

commit：
- e55ee12

提交信息：
- 0618 add M2 public flow diagnosis
```
