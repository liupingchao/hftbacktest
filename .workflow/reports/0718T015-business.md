# 线程回报

执行线程：
- 总控 auto-loop / 业务实现线程

任务ID：
- 0718T015

状态：
- 待验收

是否进行QA验收：
- 是

QA说明：
- 无

files：
- `examples/hyperliquid/cross_exchange_shared_signal_kernel.py`
- `examples/hyperliquid/test_cross_exchange_shared_signal_kernel.py`
- `examples/hyperliquid/cross_exchange_production_shadow.py`
- `examples/hyperliquid/cross_exchange_public_replay_alignment.py`
- `examples/hyperliquid/test_cross_exchange_production_shadow.py`
- `examples/hyperliquid/test_cross_exchange_public_replay_alignment.py`
- `docs/cross_exchange_price_taxonomy_contract.md`
- `.workflow/tasks/0718T015.md`
- `progress.md`
- `findings.md`

action：
- 新增不可变、可验证、可序列化的 `PricingConfigV1`，并对 normalization stats、config payload 和 decision rows 生成稳定 hash。
- kernel 现在输出 `hl_mid_px`、可验证的 `hl_micro_px`、`signal_score`、`forecast_mid_px`、`reservation_px`、`quote_bid_px`、`quote_ask_px` 与双边 `quote_intents`。
- 将 signal threshold 改为 `confidence_bucket` 审计分类；低于 threshold 的有效输入不再因为 alpha 不足而 `action=block`。
- 保留 missing feature、stale signal、invalid BBO/tick、incoherent snapshot、normalization hash mismatch 和 post-only failure 的 fail-closed gate。
- shadow 与 public replay caller 统一构造 typed pricing config，并在每个 decision row 和 manifest 中写入/比较 config hash。
- `default_normalization_stats()` 明确限定为 offline fixture helper；fixture 使用单独的 identity-stats source，production/shadow/replay 使用 supplied artifact。
- 新增价格分类合同文档；保留旧 `side`/`quote_px` 字段供只读 markout consumer 兼容，同时要求新 consumer 使用双边字段。

verify：
- `python -m pytest examples/hyperliquid/test_cross_exchange_shared_signal_kernel.py examples/hyperliquid/test_cross_exchange_public_replay_alignment.py examples/hyperliquid/test_cross_exchange_production_shadow.py -q`
  - `13 passed`
- `python -m pytest examples/hyperliquid/test_cross_exchange_price_math.py -q`
  - `14 passed`
- `python -m py_compile examples/hyperliquid/cross_exchange_shared_signal_kernel.py examples/hyperliquid/cross_exchange_public_replay_alignment.py examples/hyperliquid/cross_exchange_production_shadow.py examples/hyperliquid/test_cross_exchange_shared_signal_kernel.py`
  - pass
- `git diff --check`
  - pass
- 未执行 live、credential、private、order、cancel、network、remote 或 service action。

done：
- Principal Alignment Task 4 的 typed pricing contract、price taxonomy、quote eligibility separation 和 offline shadow/replay hash alignment 已实现，提交后进入 QA。

blockers：
- 无

commit：
- pending implementation commit

提交信息：
- pending implementation commit
