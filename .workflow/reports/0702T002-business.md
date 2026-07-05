# 0702T002 Business Report

## Status

- 任务状态: `待验收`
- 业务线程: `业务线程-collector-fix`
- 结论: 已修复 Binance REST depth snapshot 失败后仍产出可后处理样本的问题。
- Commit: 无

## Root Cause

- `0702T001` 三段样本的 `binance_public_raw/depth_snapshot.json` 均为 HTTP `429`，body 为 Binance 对 `awsserver1` 公网 IP `18.182.23.227` 的限流响应。
- Binance 文案里的 `2400 requests per minute` 是该 IP 的限额说明，不是本进程已经发出 `2400/min` snapshot 请求的直接证据。
- 当前采集器主路径理论上一段只取一次 snapshot；但代码存在实际缺陷：
  - `fetch_binance_depth_snapshot()` 对 HTTP 429 返回 `status=http_error` 后，采集流程仍把响应 body 写入 raw，并让采集子进程返回成功。
  - 后续 `binance_top5_provenance.py` 没有 `lastUpdateId`，只能输出 `snapshot_alignment_status=missing`，导致所有 Binance top5 context 为空。
  - 默认 `--snapshot-limit=1000` 对 Binance USD-M depth endpoint 是不必要的高权重请求；当前 top5 bootstrap 只需要 top5 以上的初始 book。
  - 没有低频、可观测的 retry/backoff，也没有“没有有效 snapshot 就 fail fast”的硬门禁。

## Implementation

- `examples/hyperliquid/synchronized_public_collection.py`
  - 新增默认参数：
    - `DEFAULT_BINANCE_DEPTH_SNAPSHOT_LIMIT = 100`
    - `DEFAULT_BINANCE_DEPTH_SNAPSHOT_RETRY_ATTEMPTS = 6`
    - `DEFAULT_BINANCE_DEPTH_SNAPSHOT_RETRY_BASE_DELAY = 5.0`
    - `DEFAULT_BINANCE_DEPTH_SNAPSHOT_RETRY_MAX_DELAY = 60.0`
  - `fetch_binance_depth_snapshot()` 现在捕获 request/JSON 异常，记录 `http_status`、`rate_limited`、`retry_after_seconds`，并将 HTTP `418/429` 标为 `rate_limited`。
  - 新增 `fetch_binance_depth_snapshot_with_retries()`，按 `Retry-After` 或指数退避低频重试。
  - 新增有效 snapshot 判定：必须有 `status=ok`、`lastUpdateId`、非空 `bids` 和非空 `asks`。
  - `collect_binance_public_sample()` 在 snapshot 无效时写出 manifest 证据后抛出 `RuntimeError`，采集子进程返回非 0，避免产生 Binance top5 必空的“成功样本”。
  - collection manifest 新增：
    - `depth_snapshot_http_status`
    - `depth_snapshot_attempt_count`
    - `depth_snapshot_rate_limited_attempt_count`
    - `depth_snapshot_valid`
    - `depth_snapshot_required`
    - `snapshot_limit`
  - `collect` 和 `collect-binance-public` CLI 均暴露 snapshot limit/retry 参数。

- `examples/hyperliquid/test_synchronized_public_collection.py`
  - 更新成功采集用例以覆盖 retry wrapper。
  - 新增 429 rate-limit 退避测试，确认不会 tight loop 请求 snapshot。
  - 新增无有效 snapshot 时采集必须失败、manifest 必须记录限流证据的回归测试。

## Verification

- `python -m pytest examples/hyperliquid/test_synchronized_public_collection.py -q`
  - Result: `9 passed`
- `python -m py_compile examples/hyperliquid/synchronized_public_collection.py examples/hyperliquid/test_synchronized_public_collection.py`
  - Result: passed
- `python examples/hyperliquid/synchronized_public_collection.py collect-binance-public --help`
  - Result: expected snapshot retry options present.
- `python examples/hyperliquid/synchronized_public_collection.py collect --help`
  - Result: expected Binance snapshot retry options present.
- `git diff --check -- examples/hyperliquid/synchronized_public_collection.py examples/hyperliquid/test_synchronized_public_collection.py`
  - Result: passed.

## Boundaries

- 未修改 `0702T001` 已采集 raw 数据，未补造 `depth_snapshot.json`，未补造 Binance top5。
- 未修改 alignment、join、lead-lag analysis、pricing signal、sample expansion 的验收口径。
- 未触及 live strategy、private/account/order/cancel endpoint、credential、下单、canary 或 promotion。

## Remaining Risk

- 如果 `awsserver1` 公网 IP 已被其他进程或其他用户共享打满 Binance REST 限额，本修复会低频重试并 fail fast，但不能保证一定拿到 snapshot。
- 后续定时采集如果仍遇到连续 429，任务会明确失败在采集阶段；这比生成 top5 全空样本更安全。
