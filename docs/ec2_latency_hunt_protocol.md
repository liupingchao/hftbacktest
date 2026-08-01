# EC2 Latency Hunt Protocol

## Reference Review

Before launching Goal 2, the local AWS sample project at
`/Users/liu/Documents/trading-latency-benchmark` was reviewed at commit
`8dad243`.

The reusable parts are:

- one exact probe/configuration across all candidates;
- independent cluster placement groups to broaden network placement;
- parallel execution with explicit per-instance failures;
- warmup before accepted samples;
- P50/P90/P99/P99.9 rather than mean-only comparison;
- immutable or identical host setup and content-addressed artifacts;
- Chrony/PTP state as an acceptance gate;
- host, ENA, IRQ, offload and CPU topology metadata;
- automatic artifact collection and resource cleanup.

The sample's mock-server RTT, Java/HdrHistogram protocol, diverse architecture
binary builds, CDK/Lambda control plane and full Amazon Linux OS tuning are not
copied into this task. They measure a different path or change too many
variables for the first public-exchange baseline. In particular, the sample
tuning disables SSM, while this task uses SSM as the recovery and cleanup
channel.

## Candidate Matrix

All candidates are x86_64 `xlarge`, in `ap-northeast-1c`, and use the same
Debian AMI, subnet, security group, instance profile and Linux probe binary.
Each candidate has its own cluster placement group.

1. `c7i.xlarge`
2. `c7a.xlarge`
3. `c6in.xlarge`
4. `c5n.xlarge`
5. `m7i.xlarge`
6. `m7a.xlarge`
7. `r7i.xlarge`
8. `r7a.xlarge`
9. `m5zn.xlarge`
10. `c6i.xlarge`

The matrix keeps one architecture and one 4-vCPU size class while varying CPU
generation, clock, cache/memory profile and ENA/network capability. Capacity
failures are recorded and backfilled from `m6i.xlarge`, `m6a.xlarge` and
`c6a.xlarge`.

T072 additionally treats a failure of the frozen complete clock bound
as an explicit candidate rejection. The failed family remains in the run
journal and is replaced from the same backup list; the threshold is not
relaxed after observing a candidate.

## Clock Gate

Each candidate uses Amazon Time Sync Service through Chrony:

```text
server 169.254.169.123 prefer iburst minpoll 4 maxpoll 4
```

Before measurement:

- Chrony must be active.
- Leap status must be `Normal`.
- Absolute `System time` offset must be at most `100us`.
- Root delay and root dispersion are persisted.
- PHC device, ENA driver/version, hardware timestamp capability and
  `phc_error_bound` are persisted when available.

The accepted clock bound is:

```text
abs(system_time) + 0.5 * root_delay + root_dispersion + phc_error_bound
```

For NTP-only hosts the PHC term is absent. A host that fails the gate may still
provide local monotonic pipeline evidence, but its `feed_network` values cannot
select the winner.

T072 requires this complete bound to be at most `750us` both immediately
before and immediately after the accepted measurement window. Both
`chronyc tracking` samples carry distinct UTC capture timestamps and are
included in the result archive. A setup-time snapshot cannot substitute for
either window-boundary sample.

The first two repair attempts used `500us` and failed closed at stable bounds
near `544us` on different instance families. Before the third attempt, the
threshold was re-frozen at `750us`, still below `1ms` and orders of magnitude
tighter than the invalid multi-second T071 evidence. Feed P50 results must be
reported with their clock bound; close adjacent ranks are not treated as
decisive.

The tracked `ec2_hunt_orchestrator.py` writes every bucket, placement group,
instance, SSM command and failure event to an atomically replaced state file.
Its `finally` cleanup unions state-file resources with task/run tag discovery,
so launch failures and local interruption use the same re-entrant cleanup path.

## Exact Run

Every candidate runs the same content-addressed binary and commands:

- recorder benchmark: 1,000,000 traces, P99 budget `5us`;
- Binance Futures `BTCUSDT`: 100 warmup BBOs, then 900 seconds;
- Hyperliquid `BTC`: fast `l2Book` with `fast=true`, derive BBO from the first
  level on each side, discard 100 warmup snapshots, then run 900 seconds;
- both venue probes run concurrently;
- raw NDJSON is independently summarized and byte-compared on the host;
- compressed raw artifacts are pulled back before termination.

No credentials, account reads, order, cancel or private endpoint are used.

## Ranking

Eligibility requires:

- both venue commands succeeded;
- non-zero traces and complete required chains;
- zero duplicate, out-of-order and dropped traces;
- binary hash match;
- raw-to-summary byte equality;
- clock gate pass for any `feed_network` ranking.

Reports retain separate rankings for:

- Binance `feed_network` P50/P99;
- Hyperliquid `feed_network` P50/P99;
- Binance `tick_to_wire` P50/P99;
- Hyperliquid `tick_to_wire` P50/P99;
- recorder benchmark P50/P99.

No single opaque score can hide a failed metric. The recommendation must name
the metrics and tradeoffs supporting it.

## Same-Type Retained-Host Hunt

Task `0729T001` is a second-stage placement hunt within the accepted
`c6in.xlarge` family. It changes the resource lifecycle and placement strategy:

- one regional rack-level spread placement group;
- exactly seven same-AZ `c6in.xlarge` candidates, the regional per-AZ maximum
  for one spread group;
- two canaries before launching the remaining five;
- the same AMI, subnet, security group, instance profile, binary, venue,
  symbol, duration and warmup on every candidate;
- `awsserver1` remains running and outside task/run tag cleanup scope;
- failure cleans all hunting candidates and temporary resources;
- success retains the exact measured winner, enables API termination
  protection and terminates six losers.

Host-level spread is not available in an AWS Region and is limited to
Outposts. Regional same-type hunting therefore uses `spread-level=rack`.

The winner formula is frozen before launch:

```text
4 * Binance feed P99 rank
+ 2 * Binance feed P50 rank
+ 2 * Binance tick-to-wire P99 rank
+ 1 * Hyperliquid tick-to-wire P99 rank
+ 1 * recorder benchmark P99 rank
```

All component rankings remain visible. The balanced winner and each single
metric winner are reported separately; the weighted score cannot turn a poor
single metric into a claim that the retained instance is best on that metric.

Successful final-state acceptance requires:

- all seven candidates eligible;
- fourteen remote and fourteen local raw-summary comparisons matched;
- one task-tagged winner running in the original spread group;
- winner termination protection enabled;
- six task-tagged losers terminated;
- temporary bucket explicitly absent;
- original `awsserver1` instance ID still running;
- exactly those two servers running in the account/region snapshot.
