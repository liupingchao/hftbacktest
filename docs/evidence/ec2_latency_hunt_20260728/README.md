# EC2 Latency Hunt Evidence

Task `0728T072` repaired the T071 clock and cleanup defects and ran one
content-addressed Linux x86_64 probe on 10 Tokyo `xlarge` candidates. Each
candidate used an independent cluster placement group, passed a complete
Chrony error-bound gate before and after capture, discarded 100 warmup
messages and captured both venues concurrently for 900 seconds.

## Result

- Run ID: `0728T072-20260728T090150Z`
- Linux binary SHA-256:
  `ba2fbaa4bfcca1f34dfb7f04a3b5de0f6250be328d1bfcf25d3b66a4b55c3403`
- Candidates: `10`
- Feed eligible: `10`
- Local pipeline eligible: `10`
- Complete clock bound before/after: `79.903-596.035us`
- Remote raw-to-summary rebuilds: `20/20`
- Independent local raw-to-summary rebuilds: `20/20`
- Archive SHA matches: `10/10`
- Active T072 instances after cleanup: `0`
- Remaining T072 placement groups after cleanup: `0`
- Temporary artifact bucket: deleted

## T074 Control-Plane Closure

Task `0728T074` keeps the accepted T072 10-host data and closes the remaining
control-plane defects found by T073 independent QA.

- Deterministic source archive:
  `build_source.tar.gz`
- Source archive SHA-256:
  `9b448775b6babad916d782349b0edf607477fe7fc86dff4586618017e733b991`
- Rebuilt Linux binary SHA-256:
  `ba2fbaa4bfcca1f34dfb7f04a3b5de0f6250be328d1bfcf25d3b66a4b55c3403`
- Rebuilt binary matches the accepted T072 full-run binary: yes
- Final control Canary:
  `0728T074-20260728T102927Z`
- Canary hosts: `c7i.xlarge`, `m7i.xlarge`
- Setup clock receipts: `2/2`
- Parsed setup receipt gates: `2/2`
- Run commands: `2/2`
- Upload receipt / S3 size / downloaded SHA: `2/2`
- Raw-to-summary rebuilds: `4/4`
- Cleanup journal: `verified=true`, no errors, empty post-state
- Independent post-cleanup AWS query: zero instances, zero placement groups,
  temporary bucket absent

The final Canary retained complete setup clock JSON in SSM stdout and the
orchestrator parsed it before capture. Only an explicit S3 404 is accepted as
absence; 403 and other query failures fail cleanup. Executable upload failure,
S3 403, signal/finally, setup gate/journal and create-before-record paths are
covered by `18` focused tests. Structured details are in
`control_canary.json`.

Hyperliquid uses strategy-path-compatible
`{"type":"l2Book","coin":"BTC","fast":true}` and derives BBO from the first
level on each side. Every host captured `1653` accepted fast-L2 snapshots,
about `110.2` per minute. This is materially denser than ordinary l2Book, but
message count alone is not compared directly with the separate `bbo` channel.

Raw compressed evidence is retained under
`local_live_analysis/ec2_latency_hunt_0728T072-20260728T090150Z/`.
`hunt_results.json` contains every archive hash, host row, clock boundary and
metric ranking. `execution_manifest.json` binds the successful run and all
three fail-closed/cleanup attempts.

## Main Ranking

All local pipeline values below are P99. Feed P50 ranks are retained in JSON
but close adjacent ranks are not treated as decisive because each row carries
up to its reported complete clock bound.

| Instance | Binance feed P50 | Binance feed P99 | Binance local | HL fast-L2 local | Clock max |
| --- | ---: | ---: | ---: | ---: | ---: |
| `c6in.xlarge` | `2.231ms` | `7.857ms` | `22.110us` | `29.151us` | `493.036us` |
| `m7i.xlarge` | `2.367ms` | `4.679ms` | `41.524us` | `51.574us` | `121.016us` |
| `c7a.xlarge` | `2.158ms` | `9.236ms` | `27.500us` | `37.548us` | `92.562us` |
| `m7a.xlarge` | `2.227ms` | `9.757ms` | `28.220us` | `34.945us` | `153.486us` |
| `m5zn.xlarge` | `2.393ms` | `101.820ms` | `17.415us` | `21.559us` | `596.035us` |

`c6in.xlarge` remains the balanced first winner: Binance feed P99 rank 2,
Binance local P99 rank 2 and Hyperliquid local P99 rank 2. `m7i.xlarge` is the
network-tail winner with Binance feed P99 rank 1. `m5zn.xlarge` is the local
pipeline winner but is rejected as the first deployment choice because its
Binance feed P99 is worst in the matrix.

Hyperliquid `feed_network` remains excluded from instance ranking because its
exchange timestamp is not a pure wire-arrival timestamp under the current
parser contract. Its monotonic fast-L2 local pipeline remains valid.

## Files

- `hunt_results.json`: validation rows, archive hashes and metric rankings.
- `hunt_results.csv`: compact metric table.
- `execution_manifest.json`: source hashes, run state hashes, failed attempts
  and cleanup proof.
- `build_receipt.json`: Linux build source and binary identity.
- `build_source.tar.gz`: deterministic exact Linux build source archive.
- `control_canary.json`: final T074 upload/object/clock/cleanup closure.
