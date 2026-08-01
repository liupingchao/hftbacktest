# C6in Rack-Spread Host Hunt Evidence

## Run Identity

- Task: `0729T001`
- Run: `0729T001-20260729T005823Z`
- Region / AZ: `ap-northeast-1 / ap-northeast-1c`
- Candidate type: `c6in.xlarge`
- Candidate count: `7`
- Placement group:
  `hft-001-20260729T005823Z-c6in-spread`
- Placement strategy / level: `spread / rack`
- Full window: `900s`, `100` warmup messages
- Probe binary SHA256:
  `ba2fbaa4bfcca1f34dfb7f04a3b5de0f6250be328d1bfcf25d3b66a4b55c3403`

AWS Regions do not support host-level spread groups; host level is limited to
Outposts. This run therefore uses the strongest supported regional constraint:
one rack-level spread group, with the regional maximum of seven running
instances in the same AZ.

## Integrity

- Placement receipt: `7/7`, same type, AZ and rack-spread group.
- Setup clock receipts: `7/7` pass.
- Full commands: `7/7` Success.
- Feed eligible: `7/7`.
- Local-pipeline eligible: `7/7`.
- Remote raw-to-summary rebuilds: `14/14`.
- Independent local raw-to-summary rebuilds: `14/14`.
- Archive SHA/size verification: `7/7`.
- Clock max complete bounds: `412.394-655.988us`, all below `750us`.
- Binance traces: `437,121-439,801` per host.
- Hyperliquid fast-L2 traces: exactly `1,662` per host.

## Result

The frozen balanced score was:

```text
4 * Binance feed P99 rank
+ 2 * Binance feed P50 rank
+ 2 * Binance tick-to-wire P99 rank
+ 1 * Hyperliquid tick-to-wire P99 rank
+ 1 * recorder benchmark P99 rank
```

| Rank | Candidate | Instance | Score | Binance feed P50 | Binance feed P99 | Binance local P99 | Hyperliquid local P99 |
|---:|---|---|---:|---:|---:|---:|---:|
| 1 | candidate-07 | `i-0a962e47210528526` | 31 | 2.077ms | 12.767ms | 15.441us | 31.594us |
| 2 | candidate-05 | `i-0e1ca020aedc8e712` | 32 | 2.191ms | 2.951ms | 17.153us | 34.285us |
| 3 | candidate-01 | `i-0df34289f049b7db1` | 33 | 2.121ms | 5.199ms | 18.071us | 30.611us |
| 4 | candidate-02 | `i-0bf6f9798a0bc8cbb` | 37 | 2.190ms | 3.261ms | 16.620us | 31.688us |
| 5 | candidate-03 | `i-09c0cb490d3fa080b` | 37 | 2.124ms | 3.040ms | 18.708us | 31.595us |
| 6 | candidate-04 | `i-070a4825942fefeaf` | 52 | 2.226ms | 12.273ms | 17.175us | 32.207us |
| 7 | candidate-06 | `i-0be6269f6067780ea` | 58 | 2.185ms | 14.220ms | 18.569us | 37.813us |

Balanced winner `candidate-07` is retained because it has the best feed P50,
Binance local P99 and benchmark P99, plus Hyperliquid local P99 rank 2.
It is not the Binance feed-tail winner. `candidate-05` has the best Binance
feed P99 at `2.951ms`, compared with `12.767ms` on the balanced winner.

The cohort demonstrates real placement dispersion despite identical type and
configuration:

- Binance feed P50 range: `2.077-2.226ms` (`1.07x`).
- Binance feed P99 range: `2.951-14.220ms` (`4.82x`).
- Binance local P99 range: `15.441-18.708us` (`1.21x`).
- Hyperliquid local P99 range: `30.611-37.813us` (`1.24x`).
- Recorder benchmark P99 range: `1.363-1.475us` (`1.08x`).

## AWS Final State

- Existing `awsserver1`:
  `i-02c64c088f311cbc1`, `c7i.large`, running and unchanged.
- Retained exact measured winner:
  `i-0a962e47210528526`, `c6in.xlarge`, running.
- Winner Name: `0729T001-c6in-winner`.
- Winner `DisableApiTermination`: `true`.
- Six losing candidates: terminated.
- Temporary S3 bucket: explicit `404 Not Found`.
- Rack-spread placement group: retained because the running winner remains a
  member.
- Final running-server count: exactly `2`.

Do not stop/start the winner before burn-in acceptance. A stop/start can move
the instance to different underlying hardware and would invalidate the reason
for retaining this exact measured instance.

## Files

- `execution_manifest.json`: task/run, source hashes and artifact identity.
- `runtime_control_source.tar.gz`: exact T001 runtime controller, analyzer,
  remote and test source bytes sealed before the T002 offline repair.
- `hunt_results.json`: checks, metrics, rankings and local rebuild receipts.
- `hunt_results.csv`: compact candidate metric table.
- `final_receipt.json`: retained winner, losers and protected baseline receipt.
- `running_instances.json`: independent post-run running-instance snapshot.

Raw archives and extracted NDJSON remain untracked under
`local_live_analysis/ec2_latency_hunt_0729T001-20260729T005823Z/`.

The execution manifest keeps two distinct source identities:

- `runtime_source_hashes`: code present during the actual seven-host run;
- `post_qa_repair.source_hashes`: T002 offline finalizer/test repairs.

T002 did not mutate AWS resources or change runtime evidence.
