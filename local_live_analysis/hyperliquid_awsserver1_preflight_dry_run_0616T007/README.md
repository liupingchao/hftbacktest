# 0616T007 awsserver1 Preflight Dry-Run

Status: blocked.

This artifact directory contains the pulled-back dry-run evidence from
`awsserver1`. No private endpoint, credential read, account query, order
placement, cancellation, amendment, or live bot startup was performed.

Blocking facts:

- remote repo path: `/home/admin/hft_live/hftbacktest`
- remote branch: `master`
- required branch: `cross-exchange`
- remote dirty file count: `29`
- remote `conda`: not found
- remote `rsync`: not found, so pullback used `scp`

Checksum validation passed locally with:

`cd local_live_analysis/hyperliquid_awsserver1_preflight_dry_run_0616T007 && sha256sum -c sha256sums.txt`
