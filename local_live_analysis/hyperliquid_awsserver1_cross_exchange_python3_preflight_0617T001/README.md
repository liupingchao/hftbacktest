# 0617T001 awsserver1 Cross-Exchange Python3 Preflight

Status: ready for QA.

This task created a separate remote checkout for the current local
`cross-exchange` branch:

- path: `/home/admin/hftbacktest-cross-exchange`
- branch: `cross-exchange`
- commit: `7642b16`
- dirty count: `0`
- selected remote Python: `/usr/bin/python3`
- Python version: `Python 3.13.5`

The existing Binance maker route at `/home/admin/hft_live/hftbacktest` was not
modified. It remains:

- branch: `master`
- commit: `703c149`
- dirty count: `29`

No private endpoint, credential read, account query, order placement,
cancellation, amendment, or live bot startup was performed.
