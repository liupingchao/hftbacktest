# c6in Runtime Contract

## Canonical Paths

```text
repo:        /home/admin/trading/repo
credentials: /home/admin/trading/credentials.env
env alias:   /home/admin/trading/.env
venv:        /home/admin/trading/venv
python:      /home/admin/trading/python
inspect:     /home/admin/trading/inspect
```

Run:

```bash
ssh c6in-winner '/home/admin/trading/inspect --json'
```

The aliases may resolve into `/srv/crypto-bot/research/...`. Do not replace
this command with a plain `find /home/admin`, and do not assume the SDK venv
name contains `hyperliquid`.

Expected permissions:

- `/home/admin/trading`: `700`
- resolved credential source: `600`
- `runtime-manifest.json`: `600`

The credential aliases point to the source of truth. Do not copy or modify
`.env`.

## Manifest Admission

For discovery-only work, require:

- schema `trading_runtime_discovery_v1`
- `lookup_ready=true`
- private credential permissions
- Hyperliquid credential group ready
- complete unified-account/order/cancel/query/flatten SDK surface
- every no-action boundary flag equal to its frozen value

Use:

```bash
python .agents/skills/hyperliquid-order-test/scripts/validate_runtime_manifest.py \
  /tmp/hyperliquid-runtime.json
```

For private or active work, also require a clean exact task checkout:

```bash
python .agents/skills/hyperliquid-order-test/scripts/validate_runtime_manifest.py \
  --require-clean-repo /tmp/hyperliquid-task-runtime.json
```

`lookup_ready=true` is location and interface evidence only.
`execution_runtime_ready=true` additionally proves the selected checkout is
clean; it does not prove account safety or authorize endpoints.

## Clean Task Runtime

1. Fetch the reviewed source commit without modifying the shared checkout.
2. Create a detached clean task checkout at that exact commit.
3. Build or copy the task venv with `python3 -m venv --copies` when exact
   interpreter identity matters. A symlinked venv Python can resolve to the
   system interpreter and hide installed SDK metadata.
4. Run discovery with explicit task checkout, credential and Python overrides.
5. Pass `/home/admin/trading/credentials.env` explicitly to the reviewed
   runner. Do not source it with tracing enabled.
6. Record commit, Python executable identity, SDK version and method surface
   without recording account addresses or credential values.

Required Exchange methods:

```text
order
cancel
cancel_by_cloid
schedule_cancel
market_close
```

Required Info methods:

```text
open_orders
user_state
spot_user_state
user_fills
user_fills_by_time
query_order_by_oid
query_order_by_cloid
user_role
query_user_abstraction_state
extra_agents
```

## Common False Diagnoses

- Dirty shared checkout: discovery can pass while execution readiness fails.
- Agent queried as account: history and margin can appear empty or zero.
- Symlinked venv Python: the selected path can resolve to a system interpreter
  without the expected SDK metadata.
- SDK import succeeds: this does not prove the required methods, identity,
  collateral, market, private endpoint or order path.
