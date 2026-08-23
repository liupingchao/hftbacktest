# Trading Runtime Discovery

## Purpose

`examples/hyperliquid/trading_runtime_discovery.py` provides one predictable,
redacted lookup path for the Binance and Hyperliquid private trading runtime.
It fixes two recurring discovery failures:

- checkout and credential files may be behind symlinks into
  `/srv/crypto-bot/research/...`;
- a usable Hyperliquid SDK venv may have a task-scoped name that does not
  contain `hyperliquid`.

The tool resolves every candidate symlink explicitly. It reports credential
key names and non-empty status only. It never emits credential values, hashes
of values, account addresses, signatures, order identifiers or reusable secret
material.

## Canonical c6in paths

After installation:

```text
/home/admin/trading/repo
/home/admin/trading/credentials.env
/home/admin/trading/.env
/home/admin/trading/venv
/home/admin/trading/python
/home/admin/trading/inspect
```

The aliases point to existing sources of truth. Credential values are not
copied and the runtime directory is mode `700`.

## Fast lookup

```bash
~/trading/inspect
~/trading/inspect --json
cd ~/trading/repo
~/trading/python --version
```

Live-capable commands should receive the credential path explicitly:

```bash
~/trading/python <runner.py> --env-file ~/trading/credentials.env
```

Discovery success means the sources were found and their non-secret surfaces
were checked. It does not authorize private endpoints or orders. A dirty
checkout remains visible as `execution_runtime_ready=false`.

The Hyperliquid method check includes the complete controlled-order path:

- Exchange: `order`, `cancel`, `cancel_by_cloid`, `schedule_cancel`,
  `market_close`
- Info: `open_orders`, `user_state`, `spot_user_state`, `user_fills`,
  `user_fills_by_time`, `query_order_by_oid`, `query_order_by_cloid`,
  `user_role`, `query_user_abstraction_state`, `extra_agents`

For a private preflight or real-order test, use the project Skill at
`.agents/skills/hyperliquid-order-test/SKILL.md`. It adds the required
API-wallet/unified-master identity decision, order terminal proof, final
reconciliation and fill-flattening contract.

## Security

- Keep the source credential file mode `600`.
- Do not print, copy, commit or archive credential values.
- Do not use shell tracing while loading credentials.
- Do not treat SDK importability as account or endpoint validation.
- Run task-specific account, open-order, position and market-safety preflight
  before any private action.
