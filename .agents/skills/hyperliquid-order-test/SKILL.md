---
name: hyperliquid-order-test
description: Prepare, execute, reconcile, or diagnose a controlled Hyperliquid private API or real-order test on c6in using the canonical trading runtime. Use for runtime discovery, API-wallet and unified-master resolution, account/open-order/position preflight, post-only Alo canaries, cancel-latency measurements, exact oid/cloid terminal proof, fill flattening, or investigation of empty history and zero-margin results. Do not use for strategy deployment, continuous trading, or live actions outside an explicitly authorized and frozen task scope.
---

# Hyperliquid Order Test

Use the repository's reviewed runners and the official SDK. Do not hand-roll
signing, nonce handling, or `/exchange` payloads inside the Skill.

## Select The Mode

1. `discovery`: inspect paths, permissions and SDK methods only. No wallet
   construction or endpoint calls.
2. `private-preflight`: resolve account identity and query account, market,
   order and position state. No order or cancel call.
3. `active-test`: submit the frozen canary or measurement order, reconcile it,
   and flatten a fill when the task authorizes flattening.

Active mode requires explicit authorization and frozen limits. Once the user
has authorized the current task, do not repeatedly ask for the same approval;
do not extend that approval to another symbol, account, route, cap or task.

## Workflow

1. Read `AGENTS.md`, the formal task and its reviewed execution plan.
2. Run the canonical discovery command before claiming anything is missing:

   ```bash
   ssh c6in-winner '/home/admin/trading/inspect --json' \
     > /tmp/hyperliquid-runtime.json
   python .agents/skills/hyperliquid-order-test/scripts/validate_runtime_manifest.py \
     /tmp/hyperliquid-runtime.json
   ```

3. For private or active work, create a clean task checkout at the exact
   reviewed commit. Validate its explicit repo, credential and copied-venv
   paths with `--require-clean-repo`. Never execute from the dirty shared
   checkout merely because discovery succeeded.
4. Freeze the task contract before private access: host, account environment,
   network, DEX, asset, side policy, `Alo`, attempts, order and aggregate caps,
   quote-distance rule, timeout/polling rules, loss basis, stop conditions and
   whether reduce-only flatten is authorized.
5. Resolve configured address, signer and unified master before interpreting
   history, margin, orders or positions. Read
   `references/hyperliquid-order-lifecycle.md` for the mandatory identity
   decision tree.
6. Complete the private baseline on the resolved master: conflicting runtime,
   market metadata, collateral source, exact target-DEX open orders and exact
   target position. Fail closed on malformed, missing or contradictory state.
7. Use one deterministic cloid per attempt and one live order at a time.
   Confirm exact resting state before cancel. Treat submit and cancel responses
   as transport acknowledgements, not authoritative order state.
8. Poll exact oid plus cloid for the terminal. Use cancel-by-cloid as the
   bounded rescue path. Before the next order, prove zero target open orders
   and zero target position with bounded polling.
9. On any fill, stop new attempts, reconcile fills and position, run the
   authorized reduce-only flatten, calculate the frozen loss basis, and prove
   the final zero-order/zero-position state.
10. Write only redacted evidence and report the exact execution commit,
    runtime identity, frozen caps, endpoint classes called, terminal counts,
    fills, unresolved exposure and final reconciliation.

Read `references/c6in-runtime.md` when preparing or diagnosing the remote
runtime. Read `references/hyperliquid-order-lifecycle.md` before any private or
active mode.

## Hard Boundaries

- Never print, hash into reports, copy, commit or archive credential values.
- Never use shell tracing while loading credentials.
- Never query an API-wallet agent as though it were the trading account.
- Never infer success from an accepted submit, successful cancel response,
  empty single snapshot, SDK importability or `lookup_ready=true`.
- Never start attempt N+1 before attempt N has an authoritative terminal and
  final safety reconciliation.
- Never describe a canary or latency test as strategy profitability, production
  readiness or authorization for continuous trading.
