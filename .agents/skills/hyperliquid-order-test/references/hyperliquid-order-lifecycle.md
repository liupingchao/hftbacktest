# Hyperliquid Order Lifecycle Contract

## 1. Resolve The Unified Account

Normalize the configured address and the private-key signer address, then query
`Info.user_role` for both.

1. If the configured role is `agent`, it must equal the signer. Derive the
   master from `role.data.user`.
2. If the configured role is `user`, use it as the master.
3. Reject missing, malformed or unexpected roles.
4. Require the derived master role to be `user`.
5. Require `query_user_abstraction_state(master) == "unifiedAccount"`.
6. When signer differs from master, require:
   - signer role `agent`
   - signer `role.data.user` equals master
   - an exact signer entry in `extra_agents(master)`
   - `validUntil` present and later than current time

Use the master for `Info` queries. Construct `Exchange` with the agent wallet
for signing and `account_address=master`. Store only stable one-way identity
tokens when evidence needs identity continuity; never store raw addresses.

For a unified account, collateral admission must consider both the target DEX
state and available spot USDC (`total - hold`). Do not interpret an agent
address's empty history or zero margin as an underfunded master account.

## 2. Freeze The Active Contract

Before the first private read, freeze:

- mainnet/testnet and target DEX
- exact SDK asset name
- side selection and maximum attempts
- per-order notional, aggregate position and realized-loss caps
- `Alo` post-only requirement
- tick, lot, minimum notional and quote-distance admission
- resting, terminal and final-reconciliation timeouts
- fill stop and reduce-only flatten behavior
- artifact and redaction contract

Authorization persists only for this frozen scope. Any expansion requires a
new reviewed revision.

## 3. Private Baseline

Fail closed unless all facts are authoritative:

- no conflicting live runtime on the same account/market path
- target asset metadata and precision are present
- quote distance is market-safe and non-crossing
- target-DEX open-order response is a list
- exact target position is zero or within the frozen starting-state contract
- collateral meets the aggregate cap
- any pre-existing order or position is explicitly owned and handled

HIP-3 assets require the exact DEX and SDK asset identity on every query and
action.

## 4. Submit And Rest

Generate a deterministic 128-bit cloid from task, attempt and side. Submit one
post-only `Alo` order with the frozen size and price.

Classify the response as rejected, filled or resting. An accepted/resting
response must include an oid, but that response alone is not resting proof.

Poll `query_order_by_oid(master, oid)` and require the embedded oid and cloid
to match. If exact order status is temporarily unavailable, a target-DEX
`open_orders` fallback is admissible only when exactly one row matches both oid
and cloid. Reject duplicate or partial matches.

If resting cannot be confirmed within the frozen timeout, cancel by cloid,
prove the terminal, reconcile safety and stop.

## 5. Cancel And Prove The Terminal

1. Start exact-reference terminal observation.
2. Call `cancel(asset, oid)`.
3. Treat the cancel response as acknowledgement only.
4. Poll the exact order reference until `cancel_confirmed`, `filled` or
   `rejected`.
5. If no terminal arrives, call `cancel_by_cloid` once as the bounded rescue
   and poll again.
6. Unknown, contradictory or mismatched state is unresolved exposure and a
   fatal stop.

After the exact terminal, poll private state for up to the frozen deadline
(the accepted 0822T002 contract used 5 seconds at 50 ms) until target open
orders are empty and target position is zero. A single immediate snapshot may
still show a canceled HIP-3 order and is not final proof.

## 6. Fill And Flatten

Any fill stops new attempts. Reconcile fills and target position from the
master account. If the task authorizes flattening, submit the reviewed
reduce-only or `market_close` action with its own deterministic cloid.

Require the flatten fill quantity to match the exposure and prove the final
position is zero. Calculate the task's frozen loss basis only after
authoritative flatten completion. Do not substitute mark-to-market loss when
the contract defines realized flatten slippage.

Emergency cleanup still requires a final open-order and position proof. An
attempted cancel or flatten is not evidence of success.

## 7. Evidence

Record:

- task and exact source commit
- host/boot/process/runtime identity tokens
- SDK version and admitted method surface
- frozen non-secret limits
- configured/signer/master roles and approval booleans
- redacted order-reference tokens
- submit, resting, cancel-response and authoritative-terminal classes
- fill and flatten facts
- final open-order count, final position and unresolved exposure

Never record credential values, raw private responses, raw addresses,
signatures, nonces, oid or cloid.

The failure modes above come from the independently accepted `0822T002`
c6in Hyperliquid latency run: wrong agent/master binding, a symlinked venv
interpreter, delayed HIP-3 resting visibility and stale post-cancel snapshots.
