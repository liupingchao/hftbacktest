# Hyperliquid Tiny-Live 0616T006-T008 Auto Loop

Status: controller runbook

Created: `2026-06-17 11:07 CST`

This document defines the active sequential auto loop for moving from the
`0616T006` operator packet QA gate to one bounded Hyperliquid tiny-live run on
`awsserver1`. It is not a general live authorization.

## Objective

Execute the next three workflow steps with minimal interruption while preserving
the branch boundary:

- Binance remains the lead-side public/read-only pricing context.
- Hyperliquid remains the lag/maker execution venue.
- `0616T008` is the only approved real-order window, and only after `0616T006`
  QA and `0616T007` QA both pass.

## Sequence

1. `0616T006` QA acceptance.
2. If `0616T006` QA is `已通过`, create and execute `0616T007`:
   `awsserver1` live-capable preflight dry-run.
3. If `0616T007` QA is `已通过`, create and execute `0616T008`: Hyperliquid
   tiny-live small-notional execution.
4. Stop after `0616T008` for QA and post-live evidence analysis.

## Approved 0616T008 Live Parameters

Approval date: `2026-06-17`.

BTC/USD reference used for notional caps: `65794.035`.

- symbol: `BTC`
- max order size: `0.01 BTC`
- max order notional: `700 USDC`
- max position: `0.04 BTC`
- max position notional: `2800 USDC`
- max notional: `3000 USDC`
- max loss: `30 USDC`
- duration: `10 minutes`
- host / machine: `awsserver1`
- account scope: Hyperliquid account configured on `awsserver1`
- maker-only / post-only: `true`
- real orders allowed: `true`, only inside `0616T008`

This approval does not authorize capital scaling, strategy default-on behavior,
deployment, promotion, relaxed caps, or any later live window.

## Auto-Continue Conditions

The loop may continue from one step to the next only when all conditions hold:

1. The current task has a business report ending in `待验收`.
2. The current task has a QA report ending in `已通过`.
3. The recommendation matches the pre-approved next step.
4. Task-scoped verification passed under
   `/home/liushuai/workspace/hftbacktest/.conda-envs/hft-py38/bin/python`.
5. `git diff --check` passed.
6. Scope did not expand beyond the task dispatch.
7. No unapproved private endpoint, credential disclosure, signing, nonce,
   user-stream, account query, order placement, cancellation, amendment, live
   bot startup, deployment, promotion, PnL proof, or maker viability proof was
   introduced.

## Stop Conditions

Stop and return to the controller when any condition is true:

- QA is `未通过` or `阻塞`.
- SSH to `awsserver1` is unavailable.
- The remote host is not on branch `cross-exchange`.
- The required conda/Python environment cannot be found or cannot run the
  task-scoped commands.
- Public connectivity or artifact pullback fails during `0616T007`.
- `0616T008` would exceed any approved cap.
- `0616T008` cannot prove maker-only / post-only behavior before placing an
  order.
- Credentials are missing or unsafe to use.
- Shutdown / cancel-all evidence is missing or ambiguous after a live attempt.
- Any cap mismatch, account mismatch, host mismatch, or live window mismatch is
  found.

## 0616T007 Boundary

`0616T007` is a dry-run only task.

Allowed:

- SSH to `awsserver1`.
- Host metadata, repository, branch, conda/Python, clock, disk, process, and log
  path checks.
- Public network reachability checks.
- Remote dry-run artifact directory creation.
- Remote dry-run artifact generation without credentials or private endpoints.
- Artifact archive/checksum.
- Pullback to local machine.
- Local validation.

Forbidden:

- Private endpoint calls.
- Credential reads or disclosure.
- Signing, nonce handling, user-stream implementation.
- Account query.
- Order placement, cancellation, or amendment.
- Live bot startup.
- Deployment, promotion, PnL proof, or maker viability proof.

## 0616T008 Boundary

`0616T008` is the only live-capable execution task in this loop.

Allowed only under approved caps:

- Hyperliquid `BTC` maker-only / post-only tiny live orders.
- Each order size must be at most `0.01 BTC`.
- Total position must be at most `0.04 BTC`.
- Total notional must be at most `3000 USDC`.
- Max loss stop must be `30 USDC`.
- Wall-clock duration must be at most `10 minutes`.
- Artifacts must be generated on `awsserver1`, pulled back locally, and
  validated.

Forbidden:

- Any non-`BTC` symbol.
- Any order larger than `0.01 BTC`.
- Any position larger than `0.04 BTC`.
- Any relaxed cap or extended duration.
- Taker/non-post-only behavior.
- Strategy default-on behavior.
- Deployment, promotion, scaling, or PnL/maker-viability claim.

## Required Artifacts

Each step must produce:

- `.workflow/tasks/<TASK_ID>.md`
- `.workflow/reports/<TASK_ID>-business.md`
- `.workflow/reports/<TASK_ID>-qa.md`
- latest QA copied to `docs/qa-acceptance-report.md`
- task-scoped artifacts under `local_live_analysis/**`
- a commit recording the task/report/artifact changes

`0616T008` must additionally preserve:

- run intent marker
- approved config snapshot
- public market-data manifest
- private/order audit where authorized by the task
- order lifecycle / reject / fill evidence
- kill-switch and cap audit
- shutdown / cancel-all evidence
- archive/checksum manifest
- local pullback validation result

## Current Start Point

Start with `0616T006` QA. Do not create or execute `0616T007` unless
`0616T006` QA is `已通过`. Do not create or execute `0616T008` unless
`0616T007` QA is `已通过`.
