# Plan C Binance Futures Order Trade Update Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Subscribe Binance futures live connector to both `ACCOUNT_UPDATE` and `ORDER_TRADE_UPDATE` private stream events so live position state and order lifecycle state both have WebSocket ground truth.

**Architecture:** Keep the existing `UserDataStream::process_message` and `OrderManager::update_from_ws` semantics unchanged. Refactor only the private user-stream connection layer so it can build one URL per event type and launch two private WebSocket connections using the same listen key: one for `ACCOUNT_UPDATE`, one for `ORDER_TRADE_UPDATE`. Use short live validation to prove `ORDER_TRADE_UPDATE` arrives and open-order safety no longer drifts.

**Tech Stack:** Rust connector crate, Tokio async tasks, Binance futures private WebSocket endpoint, existing `OrderManager`, existing live tick-mm Python validation scripts, cargo tests.

---

## Context

The 2026-04-28 2H live run `live_btcusdt_1777342116` stopped after about 95 minutes because of open-order state drift, not position mismatch.

Confirmed from `TODO_live_open_order_alignment.md`:

- Binance private `ACCOUNT_UPDATE` stream fix worked for position alignment.
- Position stayed safe:
  - `bad_gt_0.003 = 0`
  - `critical_position_mismatch = 0`
  - `max_position_mismatch = 0.001`
- Stop reason was open-order drift:
  - local position matched REST position
  - local open orders and REST open orders diverged

Root cause from current code:

- `connector/src/binancefutures/mod.rs` currently builds a private stream URL fixed to `events=ACCOUNT_UPDATE`.
- `connector/src/binancefutures/msg/stream.rs` already defines `EventStream::OrderTradeUpdate`.
- `connector/src/binancefutures/user_data_stream.rs` already handles `EventStream::OrderTradeUpdate` by calling `OrderManager::update_from_ws`.
- `connector/src/binancefutures/ordermanager.rs` already has REST/WS dual-channel order lifecycle logic.

Therefore the missing piece is connection-level subscription to `ORDER_TRADE_UPDATE`, not order-manager logic.

## Design decisions

1. Use two WebSocket private stream connections with the same listen key:
   - `events=ACCOUNT_UPDATE`
   - `events=ORDER_TRADE_UPDATE`
2. Do not use comma-separated events because the observed Binance private stream behavior appears to require one `events=` type per stream.
3. Do not change `OrderManager` in this plan.
4. Do not loosen live open-order safety in this plan.
5. Keep listen-key keepalive behavior minimal in the first implementation. If both streams independently keepalive the same listen key every 30 minutes, accept that initially because the call is idempotent and low frequency. A later cleanup can centralize keepalive if needed.

## Files

- Modify: `connector/src/binancefutures/mod.rs`
  - Add a small typed event enum for private user stream event names.
  - Refactor `user_data_stream_url` to accept the event enum.
  - Add tests for both `ACCOUNT_UPDATE` and `ORDER_TRADE_UPDATE` URL generation.
  - Launch both private user streams in `connect_user_data_stream`.
- Use unchanged: `connector/src/binancefutures/user_data_stream.rs`
  - Already processes `AccountUpdate` and `OrderTradeUpdate`.
- Use unchanged: `connector/src/binancefutures/msg/stream.rs`
  - Already parses `ORDER_TRADE_UPDATE`.
- Use unchanged: `connector/src/binancefutures/ordermanager.rs`
  - Already updates order lifecycle from WebSocket order trade updates.
- Update: `TODO_live_open_order_alignment.md`
  - Mark Plan C implementation/validation status after execution.
- Optional update after live validation: `local_live_analysis/<new_run_id>/live_summary.md`
  - Record whether open-order drift recurred.

---

## Task 1: Add typed private user stream event URL tests

**Files:**
- Modify: `/home/molly/project/hftbacktest/connector/src/binancefutures/mod.rs`

- [ ] **Step 1: Add failing test for account update URL with event enum**

In `connector/src/binancefutures/mod.rs`, replace the current test module with this expanded test module:

```rust
#[cfg(test)]
mod tests {
    use super::{BinanceFutures, UserDataStreamEvent};

    #[test]
    fn user_data_stream_url_uses_private_endpoint_with_account_updates() {
        let url = BinanceFutures::user_data_stream_url(
            "wss://fstream.binance.com/ws",
            "listen-key",
            UserDataStreamEvent::AccountUpdate,
        );

        assert_eq!(
            url,
            "wss://fstream.binance.com/private/ws?listenKey=listen-key&events=ACCOUNT_UPDATE"
        );
    }

    #[test]
    fn user_data_stream_url_uses_private_endpoint_with_order_trade_updates() {
        let url = BinanceFutures::user_data_stream_url(
            "wss://fstream.binance.com/ws",
            "listen-key",
            UserDataStreamEvent::OrderTradeUpdate,
        );

        assert_eq!(
            url,
            "wss://fstream.binance.com/private/ws?listenKey=listen-key&events=ORDER_TRADE_UPDATE"
        );
    }

    #[test]
    fn user_data_stream_url_trims_private_base_url_slashes() {
        let url = BinanceFutures::user_data_stream_url(
            "wss://fstream.binance.com/ws/",
            "listen-key",
            UserDataStreamEvent::OrderTradeUpdate,
        );

        assert_eq!(
            url,
            "wss://fstream.binance.com/private/ws?listenKey=listen-key&events=ORDER_TRADE_UPDATE"
        );
    }
}
```

- [ ] **Step 2: Run the targeted test and verify it fails**

Run:

```bash
cd /home/molly/project/hftbacktest && cargo test -p connector binancefutures::tests::user_data_stream_url -- --nocapture
```

Expected before implementation: compile failure because `UserDataStreamEvent` does not exist and `user_data_stream_url` accepts only two arguments.

---

## Task 2: Implement typed private user stream event URL builder

**Files:**
- Modify: `/home/molly/project/hftbacktest/connector/src/binancefutures/mod.rs`

- [ ] **Step 1: Add private user stream event enum**

In `connector/src/binancefutures/mod.rs`, below `type SharedSymbolSet = Arc<Mutex<HashSet<String>>>;`, add:

```rust
#[derive(Clone, Copy, Debug)]
enum UserDataStreamEvent {
    AccountUpdate,
    OrderTradeUpdate,
}

impl UserDataStreamEvent {
    fn as_str(self) -> &'static str {
        match self {
            Self::AccountUpdate => "ACCOUNT_UPDATE",
            Self::OrderTradeUpdate => "ORDER_TRADE_UPDATE",
        }
    }
}
```

- [ ] **Step 2: Refactor URL builder to accept the event enum**

Replace the existing `user_data_stream_url` function in `impl BinanceFutures` with:

```rust
fn user_data_stream_url(
    base_url: &str,
    listen_key: &str,
    event: UserDataStreamEvent,
) -> String {
    let private_base_url = base_url
        .trim_end_matches('/')
        .trim_end_matches("/ws");
    format!(
        "{private_base_url}/private/ws?listenKey={listen_key}&events={}",
        event.as_str()
    )
}
```

- [ ] **Step 3: Run URL builder tests**

Run:

```bash
cd /home/molly/project/hftbacktest && cargo test -p connector binancefutures::tests::user_data_stream_url -- --nocapture
```

Expected: all three URL tests pass.

---

## Task 3: Launch separate account and order user streams

**Files:**
- Modify: `/home/molly/project/hftbacktest/connector/src/binancefutures/mod.rs`

- [ ] **Step 1: Add helper to connect one private user stream event**

Inside `impl BinanceFutures`, above `pub fn connect_user_data_stream`, add this helper:

```rust
fn connect_user_data_stream_event(
    base_url: String,
    client: BinanceFuturesClient,
    ev_tx: UnboundedSender<PublishEvent>,
    order_manager: SharedOrderManager,
    instruments: SharedSymbolSet,
    symbol_tx: Sender<String>,
    event: UserDataStreamEvent,
) {
    tokio::spawn(async move {
        let _ = Retry::new(ExponentialBackoff::default())
            .error_handler(|error: BinanceFuturesError| {
                error!(
                    ?error,
                    ?event,
                    "An error occurred in the user data stream connection."
                );
                ev_tx
                    .send(PublishEvent::LiveEvent(LiveEvent::Error(LiveError::with(
                        ErrorKind::ConnectionInterrupted,
                        error.into(),
                    ))))
                    .unwrap();
                Ok(())
            })
            .retry(|| async {
                let mut stream = user_data_stream::UserDataStream::new(
                    client.clone(),
                    ev_tx.clone(),
                    order_manager.clone(),
                    instruments.clone(),
                    symbol_tx.subscribe(),
                );

                debug!(?event, "Requesting the listen key for the user data stream...");
                let listen_key = stream.get_listen_key().await?;

                debug!(?event, "Connecting to the user data stream...");
                let url = Self::user_data_stream_url(&base_url, &listen_key, event);
                stream.connect(&url).await?;
                debug!(?event, "The user data stream connection is permanently closed.");
                Ok(())
            })
            .await;
    });
}
```

This helper deliberately creates one stream per event. It requests its own listen key per stream. That avoids shared listen-key lifecycle coupling in the first implementation. If Binance allows one listen key for multiple private event streams, this can be optimized later, but correctness and isolation are more important here.

- [ ] **Step 2: Replace `connect_user_data_stream` body with two event streams**

Replace the existing `connect_user_data_stream` function body with:

```rust
pub fn connect_user_data_stream(&self, ev_tx: UnboundedSender<PublishEvent>) {
    let base_url = self.config.stream_url.clone();
    let client = self.client.clone();
    let order_manager = self.order_manager.clone();
    let instruments = self.symbols.clone();
    let symbol_tx = self.symbol_tx.clone();

    Self::connect_user_data_stream_event(
        base_url.clone(),
        client.clone(),
        ev_tx.clone(),
        order_manager.clone(),
        instruments.clone(),
        symbol_tx.clone(),
        UserDataStreamEvent::AccountUpdate,
    );

    Self::connect_user_data_stream_event(
        base_url,
        client,
        ev_tx,
        order_manager,
        instruments,
        symbol_tx,
        UserDataStreamEvent::OrderTradeUpdate,
    );
}
```

- [ ] **Step 3: Run compiler tests for connector**

Run:

```bash
cd /home/molly/project/hftbacktest && cargo test -p connector binancefutures -- --nocapture
```

Expected: tests compile and pass.

If this fails because `Sender<String>` is not in scope for the helper signature, confirm that `tokio::sync::{broadcast, broadcast::Sender, mpsc::UnboundedSender};` is already imported at the top of `mod.rs`. If not, add `broadcast::Sender` to the existing import instead of introducing another alias.

---

## Task 4: Add order-trade update observability without changing order logic

**Files:**
- Modify: `/home/molly/project/hftbacktest/connector/src/binancefutures/user_data_stream.rs`

- [ ] **Step 1: Add a debug log when `ORDER_TRADE_UPDATE` is received**

In `UserDataStream::process_message`, inside the `EventStream::OrderTradeUpdate(data) => {` branch, add this as the first statement in the branch:

```rust
let client_order_id_tail = data
    .order
    .client_order_id
    .chars()
    .rev()
    .take(8)
    .collect::<Vec<_>>()
    .into_iter()
    .rev()
    .collect::<String>();
debug!(
    symbol = %data.order.symbol,
    client_order_id_tail = %client_order_id_tail,
    order_id = data.order.order_id,
    execution_type = %data.order.execution_type,
    status = ?data.order.order_status,
    "Received Binance futures ORDER_TRADE_UPDATE."
);
```

The branch should then continue to call:

```rust
match self.order_manager.lock().unwrap().update_from_ws(&data) {
```

Do not change the `OrderManager` update behavior in this task.

- [ ] **Step 2: Run connector tests**

Run:

```bash
cd /home/molly/project/hftbacktest && cargo test -p connector binancefutures -- --nocapture
```

Expected: tests pass.

---

## Task 5: Update open-order alignment TODO with Plan C validation steps

**Files:**
- Modify: `/home/molly/project/hftbacktest/TODO_live_open_order_alignment.md`

- [ ] **Step 1: Update proposed fix section**

Replace the current `## Proposed fix` section with:

```markdown
## Plan C proposed fix

1. Add a typed Binance futures private user stream event enum:
   - `ACCOUNT_UPDATE`
   - `ORDER_TRADE_UPDATE`
2. Refactor the private user stream URL builder to accept the event enum.
3. Start two private user stream connections:
   - one `ACCOUNT_UPDATE` stream for position updates
   - one `ORDER_TRADE_UPDATE` stream for order lifecycle updates
4. Keep `OrderManager` semantics unchanged. It already consumes `ORDER_TRADE_UPDATE` through `update_from_ws`.
5. Add observability log line when `ORDER_TRADE_UPDATE` is received.
6. Verify with a non-filling submit/cancel order that connector receives `ORDER_TRADE_UPDATE`.
7. Run short live validation.
8. Then rerun 2H aligned live test.

Do not disable or loosen open-order safety unless intentionally doing a diagnostic-only run.
```

- [ ] **Step 2: Add success criteria section**

Append this section after the Plan C proposed fix:

```markdown
## Plan C success criteria

Short validation succeeds if:

- connector logs at least one `Received Binance futures ORDER_TRADE_UPDATE` message for a connector-owned order, with `client_order_id_tail` matching the expected order id suffix
- local open orders and REST open orders converge after submit/cancel lifecycle
- `position_mismatch` remains safe
- `open_order_mismatch_pending` does not persist after grace/confirmation windows

2H validation succeeds if:

- the run does not stop due to open-order drift
- if it stops, the stop reason is a different real safety condition
- the resulting live audit can be used as a cleaner Plan B/Plan C backtest-live alignment baseline
```

---

## Task 6: Run local verification

**Files:**
- No additional file modifications unless verification exposes a specific failure.

- [ ] **Step 1: Run connector tests**

Run:

```bash
cd /home/molly/project/hftbacktest && cargo test -p connector binancefutures -- --nocapture
```

Expected: all Binance futures connector tests pass.

- [ ] **Step 2: Run Rust formatting check**

Run:

```bash
cd /home/molly/project/hftbacktest && cargo fmt --check
```

Expected: exits `0`.

If formatting fails, run:

```bash
cd /home/molly/project/hftbacktest && cargo fmt
```

Then rerun:

```bash
cd /home/molly/project/hftbacktest && cargo fmt --check
```

Expected: exits `0`.

- [ ] **Step 3: Inspect diff scope**

Run:

```bash
cd /home/molly/project/hftbacktest && git diff -- connector/src/binancefutures/mod.rs connector/src/binancefutures/user_data_stream.rs TODO_live_open_order_alignment.md
```

Expected: diff only contains:

- typed event enum
- URL builder parameterization
- two stream launches
- `ORDER_TRADE_UPDATE` debug log
- TODO validation documentation

No `OrderManager` behavior changes and no live safety threshold changes should appear.

---

## Task 7: Short live validation on AWS

**Files:**
- Use existing live config and scripts. Do not change strategy parameters unless explicitly instructed.
- Use: `/home/molly/project/hftbacktest/examples/binance_tick_mm/live_tick_mm.py`
- Use current live config path selected by operator.

- [ ] **Step 1: Build/deploy connector to AWS live environment**

Use the existing project deployment flow for the connector/live bot. Before running live trading, verify the deployed binary or Python environment includes the modified connector.

Expected evidence:

```text
connector build/deploy completed
running live bot uses modified Binance futures connector
```

- [ ] **Step 2: Run non-filling submit/cancel validation**

Run a short, low-risk live validation that creates and cancels a connector-owned order far enough from touch to avoid an intentional fill.

Expected evidence in logs:

```text
Received Binance futures ORDER_TRADE_UPDATE.
```

Expected safety state:

```text
local open orders converge with REST open orders after cancel
position_mismatch remains safe
```

- [ ] **Step 3: Run 10-15 minute live validation**

Run the existing short live validation flow.

Expected outcome:

```text
no persistent open_order_mismatch_pending after grace/confirmation windows
position_mismatch remains safe
ORDER_TRADE_UPDATE logs appear for connector-owned order lifecycle events
```

If this fails because no `ORDER_TRADE_UPDATE` logs appear, stop and inspect the actual WebSocket URL and Binance private stream response before trying further fixes.

---

## Task 8: 2H aligned live rerun and documentation

**Files:**
- Create/update local analysis directory for the new run after artifacts are downloaded.
- Update: `/home/molly/project/hftbacktest/TODO_live_open_order_alignment.md`
- Optional update: relevant `local_live_analysis/<new_run_id>/live_summary.md`

- [ ] **Step 1: Run 2H aligned live test after short validation passes**

Run the same style of 2H aligned live test used for `live_btcusdt_1777342116`.

Expected outcome:

```text
run completes without open-order drift stop
```

Acceptable alternate outcome:

```text
run stops for a different real safety condition, with open-order state still aligned
```

- [ ] **Step 2: Download and analyze new live audit**

Use the existing local live analysis workflow to download the new audit/config/artifacts.

Expected evidence:

```text
live_summary.md created or updated for new run
open_order_mismatch_pending is absent or transient only
ORDER_TRADE_UPDATE caveat removed for the new run
```

- [ ] **Step 3: Update TODO status**

Append this section to `TODO_live_open_order_alignment.md`, replacing values with the observed run id and result:

```markdown
## Plan C validation result

Status: **complete** / **blocked**.

- Validation run id: `<RUN_ID>`
- Short validation: `<passed/blocked>`
- 2H validation: `<passed/blocked>`
- `ORDER_TRADE_UPDATE` observed: `<yes/no>`
- Final safety status: `<status>`
- Open-order mismatch recurrence: `<none/transient/persistent>`
- Position mismatch status: `<safe/unsafe>`

Conclusion: `<one concise sentence explaining whether Plan C fixed live open-order lifecycle alignment>`.
```

---

## Caveats

1. This plan intentionally does not modify `OrderManager` behavior. The existing manager already expects WebSocket order lifecycle messages.
2. This plan intentionally does not loosen open-order safety. Safety should validate the fix, not be bypassed.
3. The first implementation may use independent listen keys per event stream. This is acceptable because it avoids coupling lifecycle between two streams. Optimize listen-key sharing only after correctness is validated.
4. If Binance rejects separate private event streams or does not emit `ORDER_TRADE_UPDATE`, stop and inspect the raw WebSocket response before attempting another fix.
5. A clean Plan C live run should become the next backtest-live alignment baseline; the older `live_btcusdt_1777342116` run remains diagnostic because it lacked order lifecycle events.

## Verification

Local verification:

```bash
cd /home/molly/project/hftbacktest && cargo test -p connector binancefutures -- --nocapture
cd /home/molly/project/hftbacktest && cargo fmt --check
```

Live verification:

```text
ORDER_TRADE_UPDATE logs observed for connector-owned orders
local open orders converge with REST open orders
position_mismatch remains safe
short validation passes before 2H validation
```

## Self-review

- Spec coverage: The plan covers typed event URL generation, two private stream launches, observability, local verification, short live validation, and 2H validation.
- Placeholder scan: Implementation steps include concrete code and commands. Runtime validation result fields use explicit replacement markers because values are only known after live execution.
- Type consistency: The event enum is consistently named `UserDataStreamEvent` with variants `AccountUpdate` and `OrderTradeUpdate`; URL builder consistently accepts `(base_url, listen_key, event)`.
