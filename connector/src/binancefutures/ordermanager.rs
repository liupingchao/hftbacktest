use std::sync::{Arc, Mutex};

use chrono::Utc;
use hashbrown::HashMap;
use hftbacktest::types::{Order, OrderId, Status};
use tracing::error;

use crate::{
    binancefutures::{
        BinanceFuturesError,
        msg::{rest::OrderResponse, stream::OrderTradeUpdate},
    },
    connector::GetOrders,
    utils::{RefSymbolOrderId, SymbolOrderId, generate_rand_string},
};

#[derive(Debug)]
struct OrderExt {
    symbol: String,
    order: Order,
    removed_by_ws: bool,
    removed_by_rest: bool,
}

pub type SharedOrderManager = Arc<Mutex<OrderManager>>;

pub type ClientOrderId = String;

/// Binance has separated channels for REST APIs and Websocket. Order responses are delivered
/// through these channels, with no guaranteed order of transmission. To prevent duplicate handling
/// of order responses, such as order deletion due to cancellation or fill, OrderManager manages the
/// order states before transmitting the responses to a live bot.
///
/// Deletions must be confirmed by both channels. If not, differences in response times could result
/// in attempts to update an order that has already been deleted, potentially creating a ghost order
/// unintentionally.
///
/// To handle this, the `client_order_id` should include a random ID to differentiate it, even when
/// the order ID is the same(bot's order id). This is necessary because the order deletion is
/// immediately notified to the bot, but the Connector must still retain the `client_order_id` in
/// case an update arrives later from the other channel, which has not yet sent the deletion
/// message.
#[derive(Default, Debug)]
pub struct OrderManager {
    prefix: String,
    orders: HashMap<ClientOrderId, OrderExt>,
    order_id_map: HashMap<SymbolOrderId, ClientOrderId>,
}

impl OrderManager {
    pub fn new(prefix: &str) -> Self {
        Self {
            prefix: prefix.to_string(),
            orders: Default::default(),
            order_id_map: Default::default(),
        }
    }

    pub fn update_from_ws(
        &mut self,
        resp: &OrderTradeUpdate,
    ) -> Result<Option<Order>, BinanceFuturesError> {
        if !resp.order.client_order_id.starts_with(&self.prefix) {
            return Err(BinanceFuturesError::PrefixUnmatched);
        }
        let order_ext = self
            .orders
            .get_mut(&resp.order.client_order_id)
            .ok_or(BinanceFuturesError::OrderNotFound)?;

        let already_removed = order_ext.removed_by_ws || order_ext.removed_by_rest;
        let incoming_exch_timestamp = resp.transaction_time * 1_000_000;
        let incoming_is_terminal =
            !matches!(resp.order.order_status, Status::New | Status::PartiallyFilled);
        let current_is_terminal =
            !matches!(order_ext.order.status, Status::New | Status::PartiallyFilled);
        if incoming_is_terminal
            || (!current_is_terminal && incoming_exch_timestamp >= order_ext.order.exch_timestamp)
        {
            order_ext.order.qty = resp.order.original_qty;
            order_ext.order.leaves_qty =
                resp.order.original_qty - resp.order.order_filled_accumulated_qty;
            order_ext.order.side = resp.order.side;
            order_ext.order.time_in_force = resp.order.time_in_force;
            order_ext.order.exch_timestamp = incoming_exch_timestamp;
            order_ext.order.status = resp.order.order_status;
            order_ext.order.exec_price_tick =
                (resp.order.last_filled_price / order_ext.order.tick_size).round() as i64;
            order_ext.order.exec_qty = resp.order.order_last_filled_qty;
            order_ext.order.order_type = resp.order.order_type;
            if incoming_is_terminal {
                order_ext.order.req = Status::None;
            }
        }

        let result = if already_removed && !incoming_is_terminal {
            None
        } else {
            Some(order_ext.order.clone())
        };

        if order_ext.order.status != Status::New
            && order_ext.order.status != Status::PartiallyFilled
        {
            order_ext.removed_by_ws = true;
            if !already_removed {
                self.order_id_map.remove(&RefSymbolOrderId::new(
                    &order_ext.symbol,
                    order_ext.order.order_id,
                ));
            }

            if order_ext.removed_by_ws && order_ext.removed_by_rest {
                self.orders.remove(&resp.order.client_order_id).unwrap();
            }
        }

        Ok(result)
    }

    pub fn update_submit_fail(
        &mut self,
        client_order_id: &ClientOrderId,
        error: &BinanceFuturesError,
    ) -> Option<Order> {
        match error {
            BinanceFuturesError::OrderError { code: -5022, .. } => {
                // GTX rejection.
            }
            BinanceFuturesError::OrderError { code: -1008, .. } => {
                // Server is currently overloaded with other requests. Please try again in a few minutes.
                error!(
                    "Server is currently overloaded with other requests. Please try again in a few minutes."
                );
            }
            BinanceFuturesError::OrderError { code: -2019, .. } => {
                // Margin is insufficient.
                error!("Margin is insufficient.");
            }
            BinanceFuturesError::OrderError { code: -1015, .. } => {
                // Too many new orders; current limit is 300 orders per TEN_SECONDS.
                error!("Too many new orders; current limit is 300 orders per TEN_SECONDS.");
            }
            error => {
                error!(?error, "submit error");
            }
        }
        self.update_from_rest_fail(client_order_id, Some(Status::Expired))
    }

    pub fn update_cancel_fail(
        &mut self,
        client_order_id: &ClientOrderId,
        error: &BinanceFuturesError,
    ) -> Option<Order> {
        match error {
            BinanceFuturesError::OrderError { code: -2011, .. } => {
                // The given order may no longer exist; it could have already been filled or
                // canceled. But, it cannot determine the order status because it lacks the
                // necessary information.
                self.update_from_rest_fail(client_order_id, Some(Status::None))
            }
            error => {
                error!(?error, "cancel error");
                self.update_from_rest_fail(client_order_id, None)
            }
        }
    }

    pub fn update_from_rest_fail(
        &mut self,
        client_order_id: &ClientOrderId,
        status: Option<Status>,
    ) -> Option<Order> {
        let order_ext = self.orders.get_mut(client_order_id)?;
        // .ok_or(BinanceFuturesError::OrderNotFound)?;

        let already_removed = order_ext.removed_by_ws || order_ext.removed_by_rest;
        if let Some(status) = status {
            order_ext.order.status = status;
        }
        order_ext.order.req = Status::None;

        let result = if already_removed {
            None
        } else {
            Some(order_ext.order.clone())
        };

        if order_ext.order.status != Status::New
            && order_ext.order.status != Status::PartiallyFilled
        {
            order_ext.removed_by_rest = true;
            if !already_removed {
                self.order_id_map.remove(&RefSymbolOrderId::new(
                    &order_ext.symbol,
                    order_ext.order.order_id,
                ));
            }

            if order_ext.removed_by_ws && order_ext.removed_by_rest {
                self.orders.remove(client_order_id).unwrap();
            }
        }

        result
    }

    pub fn update_from_rest(
        &mut self,
        client_order_id: &ClientOrderId,
        resp: &OrderResponse,
    ) -> Option<Order> {
        let order_ext = self.orders.get_mut(client_order_id)?;
        // .ok_or(BinanceFuturesError::OrderNotFound)?;

        let already_removed = order_ext.removed_by_ws || order_ext.removed_by_rest;
        let incoming_exch_timestamp = resp.update_time * 1_000_000;
        let incoming_is_terminal = !matches!(resp.status, Status::New | Status::PartiallyFilled);
        let current_is_terminal =
            !matches!(order_ext.order.status, Status::New | Status::PartiallyFilled);
        if !order_ext.removed_by_ws
            && (incoming_is_terminal
                || (!current_is_terminal
                    && incoming_exch_timestamp >= order_ext.order.exch_timestamp))
        {
            order_ext.order.qty = resp.orig_qty;
            order_ext.order.leaves_qty = resp.orig_qty - resp.cum_qty;
            order_ext.order.side = resp.side;
            order_ext.order.time_in_force = resp.time_in_force;
            order_ext.order.exch_timestamp = incoming_exch_timestamp;
            order_ext.order.status = resp.status;
            // The last filled price isn't available in the REST response.
            // Execution details are expected to be received via the WebSocket stream.
            order_ext.order.exec_qty = resp.executed_qty;
            order_ext.order.order_type = resp.ty;
            order_ext.order.req = Status::None;
        }

        let result = if incoming_is_terminal {
            if order_ext.removed_by_ws {
                None
            } else {
                Some(order_ext.order.clone())
            }
        } else {
            if already_removed {
                None
            } else {
                Some(order_ext.order.clone())
            }
        };

        if order_ext.order.status != Status::New
            && order_ext.order.status != Status::PartiallyFilled
        {
            order_ext.removed_by_rest = true;
            if !already_removed {
                self.order_id_map.remove(&RefSymbolOrderId::new(
                    &order_ext.symbol,
                    order_ext.order.order_id,
                ));
            }

            if order_ext.removed_by_ws && order_ext.removed_by_rest {
                self.orders.remove(client_order_id).unwrap();
            }
        }

        result
    }

    pub fn prepare_client_order_id(&mut self, symbol: String, order: Order) -> Option<String> {
        let symbol_order_id = SymbolOrderId::new(symbol.clone(), order.order_id);
        if self.order_id_map.contains_key(&symbol_order_id) {
            return None;
        }

        let client_order_id = format!("{}{}", self.prefix, generate_rand_string(16));
        if self.orders.contains_key(&client_order_id) {
            return None;
        }

        self.order_id_map
            .insert(symbol_order_id, client_order_id.clone());
        self.orders.insert(
            client_order_id.clone(),
            OrderExt {
                symbol,
                order,
                removed_by_ws: false,
                removed_by_rest: false,
            },
        );
        Some(client_order_id)
    }

    pub fn get_client_order_id(&self, symbol: &str, order_id: OrderId) -> Option<String> {
        self.order_id_map
            .get(&RefSymbolOrderId::new(symbol, order_id))
            .cloned()
    }

    /// Due to API instability or network issues, discrepancies can occur where an order is deleted
    /// by one channel but remains active because its deletion wasn't confirmed by both channels.
    /// The gc method resolves this by removing orders that were deleted by one channel but not
    /// confirmed by the other, after a defined threshold period.
    pub fn gc(&mut self) {
        let now = Utc::now().timestamp_nanos_opt().unwrap();
        let stale_ts = now - 300_000_000_000;
        let stale_ids: Vec<(_, _)> = self
            .orders
            .iter()
            .filter(|&(_, wrapper)| {
                wrapper.order.status != Status::New
                    && wrapper.order.status != Status::PartiallyFilled
                    && wrapper.order.status != Status::Unsupported
                    && wrapper.order.exch_timestamp < stale_ts
            })
            .map(|(client_order_id, wrapper)| {
                (
                    client_order_id.clone(),
                    SymbolOrderId::new(wrapper.symbol.clone(), wrapper.order.order_id),
                )
            })
            .collect();
        for (client_order_id, order_id) in stale_ids.iter() {
            if self.order_id_map.contains_key(order_id) {
                // todo: something went wrong?
                self.order_id_map.remove(order_id).unwrap();
            }
            self.orders.remove(client_order_id);
        }
    }

    pub fn cancel_all_from_rest(&mut self, symbol: &str) -> Vec<Order> {
        let mut removed_orders = Vec::new();
        let mut removed_order_ids = Vec::new();
        for (client_order_id, order_ext) in &mut self.orders {
            if order_ext.symbol != symbol {
                continue;
            }
            let already_removed = order_ext.removed_by_ws || order_ext.removed_by_rest;

            order_ext.removed_by_rest = true;
            order_ext.order.status = Status::Canceled;
            // todo: check if the exchange timestamp exists in the REST response.
            order_ext.order.exch_timestamp = Utc::now().timestamp_nanos_opt().unwrap();
            if !already_removed {
                self.order_id_map
                    .remove(&RefSymbolOrderId::new(symbol, order_ext.order.order_id));
                removed_orders.push(order_ext.order.clone());
            }

            // Completely deletes the order if it is removed by both the REST response and the
            // WebSocket stream.
            if order_ext.removed_by_ws && order_ext.removed_by_rest {
                removed_order_ids.push(client_order_id.clone());
            }
        }

        for order_id in removed_order_ids {
            self.orders.remove(&order_id).unwrap();
        }
        removed_orders
    }
}

impl GetOrders for OrderManager {
    fn orders(&self, symbol: Option<String>) -> Vec<Order> {
        self.orders
            .iter()
            .filter(|(_, order)| {
                symbol.as_ref().map(|s| order.symbol == *s).unwrap_or(true) && order.order.active()
            })
            .map(|(_, order)| &order.order)
            .cloned()
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use hftbacktest::types::{OrdType, Side, TimeInForce};

    fn make_order(order_id: u64) -> Order {
        let mut order = Order::new(
            order_id,
            100,
            0.1,
            0.001,
            Side::Buy,
            OrdType::Limit,
            TimeInForce::GTC,
        );
        order.status = Status::New;
        order.req = Status::New;
        order
    }

    fn make_rest_response(
        client_order_id: &str,
        order_id: i64,
        status: Status,
        update_time: i64,
        executed_qty: f64,
    ) -> OrderResponse {
        OrderResponse {
            client_order_id: client_order_id.to_string(),
            cum_qty: executed_qty,
            cum_quote: Some(executed_qty * 100.0),
            cum_base: None,
            executed_qty,
            order_id,
            avg_price: Some(100.0),
            orig_qty: 0.001,
            price: 100.0,
            reduce_only: false,
            side: Side::Buy,
            position_side: "BOTH".to_string(),
            status,
            stop_price: 0.0,
            close_position: false,
            symbol: "btcusdt".to_string(),
            pair: None,
            time_in_force: TimeInForce::GTC,
            ty: OrdType::Limit,
            orig_type: OrdType::Limit,
            activate_price: None,
            price_rate: None,
            update_time,
            working_type: "MARK_PRICE".to_string(),
            price_protect: false,
            price_match: "NONE".to_string(),
            self_trade_prevention_mode: "NONE".to_string(),
            good_till_date: 0,
        }
    }

    fn make_ws_update(
        client_order_id: &str,
        order_id: i64,
        status: Status,
        transaction_time: i64,
        order_last_filled_qty: f64,
        order_filled_accumulated_qty: f64,
        last_filled_price: f64,
    ) -> OrderTradeUpdate {
        OrderTradeUpdate {
            event_time: transaction_time,
            transaction_time,
            order: crate::binancefutures::msg::stream::Order {
                symbol: "btcusdt".to_string(),
                client_order_id: client_order_id.to_string(),
                side: Side::Buy,
                order_type: OrdType::Limit,
                time_in_force: TimeInForce::GTC,
                original_qty: 0.001,
                original_price: 100.0,
                average_price: last_filled_price,
                stop_price: 0.0,
                execution_type: "TRADE".to_string(),
                order_status: status,
                order_id,
                order_last_filled_qty,
                order_filled_accumulated_qty,
                last_filled_price,
                order_trade_time: transaction_time,
                trade_id: 1,
            },
        }
    }

    #[test]
    fn rest_then_ws_terminal_update_keeps_late_ws_details_even_if_older() {
        let mut manager = OrderManager::new("live-");
        let symbol = "btcusdt".to_string();
        let client_order_id = manager
            .prepare_client_order_id(symbol.clone(), make_order(5016))
            .unwrap();

        let rest_resp = make_rest_response(&client_order_id, 5016, Status::Canceled, 200, 0.0);
        let rest_order = manager
            .update_from_rest(&client_order_id, &rest_resp)
            .expect("rest response should be emitted");
        assert_eq!(rest_order.status, Status::Canceled);
        assert_eq!(rest_order.exch_timestamp, 200_000_000);
        assert!(manager.get_client_order_id(&symbol, 5016).is_none());

        let ws_update = make_ws_update(
            &client_order_id,
            5016,
            Status::Filled,
            100,
            0.001,
            0.001,
            100.1,
        );
        let ws_order = manager
            .update_from_ws(&ws_update)
            .expect("late ws terminal should still be emitted")
            .expect("late ws terminal should not be dropped");
        assert_eq!(ws_order.status, Status::Filled);
        assert_eq!(ws_order.exch_timestamp, 100_000_000);
        assert_eq!(ws_order.exec_qty, 0.001);
        assert_eq!(ws_order.exec_price_tick, 1001);
        assert!(manager.orders.get(&client_order_id).is_none());
    }

    #[test]
    fn ws_terminal_state_is_not_overwritten_by_late_rest_terminal() {
        let mut manager = OrderManager::new("live-");
        let symbol = "btcusdt".to_string();
        let client_order_id = manager
            .prepare_client_order_id(symbol.clone(), make_order(5016))
            .unwrap();

        let ws_update = make_ws_update(
            &client_order_id,
            5016,
            Status::Filled,
            200,
            0.001,
            0.001,
            100.1,
        );
        let ws_order = manager
            .update_from_ws(&ws_update)
            .expect("ws response should be emitted")
            .expect("ws terminal should be emitted");
        assert_eq!(ws_order.status, Status::Filled);
        assert_eq!(ws_order.exec_price_tick, 1001);

        let rest_resp = make_rest_response(&client_order_id, 5016, Status::Canceled, 100, 0.0);
        assert!(
            manager.update_from_rest(&client_order_id, &rest_resp).is_none(),
            "late rest terminal should not overwrite the ws terminal state"
        );

        assert!(manager.orders.get(&client_order_id).is_none());
        assert!(manager.get_client_order_id(&symbol, 5016).is_none());
    }
}
