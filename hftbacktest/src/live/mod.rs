use std::collections::HashMap;

pub use bot::{BotError, LiveBot, LiveBotBuilder};
pub use recorder::LoggingRecorder;

use crate::{
    prelude::StateValues,
    types::{Event, Order, OrderId},
};

mod bot;
pub mod ipc;
mod recorder;

/// Provides asset information for internal use.
pub struct Instrument<MD> {
    connector_name: String,
    symbol: String,
    tick_size: f64,
    lot_size: f64,
    depth: MD,
    last_trades: Vec<Event>,
    orders: HashMap<OrderId, Order>,
    last_feed_latency: Option<(i64, i64)>,
    last_order_latency: Option<(i64, i64, i64)>,
    last_position_exch_ts: i64,
    state: StateValues,
}

impl<MD> Instrument<MD> {
    /// * `connector_name` - Name of the [`Connector`], which is registered by
    ///   [`register()`](`LiveBotBuilder::register()`), through which this asset will be traded.
    /// * `symbol` - Symbol of the asset. You need to check with the [`Connector`] which symbology
    ///   is used.
    /// * `tick_size` - The minimum price fluctuation.
    /// * `lot_size` -  The minimum trade size.
    /// * `depth` -  The market depth.
    pub fn new(
        connector_name: &str,
        symbol: &str,
        tick_size: f64,
        lot_size: f64,
        depth: MD,
        last_trades_capacity: usize,
    ) -> Self {
        Self {
            connector_name: connector_name.to_string(),
            symbol: symbol.to_string(),
            tick_size,
            lot_size,
            depth,
            last_trades: Vec::with_capacity(last_trades_capacity),
            orders: Default::default(),
            last_feed_latency: None,
            last_order_latency: None,
            last_position_exch_ts: i64::MIN,
            state: Default::default(),
        }
    }

    fn apply_position_update(&mut self, qty: f64, exch_ts: i64) -> bool {
        if exch_ts >= self.last_position_exch_ts {
            self.state.position = qty;
            self.last_position_exch_ts = exch_ts;
            true
        } else {
            false
        }
    }
}

#[cfg(test)]
mod tests {
    use super::Instrument;
    use crate::depth::ROIVectorMarketDepth;

    fn instrument() -> Instrument<ROIVectorMarketDepth> {
        Instrument::new(
            "bf",
            "btcusdt",
            0.1,
            0.001,
            ROIVectorMarketDepth::new(0.1, 0.001, 60_000.0, 100_000.0),
            0,
        )
    }

    #[test]
    fn live_position_update_ignores_stale_exchange_timestamp() {
        let mut instrument = instrument();

        assert!(instrument.apply_position_update(-0.001, 200));
        assert_eq!(instrument.state.position, -0.001);
        assert_eq!(instrument.last_position_exch_ts, 200);

        assert!(!instrument.apply_position_update(0.0, 100));
        assert_eq!(instrument.state.position, -0.001);
        assert_eq!(instrument.last_position_exch_ts, 200);

        assert!(instrument.apply_position_update(-0.002, 300));
        assert_eq!(instrument.state.position, -0.002);
        assert_eq!(instrument.last_position_exch_ts, 300);
    }

    #[test]
    fn live_position_update_accepts_startup_zero_timestamp() {
        let mut instrument = instrument();

        assert!(instrument.apply_position_update(0.0, 0));
        assert_eq!(instrument.state.position, 0.0);
        assert_eq!(instrument.last_position_exch_ts, 0);
    }
}
