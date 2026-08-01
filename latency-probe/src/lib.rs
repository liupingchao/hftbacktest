use std::{
    collections::{BTreeMap, HashSet},
    fs::File,
    io::{BufRead, BufWriter, Write},
    path::Path,
    sync::{
        Arc,
        atomic::{AtomicU64, Ordering},
    },
    thread,
    time::{Instant, SystemTime, UNIX_EPOCH},
};

use anyhow::{Context, Result, bail};
use crossbeam_channel::{Receiver, Sender, TrySendError, bounded};
use serde::{Deserialize, Serialize};

pub const SCHEMA_VERSION: &str = "latency-trace-v1";
pub const STAGE_COUNT: usize = 15;
pub const REQUIRED_PUBLIC_STAGES: [Stage; 8] = [
    Stage::WsFrameReceive,
    Stage::ParseDone,
    Stage::BookApplyDone,
    Stage::SignalStart,
    Stage::SignalEnd,
    Stage::Decision,
    Stage::OrderEncodeDone,
    Stage::SocketSend,
];

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Venue {
    BinanceFutures,
    Hyperliquid,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[repr(usize)]
#[serde(rename_all = "snake_case")]
pub enum Stage {
    ExchangeEvent = 0,
    SocketReceive = 1,
    WsFrameReceive = 2,
    ParseDone = 3,
    BookApplyDone = 4,
    SignalStart = 5,
    SignalEnd = 6,
    Decision = 7,
    OrderEncodeDone = 8,
    SocketSend = 9,
    ExchangeAck = 10,
    Resting = 11,
    Fill = 12,
    CancelRequest = 13,
    CancelAck = 14,
}

impl Stage {
    pub const ALL: [Self; STAGE_COUNT] = [
        Self::ExchangeEvent,
        Self::SocketReceive,
        Self::WsFrameReceive,
        Self::ParseDone,
        Self::BookApplyDone,
        Self::SignalStart,
        Self::SignalEnd,
        Self::Decision,
        Self::OrderEncodeDone,
        Self::SocketSend,
        Self::ExchangeAck,
        Self::Resting,
        Self::Fill,
        Self::CancelRequest,
        Self::CancelAck,
    ];

    pub const fn as_str(self) -> &'static str {
        match self {
            Self::ExchangeEvent => "exchange_event",
            Self::SocketReceive => "socket_receive",
            Self::WsFrameReceive => "ws_frame_receive",
            Self::ParseDone => "parse_done",
            Self::BookApplyDone => "book_apply_done",
            Self::SignalStart => "signal_start",
            Self::SignalEnd => "signal_end",
            Self::Decision => "decision",
            Self::OrderEncodeDone => "order_encode_done",
            Self::SocketSend => "socket_send",
            Self::ExchangeAck => "exchange_ack",
            Self::Resting => "resting",
            Self::Fill => "fill",
            Self::CancelRequest => "cancel_request",
            Self::CancelAck => "cancel_ack",
        }
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct TraceRecord {
    pub schema_version: String,
    pub venue: Venue,
    pub channel: String,
    pub symbol: String,
    pub trace_id: u64,
    pub exchange_sequence: Option<u64>,
    pub unix_ns: [Option<u64>; STAGE_COUNT],
    pub monotonic_ns: [Option<u64>; STAGE_COUNT],
    #[serde(default)]
    pub duplicate_stage_marks: u32,
}

impl TraceRecord {
    pub fn new(
        venue: Venue,
        channel: impl Into<String>,
        symbol: impl Into<String>,
        trace_id: u64,
    ) -> Self {
        Self {
            schema_version: SCHEMA_VERSION.to_owned(),
            venue,
            channel: channel.into(),
            symbol: symbol.into(),
            trace_id,
            exchange_sequence: None,
            unix_ns: [None; STAGE_COUNT],
            monotonic_ns: [None; STAGE_COUNT],
            duplicate_stage_marks: 0,
        }
    }

    pub fn mark_unix(&mut self, stage: Stage, timestamp_ns: u64) {
        let slot = &mut self.unix_ns[stage as usize];
        if slot.is_some() {
            self.duplicate_stage_marks += 1;
        }
        *slot = Some(timestamp_ns);
    }

    pub fn mark_monotonic(&mut self, stage: Stage, timestamp_ns: u64) {
        let slot = &mut self.monotonic_ns[stage as usize];
        if slot.is_some() {
            self.duplicate_stage_marks += 1;
        }
        *slot = Some(timestamp_ns);
    }

    pub fn unix(&self, stage: Stage) -> Option<u64> {
        self.unix_ns[stage as usize]
    }

    pub fn monotonic(&self, stage: Stage) -> Option<u64> {
        self.monotonic_ns[stage as usize]
    }

    pub fn validate(&self) -> Result<()> {
        if self.schema_version != SCHEMA_VERSION {
            bail!(
                "unsupported schema version {:?}, expected {SCHEMA_VERSION}",
                self.schema_version
            );
        }
        Ok(())
    }

    pub fn public_validation(&self) -> TraceValidation {
        if self.duplicate_stage_marks != 0 {
            return TraceValidation::OutOfOrder;
        }
        let mut previous = 0_u64;
        for stage in REQUIRED_PUBLIC_STAGES {
            let Some(timestamp) = self.monotonic(stage) else {
                return TraceValidation::Incomplete;
            };
            if timestamp < previous {
                return TraceValidation::OutOfOrder;
            }
            previous = timestamp;
        }
        TraceValidation::Complete
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TraceValidation {
    Complete,
    Incomplete,
    OutOfOrder,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ClockDomain {
    Unix,
    Monotonic,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct IntervalDefinition {
    pub name: &'static str,
    pub start: Stage,
    pub end: Stage,
    pub clock: ClockDomain,
}

pub const INTERVALS: [IntervalDefinition; 14] = [
    IntervalDefinition {
        name: "feed_network",
        start: Stage::ExchangeEvent,
        end: Stage::WsFrameReceive,
        clock: ClockDomain::Unix,
    },
    IntervalDefinition {
        name: "socket_to_frame",
        start: Stage::SocketReceive,
        end: Stage::WsFrameReceive,
        clock: ClockDomain::Monotonic,
    },
    IntervalDefinition {
        name: "frame_to_parse",
        start: Stage::WsFrameReceive,
        end: Stage::ParseDone,
        clock: ClockDomain::Monotonic,
    },
    IntervalDefinition {
        name: "parse_to_book",
        start: Stage::ParseDone,
        end: Stage::BookApplyDone,
        clock: ClockDomain::Monotonic,
    },
    IntervalDefinition {
        name: "book_to_signal",
        start: Stage::BookApplyDone,
        end: Stage::SignalStart,
        clock: ClockDomain::Monotonic,
    },
    IntervalDefinition {
        name: "signal_compute",
        start: Stage::SignalStart,
        end: Stage::SignalEnd,
        clock: ClockDomain::Monotonic,
    },
    IntervalDefinition {
        name: "signal_to_decision",
        start: Stage::SignalEnd,
        end: Stage::Decision,
        clock: ClockDomain::Monotonic,
    },
    IntervalDefinition {
        name: "decision_to_encode",
        start: Stage::Decision,
        end: Stage::OrderEncodeDone,
        clock: ClockDomain::Monotonic,
    },
    IntervalDefinition {
        name: "encode_to_send",
        start: Stage::OrderEncodeDone,
        end: Stage::SocketSend,
        clock: ClockDomain::Monotonic,
    },
    IntervalDefinition {
        name: "tick_to_wire",
        start: Stage::WsFrameReceive,
        end: Stage::SocketSend,
        clock: ClockDomain::Monotonic,
    },
    IntervalDefinition {
        name: "send_to_ack",
        start: Stage::SocketSend,
        end: Stage::ExchangeAck,
        clock: ClockDomain::Monotonic,
    },
    IntervalDefinition {
        name: "send_to_resting",
        start: Stage::SocketSend,
        end: Stage::Resting,
        clock: ClockDomain::Monotonic,
    },
    IntervalDefinition {
        name: "resting_to_fill",
        start: Stage::Resting,
        end: Stage::Fill,
        clock: ClockDomain::Monotonic,
    },
    IntervalDefinition {
        name: "cancel_round_trip",
        start: Stage::CancelRequest,
        end: Stage::CancelAck,
        clock: ClockDomain::Monotonic,
    },
];

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct IntervalSummary {
    pub clock: ClockDomain,
    pub start_stage: String,
    pub end_stage: String,
    pub count: u64,
    pub missing: u64,
    pub invalid: u64,
    pub mean_ns: Option<f64>,
    pub p50_ns: Option<f64>,
    pub p90_ns: Option<f64>,
    pub p99_ns: Option<f64>,
    pub p999_ns: Option<f64>,
    pub max_ns: Option<u64>,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct Summary {
    pub schema_version: String,
    pub trace_count: u64,
    pub integrity: IntegritySummary,
    pub intervals: BTreeMap<String, IntervalSummary>,
}

pub fn summarize(records: &[TraceRecord]) -> Result<Summary> {
    summarize_with_drops(records, 0)
}

pub fn summarize_with_drops(records: &[TraceRecord], dropped_traces: u64) -> Result<Summary> {
    for record in records {
        record.validate()?;
    }

    let mut trace_ids = HashSet::with_capacity(records.len());
    let mut duplicate_trace_ids = 0_u64;
    let mut duplicate_stage_marks = 0_u64;
    let mut complete_required = 0_u64;
    let mut incomplete_required = 0_u64;
    let mut out_of_order = 0_u64;
    for record in records {
        if !trace_ids.insert(record.trace_id) {
            duplicate_trace_ids += 1;
        }
        duplicate_stage_marks += u64::from(record.duplicate_stage_marks);
        match record.public_validation() {
            TraceValidation::Complete => complete_required += 1,
            TraceValidation::Incomplete => incomplete_required += 1,
            TraceValidation::OutOfOrder => out_of_order += 1,
        }
    }

    let mut intervals = BTreeMap::new();
    for definition in INTERVALS {
        let mut valid = Vec::with_capacity(records.len());
        let mut missing = 0_u64;
        let mut invalid = 0_u64;

        for record in records {
            let timestamps = match definition.clock {
                ClockDomain::Unix => &record.unix_ns,
                ClockDomain::Monotonic => &record.monotonic_ns,
            };
            let start = timestamps[definition.start as usize];
            let end = timestamps[definition.end as usize];
            match (start, end) {
                (Some(start), Some(end)) if end >= start => valid.push(end - start),
                (Some(_), Some(_)) => invalid += 1,
                _ => missing += 1,
            }
        }

        valid.sort_unstable();
        let mean_ns = if valid.is_empty() {
            None
        } else {
            Some(valid.iter().map(|value| *value as f64).sum::<f64>() / valid.len() as f64)
        };
        let interval_summary = IntervalSummary {
            clock: definition.clock,
            start_stage: definition.start.as_str().to_owned(),
            end_stage: definition.end.as_str().to_owned(),
            count: valid.len() as u64,
            missing,
            invalid,
            mean_ns,
            p50_ns: quantile_r7(&valid, 0.5),
            p90_ns: quantile_r7(&valid, 0.9),
            p99_ns: quantile_r7(&valid, 0.99),
            p999_ns: quantile_r7(&valid, 0.999),
            max_ns: valid.last().copied(),
        };
        intervals.insert(definition.name.to_owned(), interval_summary);
    }

    Ok(Summary {
        schema_version: SCHEMA_VERSION.to_owned(),
        trace_count: records.len() as u64,
        integrity: IntegritySummary {
            traces_seen: records.len() as u64,
            unique_trace_ids: trace_ids.len() as u64,
            duplicate_trace_ids,
            duplicate_stage_marks,
            complete_required,
            incomplete_required,
            out_of_order,
            dropped_traces,
        },
        intervals,
    })
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct IntegritySummary {
    pub traces_seen: u64,
    pub unique_trace_ids: u64,
    pub duplicate_trace_ids: u64,
    pub duplicate_stage_marks: u64,
    pub complete_required: u64,
    pub incomplete_required: u64,
    pub out_of_order: u64,
    pub dropped_traces: u64,
}

fn quantile_r7(sorted: &[u64], probability: f64) -> Option<f64> {
    if sorted.is_empty() {
        return None;
    }
    if sorted.len() == 1 {
        return Some(sorted[0] as f64);
    }

    let index = probability * (sorted.len() - 1) as f64;
    let lower = index.floor() as usize;
    let upper = index.ceil() as usize;
    let weight = index - lower as f64;
    Some(sorted[lower] as f64 * (1.0 - weight) + sorted[upper] as f64 * weight)
}

pub fn write_ndjson<W: Write>(mut writer: W, records: &[TraceRecord]) -> Result<()> {
    for record in records {
        serde_json::to_writer(&mut writer, record).context("serialize trace record")?;
        writer.write_all(b"\n").context("write trace newline")?;
    }
    Ok(())
}

pub fn read_ndjson<R: BufRead>(reader: R) -> Result<Vec<TraceRecord>> {
    let mut records = Vec::new();
    for (index, line) in reader.lines().enumerate() {
        let line = line.with_context(|| format!("read NDJSON line {}", index + 1))?;
        if line.trim().is_empty() {
            continue;
        }
        let record: TraceRecord = serde_json::from_str(&line)
            .with_context(|| format!("parse NDJSON line {}", index + 1))?;
        record.validate()?;
        records.push(record);
    }
    Ok(records)
}

pub struct TraceHandoff {
    sender: Option<Sender<TraceRecord>>,
    dropped: Arc<AtomicU64>,
    writer_thread: Option<thread::JoinHandle<Result<()>>>,
}

impl TraceHandoff {
    pub fn create(path: &Path, capacity: usize) -> Result<Self> {
        if capacity == 0 {
            bail!("trace handoff capacity must be positive");
        }
        let file = File::create(path)
            .with_context(|| format!("create trace output {}", path.display()))?;
        let (sender, receiver) = bounded(capacity);
        let writer_thread = thread::spawn(move || write_trace_receiver(file, receiver));
        Ok(Self {
            sender: Some(sender),
            dropped: Arc::new(AtomicU64::new(0)),
            writer_thread: Some(writer_thread),
        })
    }

    pub fn try_send(&self, record: TraceRecord) -> bool {
        let Some(sender) = self.sender.as_ref() else {
            return false;
        };
        match sender.try_send(record) {
            Ok(()) => true,
            Err(TrySendError::Full(_)) | Err(TrySendError::Disconnected(_)) => {
                self.dropped.fetch_add(1, Ordering::Relaxed);
                false
            }
        }
    }

    pub fn dropped(&self) -> u64 {
        self.dropped.load(Ordering::Relaxed)
    }

    pub fn finish(mut self) -> Result<u64> {
        self.sender.take();
        if let Some(handle) = self.writer_thread.take() {
            handle
                .join()
                .map_err(|_| anyhow::anyhow!("trace writer thread panicked"))??;
        }
        Ok(self.dropped())
    }
}

fn write_trace_receiver(file: File, receiver: Receiver<TraceRecord>) -> Result<()> {
    let mut writer = BufWriter::new(file);
    for record in receiver {
        serde_json::to_writer(&mut writer, &record).context("serialize trace record")?;
        writer.write_all(b"\n").context("write trace newline")?;
    }
    writer.flush().context("flush trace output")
}

#[derive(Debug)]
pub struct ProcessClock {
    anchor: Instant,
}

impl Default for ProcessClock {
    fn default() -> Self {
        Self {
            anchor: Instant::now(),
        }
    }
}

impl ProcessClock {
    pub fn monotonic_ns(&self) -> u64 {
        self.anchor.elapsed().as_nanos().min(u128::from(u64::MAX)) as u64
    }

    pub fn unix_ns() -> Result<u64> {
        let duration = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .context("system clock is before Unix epoch")?;
        Ok(duration.as_nanos().min(u128::from(u64::MAX)) as u64)
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct BenchmarkSummary {
    pub iterations: u64,
    pub batch_size: u64,
    pub dropped_traces: u64,
    pub p50_ns_per_trace: u64,
    pub p99_ns_per_trace: u64,
    pub max_ns_per_trace: u64,
    pub max_p99_ns: u64,
    pub passed: bool,
}

pub fn benchmark_recorder(
    iterations: u64,
    queue_capacity: usize,
    max_p99_ns: u64,
) -> Result<BenchmarkSummary> {
    if iterations == 0 {
        bail!("benchmark iterations must be positive");
    }
    if queue_capacity == 0 {
        bail!("benchmark queue capacity must be positive");
    }
    let (sender, receiver) = bounded::<TraceRecord>(queue_capacity);
    let consumer = thread::spawn(move || {
        while let Ok(record) = receiver.recv() {
            std::hint::black_box(record);
        }
    });
    let clock = ProcessClock::default();
    let mut samples = Vec::with_capacity(iterations.min(1_000_000) as usize);
    let mut dropped_traces = 0_u64;

    for trace_id in 1..=iterations {
        let started = Instant::now();
        let mut record = TraceRecord::new(Venue::BinanceFutures, "bookTicker", "BTCUSDT", trace_id);
        for stage in REQUIRED_PUBLIC_STAGES {
            record.mark_monotonic(stage, clock.monotonic_ns());
        }
        if sender.try_send(record).is_err() {
            dropped_traces += 1;
        }
        let elapsed_ns = started.elapsed().as_nanos().min(u128::from(u64::MAX)) as u64;
        samples.push(elapsed_ns);
    }
    drop(sender);
    consumer
        .join()
        .map_err(|_| anyhow::anyhow!("benchmark consumer thread panicked"))?;
    samples.sort_unstable();
    let p50_ns_per_trace = quantile_r7(&samples, 0.5).unwrap_or(0.0).round() as u64;
    let p99_ns_per_trace = quantile_r7(&samples, 0.99).unwrap_or(0.0).round() as u64;
    let max_ns_per_trace = samples.last().copied().unwrap_or(0);
    Ok(BenchmarkSummary {
        iterations,
        batch_size: 1,
        dropped_traces,
        p50_ns_per_trace,
        p99_ns_per_trace,
        max_ns_per_trace,
        max_p99_ns,
        passed: dropped_traces == 0 && p99_ns_per_trace <= max_p99_ns,
    })
}

#[derive(Clone, Debug, PartialEq)]
pub struct PublicBbo {
    pub symbol: String,
    pub exchange_timestamp_ms: u64,
    pub exchange_sequence: Option<u64>,
    pub bid_price: f64,
    pub ask_price: f64,
}

#[derive(Deserialize)]
struct BinanceBookTicker {
    #[serde(rename = "E")]
    event_time: u64,
    #[serde(rename = "s")]
    symbol: String,
    #[serde(rename = "u")]
    update_id: Option<u64>,
    #[serde(rename = "b")]
    bid_price: String,
    #[serde(rename = "a")]
    ask_price: String,
}

#[derive(Deserialize)]
#[serde(untagged)]
enum BinanceMessage {
    Combined {
        #[allow(dead_code)]
        stream: String,
        data: BinanceBookTicker,
    },
    Direct(BinanceBookTicker),
}

pub fn parse_binance_book_ticker(payload: &str) -> Result<PublicBbo> {
    let message: BinanceMessage =
        serde_json::from_str(payload).context("parse Binance bookTicker")?;
    let ticker = match message {
        BinanceMessage::Combined { data, .. } | BinanceMessage::Direct(data) => data,
    };
    let bid_price = ticker.bid_price.parse().context("parse Binance bid")?;
    let ask_price = ticker.ask_price.parse().context("parse Binance ask")?;
    validate_bbo(bid_price, ask_price)?;
    Ok(PublicBbo {
        symbol: ticker.symbol,
        exchange_timestamp_ms: ticker.event_time,
        exchange_sequence: ticker.update_id,
        bid_price,
        ask_price,
    })
}

#[derive(Deserialize)]
struct HyperliquidBbo {
    coin: String,
    time: u64,
    bbo: [Option<HyperliquidLevel>; 2],
}

#[derive(Deserialize)]
struct HyperliquidL2Book {
    coin: String,
    time: u64,
    levels: [Vec<HyperliquidLevel>; 2],
}

#[derive(Deserialize)]
struct HyperliquidLevel {
    px: String,
}

#[derive(Deserialize)]
#[serde(tag = "channel", content = "data")]
enum HyperliquidMessage {
    #[serde(rename = "bbo")]
    Bbo(HyperliquidBbo),
    #[serde(rename = "l2Book")]
    L2Book(HyperliquidL2Book),
    #[serde(rename = "subscriptionResponse")]
    SubscriptionResponse(serde::de::IgnoredAny),
}

pub fn parse_hyperliquid_message(payload: &str) -> Result<Option<PublicBbo>> {
    let message: HyperliquidMessage =
        serde_json::from_str(payload).context("parse Hyperliquid message")?;
    let (symbol, time, bid_price, ask_price) = match message {
        HyperliquidMessage::Bbo(bbo) => {
            let bid = bbo.bbo[0].as_ref().context("Hyperliquid bbo missing bid")?;
            let ask = bbo.bbo[1].as_ref().context("Hyperliquid bbo missing ask")?;
            (
                bbo.coin,
                bbo.time,
                bid.px.parse().context("parse Hyperliquid bid")?,
                ask.px.parse().context("parse Hyperliquid ask")?,
            )
        }
        HyperliquidMessage::L2Book(book) => {
            let bid = book.levels[0]
                .first()
                .context("Hyperliquid l2Book missing bid")?;
            let ask = book.levels[1]
                .first()
                .context("Hyperliquid l2Book missing ask")?;
            (
                book.coin,
                book.time,
                bid.px.parse().context("parse Hyperliquid bid")?,
                ask.px.parse().context("parse Hyperliquid ask")?,
            )
        }
        HyperliquidMessage::SubscriptionResponse(_) => return Ok(None),
    };
    validate_bbo(bid_price, ask_price)?;
    Ok(Some(PublicBbo {
        symbol,
        exchange_timestamp_ms: time,
        exchange_sequence: None,
        bid_price,
        ask_price,
    }))
}

pub fn parse_hyperliquid_bbo(payload: &str) -> Result<PublicBbo> {
    parse_hyperliquid_message(payload)?.context("expected Hyperliquid bbo message")
}

fn validate_bbo(bid_price: f64, ask_price: f64) -> Result<()> {
    if !bid_price.is_finite()
        || !ask_price.is_finite()
        || bid_price <= 0.0
        || ask_price <= 0.0
        || bid_price > ask_price
    {
        bail!("invalid BBO bid={bid_price} ask={ask_price}");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn base_trace(trace_id: u64) -> TraceRecord {
        TraceRecord::new(Venue::BinanceFutures, "bookTicker", "BTCUSDT", trace_id)
    }

    #[test]
    fn summarizes_quantiles_with_r7_interpolation() {
        let mut records = Vec::new();
        for (trace_id, duration) in [10_u64, 20, 30, 40, 50].into_iter().enumerate() {
            let mut trace = base_trace(trace_id as u64);
            trace.mark_monotonic(Stage::WsFrameReceive, 100);
            trace.mark_monotonic(Stage::ParseDone, 100 + duration);
            records.push(trace);
        }

        let summary = summarize(&records).unwrap();
        let stats = &summary.intervals["frame_to_parse"];
        assert_eq!(stats.count, 5);
        assert_eq!(stats.missing, 0);
        assert_eq!(stats.invalid, 0);
        assert_eq!(stats.mean_ns, Some(30.0));
        assert_eq!(stats.p50_ns, Some(30.0));
        assert_eq!(stats.p90_ns, Some(46.0));
        assert_eq!(stats.p99_ns, Some(49.6));
        assert!((stats.p999_ns.unwrap() - 49.96).abs() < 1e-12);
        assert_eq!(stats.max_ns, Some(50));
    }

    #[test]
    fn counts_missing_and_reverse_timestamps() {
        let missing = base_trace(1);
        let mut invalid = base_trace(2);
        invalid.mark_monotonic(Stage::SignalStart, 20);
        invalid.mark_monotonic(Stage::SignalEnd, 10);

        let summary = summarize(&[missing, invalid]).unwrap();
        let stats = &summary.intervals["signal_compute"];
        assert_eq!(stats.count, 0);
        assert_eq!(stats.missing, 1);
        assert_eq!(stats.invalid, 1);
        assert_eq!(stats.p99_ns, None);
    }

    #[test]
    fn counts_trace_association_integrity() {
        let mut complete = base_trace(1);
        for (index, stage) in REQUIRED_PUBLIC_STAGES.into_iter().enumerate() {
            complete.mark_monotonic(stage, index as u64 + 1);
        }
        let mut duplicate_id = complete.clone();
        duplicate_id.mark_monotonic(Stage::Decision, 99);
        let mut incomplete = base_trace(2);
        incomplete.mark_monotonic(Stage::WsFrameReceive, 1);

        let summary = summarize_with_drops(&[complete, duplicate_id, incomplete], 3).unwrap();
        assert_eq!(summary.integrity.traces_seen, 3);
        assert_eq!(summary.integrity.unique_trace_ids, 2);
        assert_eq!(summary.integrity.duplicate_trace_ids, 1);
        assert_eq!(summary.integrity.duplicate_stage_marks, 1);
        assert_eq!(summary.integrity.complete_required, 1);
        assert_eq!(summary.integrity.incomplete_required, 1);
        assert_eq!(summary.integrity.out_of_order, 1);
        assert_eq!(summary.integrity.dropped_traces, 3);
    }

    #[test]
    fn keeps_unix_and_monotonic_domains_separate() {
        let mut trace = base_trace(1);
        trace.mark_unix(Stage::ExchangeEvent, 1_000);
        trace.mark_unix(Stage::WsFrameReceive, 1_250);
        trace.mark_monotonic(Stage::ExchangeEvent, 9_000);
        trace.mark_monotonic(Stage::WsFrameReceive, 9_001);

        let summary = summarize(&[trace]).unwrap();
        assert_eq!(summary.intervals["feed_network"].p50_ns, Some(250.0));
    }

    #[test]
    fn ndjson_round_trip_is_deterministic() {
        let mut trace = base_trace(7);
        trace.mark_unix(Stage::ExchangeEvent, 1_000);
        trace.mark_unix(Stage::WsFrameReceive, 2_000);
        trace.mark_monotonic(Stage::WsFrameReceive, 10);
        trace.mark_monotonic(Stage::ParseDone, 15);
        let records = vec![trace];

        let mut first = Vec::new();
        write_ndjson(&mut first, &records).unwrap();
        let rebuilt = read_ndjson(first.as_slice()).unwrap();
        let mut second = Vec::new();
        write_ndjson(&mut second, &rebuilt).unwrap();
        assert_eq!(first, second);
        assert_eq!(summarize(&records).unwrap(), summarize(&rebuilt).unwrap());
    }

    #[test]
    fn parses_binance_direct_and_combined_book_ticker() {
        let direct = r#"{"e":"bookTicker","u":400900217,"E":1568014460893,"T":1568014460891,"s":"BNBUSDT","b":"25.35190000","B":"31.21000000","a":"25.36520000","A":"40.66000000"}"#;
        let combined = format!(r#"{{"stream":"bnbusdt@bookTicker","data":{direct}}}"#);

        let direct_bbo = parse_binance_book_ticker(direct).unwrap();
        let combined_bbo = parse_binance_book_ticker(&combined).unwrap();
        assert_eq!(direct_bbo, combined_bbo);
        assert_eq!(direct_bbo.symbol, "BNBUSDT");
        assert_eq!(direct_bbo.exchange_sequence, Some(400900217));
    }

    #[test]
    fn parses_hyperliquid_bbo() {
        let payload = r#"{"channel":"bbo","data":{"coin":"BTC","time":1728725771363,"bbo":[{"px":"67000.0","sz":"1.2","n":3},{"px":"67000.5","sz":"0.8","n":2}]}}"#;
        let bbo = parse_hyperliquid_bbo(payload).unwrap();
        assert_eq!(bbo.symbol, "BTC");
        assert_eq!(bbo.exchange_timestamp_ms, 1_728_725_771_363);
        assert_eq!(bbo.bid_price, 67_000.0);
        assert_eq!(bbo.ask_price, 67_000.5);
    }

    #[test]
    fn parses_hyperliquid_fast_l2book_top_of_book() {
        let payload = r#"{"channel":"l2Book","data":{"coin":"BTC","time":1728725771363,"levels":[[{"px":"67000.0","sz":"1.2","n":3},{"px":"66999.5","sz":"2.0","n":4}],[{"px":"67000.5","sz":"0.8","n":2},{"px":"67001.0","sz":"1.5","n":5}]]}}"#;
        let bbo = parse_hyperliquid_message(payload).unwrap().unwrap();
        assert_eq!(bbo.symbol, "BTC");
        assert_eq!(bbo.exchange_timestamp_ms, 1_728_725_771_363);
        assert_eq!(bbo.bid_price, 67_000.0);
        assert_eq!(bbo.ask_price, 67_000.5);
    }

    #[test]
    fn ignores_hyperliquid_subscription_response_without_hiding_unknown_channels() {
        let response = r#"{"channel":"subscriptionResponse","data":{"method":"subscribe"}}"#;
        assert_eq!(parse_hyperliquid_message(response).unwrap(), None);
        let unknown = r#"{"channel":"unexpected","data":{}}"#;
        assert!(parse_hyperliquid_message(unknown).is_err());
    }

    #[test]
    fn rejects_crossed_bbo() {
        let payload = r#"{"E":1,"s":"BTCUSDT","u":2,"b":"101","a":"100"}"#;
        assert!(parse_binance_book_ticker(payload).is_err());
    }

    #[test]
    fn recorder_benchmark_reports_all_integrity_fields() {
        let summary = benchmark_recorder(10_000, 16_384, u64::MAX).unwrap();
        assert_eq!(summary.iterations, 10_000);
        assert_eq!(summary.batch_size, 1);
        assert_eq!(summary.dropped_traces, 0);
        assert!(summary.p50_ns_per_trace > 0);
        assert!(summary.p99_ns_per_trace >= summary.p50_ns_per_trace);
        assert!(summary.passed);
    }

    #[test]
    fn trace_handoff_writes_ndjson_and_flushes() {
        let path = std::env::temp_dir().join(format!(
            "latency-probe-handoff-{}-{}.ndjson",
            std::process::id(),
            ProcessClock::unix_ns().unwrap()
        ));
        let handoff = TraceHandoff::create(&path, 8).unwrap();
        assert!(handoff.try_send(base_trace(1)));
        assert!(handoff.try_send(base_trace(2)));
        assert_eq!(handoff.finish().unwrap(), 0);

        let file = File::open(&path).unwrap();
        let records = read_ndjson(std::io::BufReader::new(file)).unwrap();
        std::fs::remove_file(path).unwrap();
        assert_eq!(records.len(), 2);
        assert_eq!(records[0].trace_id, 1);
        assert_eq!(records[1].trace_id, 2);
    }

    #[test]
    fn trace_handoff_counts_full_and_disconnected_channels() {
        let (sender, receiver) = bounded(1);
        let full = TraceHandoff {
            sender: Some(sender),
            dropped: Arc::new(AtomicU64::new(0)),
            writer_thread: None,
        };
        assert!(full.try_send(base_trace(1)));
        assert!(!full.try_send(base_trace(2)));
        assert_eq!(full.dropped(), 1);
        drop(receiver);

        let (sender, receiver) = bounded(1);
        drop(receiver);
        let disconnected = TraceHandoff {
            sender: Some(sender),
            dropped: Arc::new(AtomicU64::new(0)),
            writer_thread: None,
        };
        assert!(!disconnected.try_send(base_trace(3)));
        assert_eq!(disconnected.dropped(), 1);
    }
}
