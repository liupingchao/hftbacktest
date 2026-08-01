use std::{
    fs::File,
    io::{BufReader, BufWriter, Write},
    net::{IpAddr, SocketAddr},
    path::{Path, PathBuf},
    sync::Arc,
    time::Duration,
};

use anyhow::{Context, Result, bail};
use clap::{Args, Parser, Subcommand, ValueEnum};
use futures_util::{SinkExt, StreamExt};
use latency_probe::{
    ProcessClock, Stage, Summary, TraceHandoff, TraceRecord, Venue, benchmark_recorder,
    parse_binance_book_ticker, parse_hyperliquid_message, read_ndjson, summarize_with_drops,
};
use serde::Serialize;
use tokio::{
    net::{TcpStream, UdpSocket},
    time::Instant,
};
use tokio_tungstenite::{
    Connector, MaybeTlsStream, WebSocketStream, client_async_tls_with_config, connect_async,
    connect_async_tls_with_config, tungstenite::Message,
};

#[derive(Debug, Parser)]
#[command(about = "Binance/Hyperliquid unified latency probe")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    /// Capture public BBO messages and measure the local processing pipeline.
    Public(PublicArgs),
    /// Rebuild deterministic statistics from raw NDJSON traces.
    Summarize(SummarizeArgs),
    /// Generate deterministic traces for measurement-system verification.
    Synthetic(SyntheticArgs),
    /// Measure fixed stage marking plus bounded handoff overhead.
    Benchmark(BenchmarkArgs),
}

#[derive(Clone, Copy, Debug, ValueEnum)]
enum ProbeVenue {
    BinanceFutures,
    Hyperliquid,
}

impl From<ProbeVenue> for Venue {
    fn from(value: ProbeVenue) -> Self {
        match value {
            ProbeVenue::BinanceFutures => Self::BinanceFutures,
            ProbeVenue::Hyperliquid => Self::Hyperliquid,
        }
    }
}

#[derive(Debug, Args)]
struct PublicArgs {
    #[arg(long, value_enum)]
    venue: ProbeVenue,
    #[arg(long)]
    symbol: String,
    #[arg(long, default_value_t = 30)]
    duration_seconds: u64,
    #[arg(long, default_value_t = 0)]
    warmup_messages: u64,
    #[arg(long)]
    max_messages: Option<u64>,
    #[arg(long)]
    connect_ip: Option<IpAddr>,
    #[arg(long)]
    tls12_only: bool,
    #[arg(long)]
    raw_output: PathBuf,
    #[arg(long)]
    summary_output: PathBuf,
    #[arg(long, default_value_t = 65_536)]
    queue_capacity: usize,
}

#[derive(Debug, Args)]
struct SummarizeArgs {
    #[arg(long)]
    raw_input: PathBuf,
    #[arg(long)]
    summary_output: PathBuf,
    #[arg(long, default_value_t = 0)]
    dropped_traces: u64,
}

#[derive(Debug, Args)]
struct SyntheticArgs {
    #[arg(long, default_value_t = 1000)]
    traces: u64,
    #[arg(long)]
    raw_output: PathBuf,
    #[arg(long)]
    summary_output: PathBuf,
    #[arg(long, default_value_t = 65_536)]
    queue_capacity: usize,
}

#[derive(Debug, Args)]
struct BenchmarkArgs {
    #[arg(long, default_value_t = 1_000_000)]
    iterations: u64,
    #[arg(long, default_value_t = 65_536)]
    queue_capacity: usize,
    #[arg(long, default_value_t = 5_000)]
    max_p99_ns: u64,
}

#[derive(Debug, Serialize)]
struct ProbeOrderIntent<'a> {
    symbol: &'a str,
    side: &'static str,
    price: f64,
    observed_spread: f64,
    trace_id: u64,
}

#[derive(Debug, Default)]
struct BboState {
    bid_price: f64,
    ask_price: f64,
}

impl BboState {
    fn apply(&mut self, bbo: &latency_probe::PublicBbo) {
        self.bid_price = bbo.bid_price;
        self.ask_price = bbo.ask_price;
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Command::Public(args) => run_public(args).await,
        Command::Summarize(args) => run_summarize(args),
        Command::Synthetic(args) => run_synthetic(args),
        Command::Benchmark(args) => run_benchmark(args),
    }
}

async fn run_public(args: PublicArgs) -> Result<()> {
    if args.duration_seconds == 0 {
        bail!("--duration-seconds must be positive");
    }
    if args.max_messages == Some(0) {
        bail!("--max-messages must be positive when supplied");
    }

    let clock = ProcessClock::default();
    let (url, channel) = public_endpoint(args.venue, &args.symbol);
    let mut websocket = connect_public_websocket(&url, args.connect_ip, args.tls12_only).await?;
    if matches!(args.venue, ProbeVenue::Hyperliquid) {
        let subscription = serde_json::json!({
            "method": "subscribe",
            "subscription": {"type": "l2Book", "coin": args.symbol, "fast": true}
        });
        websocket
            .send(Message::Text(subscription.to_string().into()))
            .await
            .context("send Hyperliquid fast l2Book subscription")?;
    }

    let receiver = UdpSocket::bind("127.0.0.1:0")
        .await
        .context("bind loopback UDP receiver")?;
    let sender = UdpSocket::bind("127.0.0.1:0")
        .await
        .context("bind loopback UDP sender")?;
    sender
        .connect(receiver.local_addr().context("read UDP receiver address")?)
        .await
        .context("connect loopback UDP sender")?;

    let drain_task = tokio::spawn(async move {
        let mut sink_buffer = [0_u8; 4096];
        while receiver.recv_from(&mut sink_buffer).await.is_ok() {}
    });
    let handoff = TraceHandoff::create(&args.raw_output, args.queue_capacity)?;
    let mut deadline = Instant::now() + Duration::from_secs(args.duration_seconds);
    let mut warmup_remaining = args.warmup_messages;
    let mut accepted = 0_u64;
    let mut bbo_state = BboState::default();

    while Instant::now() < deadline {
        if args.max_messages.is_some_and(|limit| accepted >= limit) {
            break;
        }

        let remaining = deadline.saturating_duration_since(Instant::now());
        let message = match tokio::time::timeout(remaining, websocket.next()).await {
            Ok(Some(message)) => message,
            Ok(None) | Err(_) => break,
        };
        let message = message.context("read public WebSocket message")?;
        if !message.is_text() {
            continue;
        }

        let frame_unix_ns = ProcessClock::unix_ns()?;
        let frame_monotonic_ns = clock.monotonic_ns();
        let payload = message.into_text().context("decode WebSocket text frame")?;
        let bbo = match args.venue {
            ProbeVenue::BinanceFutures => parse_binance_book_ticker(&payload)?,
            ProbeVenue::Hyperliquid => match parse_hyperliquid_message(&payload)? {
                Some(bbo) => bbo,
                None => continue,
            },
        };

        if warmup_remaining > 0 {
            warmup_remaining -= 1;
            if warmup_remaining == 0 {
                deadline = Instant::now() + Duration::from_secs(args.duration_seconds);
            }
            continue;
        }

        let trace_id = accepted + 1;
        let mut trace = TraceRecord::new(args.venue.into(), channel, &bbo.symbol, trace_id);
        trace.exchange_sequence = bbo.exchange_sequence;
        trace.mark_unix(
            Stage::ExchangeEvent,
            bbo.exchange_timestamp_ms.saturating_mul(1_000_000),
        );
        trace.mark_unix(Stage::WsFrameReceive, frame_unix_ns);
        trace.mark_monotonic(Stage::WsFrameReceive, frame_monotonic_ns);
        trace.mark_monotonic(Stage::ParseDone, clock.monotonic_ns());

        bbo_state.apply(&bbo);
        trace.mark_monotonic(Stage::BookApplyDone, clock.monotonic_ns());
        trace.mark_monotonic(Stage::SignalStart, clock.monotonic_ns());
        let midpoint = (bbo_state.bid_price + bbo_state.ask_price) * 0.5;
        let spread = bbo_state.ask_price - bbo_state.bid_price;
        trace.mark_monotonic(Stage::SignalEnd, clock.monotonic_ns());
        let side = if trace_id.is_multiple_of(2) {
            "buy"
        } else {
            "sell"
        };
        let price = if side == "buy" {
            bbo_state.bid_price
        } else {
            bbo_state.ask_price
        };
        trace.mark_monotonic(Stage::Decision, clock.monotonic_ns());
        let intent = ProbeOrderIntent {
            symbol: &bbo.symbol,
            side,
            price: if midpoint.is_finite() {
                price
            } else {
                bbo_state.bid_price
            },
            observed_spread: spread,
            trace_id,
        };
        let encoded = serde_json::to_vec(&intent).context("encode local probe intent")?;
        trace.mark_monotonic(Stage::OrderEncodeDone, clock.monotonic_ns());
        sender
            .send(&encoded)
            .await
            .context("write local probe intent to UDP socket")?;
        trace.mark_monotonic(Stage::SocketSend, clock.monotonic_ns());
        handoff.try_send(trace);
        accepted += 1;
    }

    drain_task.abort();
    let _ = drain_task.await;
    let dropped = handoff.finish()?;
    let input = File::open(&args.raw_output)
        .with_context(|| format!("open {}", args.raw_output.display()))?;
    let records = read_ndjson(BufReader::new(input))?;
    let summary = summarize_with_drops(&records, dropped)?;
    validate_public_integrity(&summary, accepted)?;
    write_summary_file(&args.summary_output, &summary)?;
    eprintln!(
        "captured {} traces with {} drops into {}",
        records.len(),
        dropped,
        args.raw_output.display()
    );
    Ok(())
}

fn run_summarize(args: SummarizeArgs) -> Result<()> {
    let input = File::open(&args.raw_input)
        .with_context(|| format!("open {}", args.raw_input.display()))?;
    let records = read_ndjson(BufReader::new(input))?;
    write_summary_file(
        &args.summary_output,
        &summarize_with_drops(&records, args.dropped_traces)?,
    )
}

fn run_synthetic(args: SyntheticArgs) -> Result<()> {
    if args.traces == 0 {
        bail!("--traces must be positive");
    }
    let handoff = TraceHandoff::create(&args.raw_output, args.queue_capacity)?;
    for index in 0..args.traces {
        handoff.try_send(synthetic_record(index));
    }
    let dropped = handoff.finish()?;
    let input = File::open(&args.raw_output)
        .with_context(|| format!("open {}", args.raw_output.display()))?;
    let records = read_ndjson(BufReader::new(input))?;
    let summary = summarize_with_drops(&records, dropped)?;
    validate_public_integrity(&summary, args.traces)?;
    write_summary_file(&args.summary_output, &summary)
}

fn run_benchmark(args: BenchmarkArgs) -> Result<()> {
    let summary = benchmark_recorder(args.iterations, args.queue_capacity, args.max_p99_ns)?;
    let mut output = BufWriter::new(std::io::stdout().lock());
    serde_json::to_writer_pretty(&mut output, &summary).context("serialize benchmark summary")?;
    output.write_all(b"\n")?;
    output.flush()?;
    if !summary.passed {
        bail!(
            "benchmark failed: p99={}ns dropped={} budget={}ns",
            summary.p99_ns_per_trace,
            summary.dropped_traces,
            summary.max_p99_ns
        );
    }
    Ok(())
}

fn synthetic_record(index: u64) -> TraceRecord {
    let venue = if index.is_multiple_of(2) {
        Venue::BinanceFutures
    } else {
        Venue::Hyperliquid
    };
    let mut trace = TraceRecord::new(venue, "synthetic_bbo", "BTC", index + 1);
    trace.exchange_sequence = Some(index + 10_000);
    let unix_base = 1_700_000_000_000_000_000_u64 + index * 1_000_000;
    trace.mark_unix(Stage::ExchangeEvent, unix_base);
    trace.mark_unix(
        Stage::WsFrameReceive,
        unix_base + 500_000 + (index % 50) * 1_000,
    );

    let mono_base = index * 2_000_000;
    for (stage_index, stage) in latency_probe::REQUIRED_PUBLIC_STAGES
        .into_iter()
        .enumerate()
    {
        trace.mark_monotonic(
            stage,
            mono_base + (stage_index as u64 + 1) * 10_000 + index % 101,
        );
    }
    trace
}

fn public_endpoint(venue: ProbeVenue, symbol: &str) -> (String, &'static str) {
    match venue {
        ProbeVenue::BinanceFutures => (
            format!(
                "wss://fstream.binance.com/ws/{}@bookTicker",
                symbol.to_ascii_lowercase()
            ),
            "bookTicker",
        ),
        ProbeVenue::Hyperliquid => ("wss://api.hyperliquid.xyz/ws".to_owned(), "l2Book_fast"),
    }
}

async fn connect_public_websocket(
    url: &str,
    connect_ip: Option<IpAddr>,
    tls12_only: bool,
) -> Result<WebSocketStream<MaybeTlsStream<TcpStream>>> {
    let connector = tls12_only.then(tls12_connector).transpose()?;
    let result = match (connect_ip, connector) {
        (Some(ip), connector) => {
            let stream = TcpStream::connect(SocketAddr::new(ip, 443))
                .await
                .with_context(|| format!("connect {ip}:443 for {url}"))?;
            client_async_tls_with_config(url, stream, None, connector).await
        }
        (None, Some(connector)) => {
            connect_async_tls_with_config(url, None, false, Some(connector)).await
        }
        (None, None) => connect_async(url).await,
    };
    result
        .map(|(websocket, _)| websocket)
        .with_context(|| format!("connect public WebSocket {url}"))
}

fn tls12_connector() -> Result<Connector> {
    let roots = rustls::RootCertStore::from_iter(webpki_roots::TLS_SERVER_ROOTS.iter().cloned());
    let provider = Arc::new(rustls::crypto::ring::default_provider());
    let config = rustls::ClientConfig::builder_with_provider(provider)
        .with_protocol_versions(&[&rustls::version::TLS12])
        .context("configure TLS 1.2 protocol")?
        .with_root_certificates(roots)
        .with_no_client_auth();
    Ok(Connector::Rustls(Arc::new(config)))
}

fn validate_public_integrity(summary: &Summary, attempted: u64) -> Result<()> {
    let integrity = &summary.integrity;
    if attempted == 0 {
        bail!("public probe completed without a valid BBO trace");
    }
    if integrity.complete_required != attempted
        || integrity.incomplete_required != 0
        || integrity.out_of_order != 0
        || integrity.duplicate_trace_ids != 0
        || integrity.duplicate_stage_marks != 0
        || integrity.dropped_traces != 0
    {
        bail!("trace integrity acceptance failed: {integrity:?}");
    }
    Ok(())
}

fn write_summary_file(path: &Path, summary: &Summary) -> Result<()> {
    let output = File::create(path).with_context(|| format!("create {}", path.display()))?;
    let mut writer = BufWriter::new(output);
    serde_json::to_writer_pretty(&mut writer, summary).context("serialize summary")?;
    writer.write_all(b"\n").context("write summary newline")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_zero_valid_public_traces() {
        let summary = summarize_with_drops(&[], 0).unwrap();
        assert!(validate_public_integrity(&summary, 0).is_err());
    }

    #[test]
    fn parses_public_warmup_messages() {
        let cli = Cli::try_parse_from([
            "latency-probe",
            "public",
            "--venue",
            "binance-futures",
            "--symbol",
            "BTCUSDT",
            "--duration-seconds",
            "900",
            "--warmup-messages",
            "100",
            "--raw-output",
            "raw.ndjson",
            "--summary-output",
            "summary.json",
        ])
        .unwrap();
        let Command::Public(args) = cli.command else {
            panic!("expected public command");
        };
        assert_eq!(args.warmup_messages, 100);
        assert_eq!(args.duration_seconds, 900);
    }
}
