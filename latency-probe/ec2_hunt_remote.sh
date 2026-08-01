#!/bin/bash
set -uo pipefail

usage() {
  echo "usage: $0 setup <binary-get-url> <binary-sha256> | run <archive-put-url> <duration-seconds> <warmup-messages> <label>" >&2
  exit 2
}

instance_metadata() {
  local path=$1
  local token
  token=$(curl -fsS -X PUT \
    -H "X-aws-ec2-metadata-token-ttl-seconds: 21600" \
    http://169.254.169.254/latest/api/token)
  curl -fsS -H "X-aws-ec2-metadata-token: $token" \
    "http://169.254.169.254/latest/meta-data/$path"
}

capture_clock() {
  local output=$1
  local attempts=$2
  local bound_limit_us=$3
  local tracking captured_at offset_s offset_us root_delay_s root_dispersion_s
  local error_bound_us leap gate
  for _ in $(seq 1 "$attempts"); do
    tracking=$(chronyc tracking 2>/dev/null || true)
    captured_at=$(date -u +%Y-%m-%dT%H:%M:%S.%NZ)
    leap=$(printf '%s\n' "$tracking" | awk -F': *' '/^Leap status/ {print $2}')
    offset_s=$(printf '%s\n' "$tracking" | awk -F': *' '/^System time/ {print $2}' | awk '{print $1}')
    root_delay_s=$(printf '%s\n' "$tracking" | awk -F': *' '/^Root delay/ {print $2}' | awk '{print $1}')
    root_dispersion_s=$(printf '%s\n' "$tracking" | awk -F': *' '/^Root dispersion/ {print $2}' | awk '{print $1}')
    gate=fail
    if [ "$leap" = "Normal" ] && [ -n "$offset_s" ] &&
       [ -n "$root_delay_s" ] && [ -n "$root_dispersion_s" ]; then
      read -r offset_us error_bound_us <<EOF
$(awk -v offset="$offset_s" -v delay="$root_delay_s" -v dispersion="$root_dispersion_s" 'BEGIN {
  if (offset < 0) offset = -offset;
  printf("%.3f %.3f\n", offset * 1000000, (offset + 0.5 * delay + dispersion) * 1000000)
}')
EOF
      if awk -v value="$error_bound_us" -v limit="$bound_limit_us" \
        'BEGIN { exit !(value <= limit) }'; then
        gate=pass
      fi
    fi
    jq -n \
      --arg captured_at_utc "$captured_at" \
      --arg leap_status "${leap:-unknown}" \
      --arg tracking "$tracking" \
      --arg gate "$gate" \
      --argjson system_offset_us "${offset_us:-null}" \
      --argjson root_delay_us "$(awk -v value="${root_delay_s:-0}" 'BEGIN { printf("%.3f", value * 1000000) }')" \
      --argjson root_dispersion_us "$(awk -v value="${root_dispersion_s:-0}" 'BEGIN { printf("%.3f", value * 1000000) }')" \
      --argjson error_bound_us "${error_bound_us:-null}" \
      --argjson error_bound_limit_us "$bound_limit_us" \
      '{
        captured_at_utc: $captured_at_utc,
        leap_status: $leap_status,
        system_offset_us: $system_offset_us,
        root_delay_us: $root_delay_us,
        root_dispersion_us: $root_dispersion_us,
        error_bound_us: $error_bound_us,
        error_bound_limit_us: $error_bound_limit_us,
        gate: $gate,
        chrony_tracking: $tracking
      }' > "$output"
    if [ "$gate" = "pass" ]; then
      return 0
    fi
    sleep 5
  done
  return 1
}

setup_host() {
  [ "$#" -eq 2 ] || usage
  local binary_url=$1
  local expected_sha=$2
  local setup_log=/tmp/ec2-hunt-setup.log

  export DEBIAN_FRONTEND=noninteractive
  : > "$setup_log"
  apt-get update >> "$setup_log" 2>&1 || {
    tail -n 100 "$setup_log" >&2
    return 1
  }
  apt-get install -y ca-certificates chrony curl ethtool gzip jq util-linux \
    >> "$setup_log" 2>&1 || {
      tail -n 100 "$setup_log" >&2
      return 1
    }

  systemctl disable --now systemd-timesyncd 2>/dev/null || true
  cat > /etc/chrony/chrony.conf <<'EOF'
server 169.254.169.123 prefer iburst minpoll 4 maxpoll 4
driftfile /var/lib/chrony/chrony.drift
makestep 0.1 3
rtcsync
logchange 0.1
EOF
  systemctl enable chrony >> "$setup_log" 2>&1 || {
    tail -n 100 "$setup_log" >&2
    return 1
  }
  systemctl restart chrony >> "$setup_log" 2>&1 || {
    tail -n 100 "$setup_log" >&2
    return 1
  }

  install -d -o admin -g admin /home/admin/latency-hunt
  curl -fsS "$binary_url" -o /home/admin/latency-hunt/latency-probe
  chmod 0755 /home/admin/latency-hunt/latency-probe
  chown admin:admin /home/admin/latency-hunt/latency-probe

  local actual_sha
  actual_sha=$(sha256sum /home/admin/latency-hunt/latency-probe | awk '{print $1}')
  if [ "$actual_sha" != "$expected_sha" ]; then
    echo "binary hash mismatch: expected=$expected_sha actual=$actual_sha" >&2
    exit 1
  fi
  printf '%s\n' "$actual_sha" > /home/admin/latency-hunt/binary-sha256.txt
  chown admin:admin /home/admin/latency-hunt/binary-sha256.txt

  local clock_status=0
  capture_clock /home/admin/latency-hunt/setup-clock.json 180 750 || clock_status=1
  cp "$setup_log" /home/admin/latency-hunt/setup.log
  chown -R admin:admin /home/admin/latency-hunt
  cat /home/admin/latency-hunt/setup-clock.json || return 1
  return "$clock_status"
}

write_metadata() {
  local output=$1
  local clock_evidence=$2
  local iface instance_id instance_type az
  iface=$(ip route show default | awk '{print $5; exit}')
  instance_id=$(instance_metadata instance-id)
  instance_type=$(instance_metadata instance-type)
  az=$(instance_metadata placement/availability-zone)

  jq -n \
    --arg instance_id "$instance_id" \
    --arg instance_type "$instance_type" \
    --arg availability_zone "$az" \
    --arg kernel "$(uname -r)" \
    --arg cpu_model "$(awk -F': *' '/model name/ {print $2; exit}' /proc/cpuinfo)" \
    --arg clocksource "$(cat /sys/devices/system/clocksource/clocksource0/current_clocksource 2>/dev/null || true)" \
    --arg interface "$iface" \
    --arg ethtool_driver "$(ethtool -i "$iface" 2>/dev/null || true)" \
    --arg timestamp_capabilities "$(ethtool -T "$iface" 2>/dev/null || true)" \
    --arg offloads "$(ethtool -k "$iface" 2>/dev/null || true)" \
    --arg channels "$(ethtool -l "$iface" 2>/dev/null || true)" \
    --slurpfile clock_evidence "$clock_evidence" \
    '{
      instance_id: $instance_id,
      instance_type: $instance_type,
      availability_zone: $availability_zone,
      kernel: $kernel,
      cpu_model: $cpu_model,
      clocksource: $clocksource,
      interface: $interface,
      ethtool_driver: $ethtool_driver,
      timestamp_capabilities: $timestamp_capabilities,
      offloads: $offloads,
      channels: $channels,
      clock_evidence: $clock_evidence[0]
    }' > "$output"
}

upload_archive() {
  [ "$#" -eq 4 ] || usage
  local archive=$1
  local upload_url=$2
  local label=$3
  local instance_id=$4
  local archive_sha archive_bytes upload_status=0
  archive_sha=$(sha256sum "$archive" | awk '{print $1}')
  archive_bytes=$(stat -c %s "$archive")
  curl -fsS -X PUT --upload-file "$archive" "$upload_url" || upload_status=1
  jq -cn \
    --arg label "$label" \
    --arg instance_id "$instance_id" \
    --arg archive_sha256 "$archive_sha" \
    --argjson archive_bytes "$archive_bytes" \
    --argjson upload_status "$upload_status" \
    '{
      label: $label,
      instance_id: $instance_id,
      archive_sha256: $archive_sha256,
      archive_bytes: $archive_bytes,
      upload_status: $upload_status
    }'
  return "$upload_status"
}

run_probe() {
  [ "$#" -eq 4 ] || usage
  local upload_url=$1
  local duration=$2
  local warmup=$3
  local label=$4
  local base=/home/admin/latency-hunt
  local binary=$base/latency-probe
  local instance_id
  local output=$base/$label
  local binance_status=99 hyperliquid_status=99 benchmark_status=99 rebuild_status=99
  local clock_before_status=0 clock_after_status=0

  instance_id=$(instance_metadata instance-id)
  rm -rf "$output"
  install -d -o admin -g admin "$output"
  capture_clock "$output/clock-before.json" 180 750 || clock_before_status=1
  write_metadata "$output/metadata-before.json" "$output/clock-before.json"

  if [ "$clock_before_status" -eq 0 ]; then
    sudo -u admin "$binary" benchmark --iterations 1000000 --max-p99-ns 5000 \
      > "$output/benchmark.json" 2> "$output/benchmark.stderr"
    benchmark_status=$?

    sudo -u admin taskset -c 0,1 "$binary" public \
      --venue binance-futures --symbol BTCUSDT \
      --duration-seconds "$duration" --warmup-messages "$warmup" \
      --raw-output "$output/binance.ndjson" \
      --summary-output "$output/binance-summary.json" \
      > "$output/binance.stdout" 2> "$output/binance.stderr" &
    local binance_pid=$!

    sudo -u admin taskset -c 2,3 "$binary" public \
      --venue hyperliquid --symbol BTC \
      --duration-seconds "$duration" --warmup-messages "$warmup" \
      --raw-output "$output/hyperliquid.ndjson" \
      --summary-output "$output/hyperliquid-summary.json" \
      > "$output/hyperliquid.stdout" 2> "$output/hyperliquid.stderr" &
    local hyperliquid_pid=$!

    wait "$binance_pid"; binance_status=$?
    wait "$hyperliquid_pid"; hyperliquid_status=$?

    rebuild_status=0
    if [ "$binance_status" -eq 0 ]; then
      sudo -u admin "$binary" summarize \
        --raw-input "$output/binance.ndjson" \
        --summary-output "$output/binance-summary-rebuilt.json" || rebuild_status=1
      cmp -s "$output/binance-summary.json" "$output/binance-summary-rebuilt.json" || rebuild_status=1
    fi
    if [ "$hyperliquid_status" -eq 0 ]; then
      sudo -u admin "$binary" summarize \
        --raw-input "$output/hyperliquid.ndjson" \
        --summary-output "$output/hyperliquid-summary-rebuilt.json" || rebuild_status=1
      cmp -s "$output/hyperliquid-summary.json" "$output/hyperliquid-summary-rebuilt.json" || rebuild_status=1
    fi
  fi

  capture_clock "$output/clock-after.json" 36 750 || clock_after_status=1
  write_metadata "$output/metadata-after.json" "$output/clock-after.json"
  sha256sum "$binary" "$output"/* > "$output/sha256sums.txt" 2>/dev/null || true
  jq -n \
    --arg label "$label" \
    --argjson duration_seconds "$duration" \
    --argjson warmup_messages "$warmup" \
    --argjson benchmark_status "$benchmark_status" \
    --argjson binance_status "$binance_status" \
    --argjson hyperliquid_status "$hyperliquid_status" \
    --argjson rebuild_status "$rebuild_status" \
    --argjson clock_before_status "$clock_before_status" \
    --argjson clock_after_status "$clock_after_status" \
    '{
      label: $label,
      duration_seconds: $duration_seconds,
      warmup_messages: $warmup_messages,
      benchmark_status: $benchmark_status,
      binance_status: $binance_status,
      hyperliquid_status: $hyperliquid_status,
      rebuild_status: $rebuild_status,
      clock_before_status: $clock_before_status,
      clock_after_status: $clock_after_status
    }' > "$output/run-status.json"

  local archive=$base/${label}.tar.gz
  local upload_status=0 upload_receipt
  tar -C "$base" -czf "$archive" "$label"
  upload_receipt=$(upload_archive "$archive" "$upload_url" "$label" "$instance_id") \
    || upload_status=$?
  printf '%s\n' "$upload_receipt"

  if [ "$benchmark_status" -ne 0 ] || [ "$binance_status" -ne 0 ] ||
     [ "$hyperliquid_status" -ne 0 ] || [ "$rebuild_status" -ne 0 ] ||
     [ "$clock_before_status" -ne 0 ] || [ "$clock_after_status" -ne 0 ] ||
     [ "$upload_status" -ne 0 ]; then
    exit 1
  fi
}

if [ "${BASH_SOURCE[0]}" = "$0" ]; then
  mode=${1:-}
  shift || true
  case "$mode" in
    setup) setup_host "$@" ;;
    run) run_probe "$@" ;;
    *) usage ;;
  esac
fi
