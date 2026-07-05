#!/usr/bin/env bash
set -euo pipefail

TASK_ID="0702T001"
export PATH="/home/molly/anaconda3/bin:/usr/local/bin:/usr/bin:/bin:${PATH:-}"
PROJECT_ROOT="${PROJECT_ROOT:-/home/molly/project/hftbacktest}"
PYTHON_BIN="${PYTHON_BIN:-/home/molly/anaconda3/bin/python}"
SCP_BIN="${SCP_BIN:-/usr/bin/scp}"
SHA256_BIN="${SHA256_BIN:-/usr/bin/sha256sum}"
ANALYSIS_ROOT="${PROJECT_ROOT}/local_live_analysis"
REMOTE_ROOT="${REMOTE_ROOT:-/home/admin/hft_live/hftbacktest_0627T001}"
REMOTE_ANALYSIS_ROOT="${REMOTE_ROOT}/local_live_analysis"
LOG_DIR="${PROJECT_ROOT}/logs/${TASK_ID}_local_postprocess"
CONTRACT_DIR="${ANALYSIS_ROOT}/binance_led_hyperliquid_data_contract_0601T004"
PACKAGE_DIR="${ANALYSIS_ROOT}/cross_exchange_mvp_hl_fast_sample_expansion_${TASK_ID}"

DEFAULT_SAMPLES=(
  "xemm_0702_t001_hlfast_bjt1945_a"
  "xemm_0702_t001_hlfast_bjt2015_b"
  "xemm_0702_t001_hlfast_bjt2045_c"
)

if [ -n "${SAMPLE_IDS_CSV:-}" ]; then
  IFS=',' read -r -a SAMPLES <<< "${SAMPLE_IDS_CSV}"
else
  SAMPLES=("${DEFAULT_SAMPLES[@]}")
fi

mkdir -p "${ANALYSIS_ROOT}" "${LOG_DIR}"
cd "${PROJECT_ROOT}"

log() {
  printf '[%s] %s\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" "$*" | tee -a "${LOG_DIR}/postprocess.log"
}

verify_sha() {
  local raw_file="$1"
  local sha_file="$2"
  local expected
  local actual
  expected="$(tr -d '[:space:]' < "${sha_file}")"
  actual="$("${SHA256_BIN}" "${raw_file}" | awk '{print $1}')"
  if [ "${expected}" != "${actual}" ]; then
    printf 'sha256 mismatch for %s expected=%s actual=%s\n' "${raw_file}" "${expected}" "${actual}" >&2
    return 1
  fi
}

run_logged() {
  local name="$1"
  shift
  log "run ${name}: $*"
  "$@" > "${LOG_DIR}/${name}.log" 2>&1
}

log "task=${TASK_ID} local_postprocess_start"
log "project_root=${PROJECT_ROOT}"

for sample_id in "${SAMPLES[@]}"; do
  remote_dir="${REMOTE_ANALYSIS_ROOT}/cross_exchange_public_sample_${sample_id}"
  local_dir="${ANALYSIS_ROOT}/cross_exchange_public_sample_${sample_id}"
  log "copyback sample=${sample_id} from admin@awsserver1:${remote_dir}"
  "${SCP_BIN}" -r "admin@awsserver1:${remote_dir}" "${ANALYSIS_ROOT}/" > "${LOG_DIR}/scp_${sample_id}.log" 2>&1

  "${PYTHON_BIN}" -m json.tool "${local_dir}/run_manifest.json" > /dev/null
  "${PYTHON_BIN}" -m json.tool "${local_dir}/sample_manifest.json" > /dev/null
  "${PYTHON_BIN}" -m json.tool "${local_dir}/synchronization_quality_summary.json" > /dev/null
  "${PYTHON_BIN}" - "${local_dir}/run_manifest.json" <<'PY'
import json
import sys
path = sys.argv[1]
with open(path, encoding="utf-8") as fh:
    payload = json.load(fh)
if payload.get("alignment_status") != "skipped":
    raise SystemExit(f"{path}: AWS manifest does not show skipped alignment")
if payload.get("binance_alignment", {}).get("status") != "skipped":
    raise SystemExit(f"{path}: AWS manifest does not show skipped Binance alignment")
if payload.get("hyperliquid_alignment", {}).get("status") != "skipped":
    raise SystemExit(f"{path}: AWS manifest does not show skipped Hyperliquid alignment")
if "raw_collection_only" in payload and payload.get("raw_collection_only") is not True:
    raise SystemExit(f"{path}: AWS manifest raw_collection_only is present but not true")
PY

  verify_sha "${local_dir}/binance_public_raw/raw.gz" "${local_dir}/binance_public_raw/raw.sha256"
  verify_sha "${local_dir}/hyperliquid_public_sample/raw.gz" "${local_dir}/hyperliquid_public_sample/raw.sha256"

  run_logged "binance_alignment_${sample_id}" \
    "${PYTHON_BIN}" examples/binance_tick_mm/binance_top5_provenance.py build-sidecars \
      --input-gz "${local_dir}/binance_public_raw/raw.gz" \
      --out-dir "${local_dir}/binance_alignment" \
      --sample-id "${TASK_ID}" \
      --symbol BTCUSDT \
      --tick-size 0.1 \
      --opt t \
      --buffer-size 10000000

  run_logged "hyperliquid_alignment_${sample_id}" \
    "${PYTHON_BIN}" examples/hyperliquid/hyperliquid_raw_alignment.py \
      --input-gzip "${local_dir}/hyperliquid_public_sample/raw.gz" \
      --output-dir "${local_dir}/hyperliquid_public_sample/alignment" \
      --source-label "hyperliquid_lag_public_sample_${TASK_ID}" \
      --task-id "${TASK_ID}" \
      --collection-manifest "${local_dir}/hyperliquid_public_sample/collection_manifest.json" \
      --recovery-snapshots "${local_dir}/hyperliquid_public_sample/recovery_snapshots.jsonl" \
      --buffer-size 1000000

  join_dir="${ANALYSIS_ROOT}/cross_exchange_lead_lag_join_${sample_id}"
  analysis_dir="${ANALYSIS_ROOT}/cross_exchange_lead_lag_analysis_${sample_id}"
  pricing_dir="${ANALYSIS_ROOT}/binance_led_hyperliquid_pricing_signal_${sample_id}"

  run_logged "join_${sample_id}" \
    "${PYTHON_BIN}" examples/hyperliquid/cross_exchange_lead_lag_join.py \
      --sample-dir "${local_dir}" \
      --output-dir "${join_dir}" \
      --binance-symbol BTCUSDT \
      --hyperliquid-coin BTC \
      --tick-size 0.1

  run_logged "analysis_${sample_id}" \
    "${PYTHON_BIN}" examples/hyperliquid/cross_exchange_lead_lag_analysis.py \
      --input-dir "${join_dir}" \
      --output-dir "${analysis_dir}" \
      --tick-size 0.1

  run_logged "pricing_${sample_id}" \
    "${PYTHON_BIN}" examples/hyperliquid/binance_led_pricing_signal_runner.py \
      --join-dir "${join_dir}" \
      --analysis-dir "${analysis_dir}" \
      --contract-dir "${CONTRACT_DIR}" \
      --output-dir "${pricing_dir}" \
      --tick-size 0.1
done

sample_args=()
for sample_id in "${SAMPLES[@]}"; do
  sample_args+=(--sample-id "${sample_id}")
done

run_logged "sample_expansion_${TASK_ID}" \
  "${PYTHON_BIN}" examples/hyperliquid/cross_exchange_sample_expansion.py \
    --analysis-root "${ANALYSIS_ROOT}" \
    --task-id "${TASK_ID}" \
    --output-dir "${PACKAGE_DIR}" \
    "${sample_args[@]}"

"${PYTHON_BIN}" -m json.tool "${PACKAGE_DIR}/sample_expansion_manifest.json" > /dev/null
log "task=${TASK_ID} local_postprocess_completed package=${PACKAGE_DIR}"
