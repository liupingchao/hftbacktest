#!/usr/bin/env bash
set -euo pipefail

TASK_ID="0702T001"
REMOTE_ROOT="${REMOTE_ROOT:-/home/admin/hft_live/hftbacktest_0627T001}"
PYTHON_BIN="${PYTHON_BIN:-/home/admin/hft_live/venv/bin/python}"
ANALYSIS_ROOT="${REMOTE_ROOT}/local_live_analysis"
LOG_DIR="${REMOTE_ROOT}/logs/${TASK_ID}"
STATUS_LOG="${LOG_DIR}/aws_collect_status.log"

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

mkdir -p "${LOG_DIR}" "${ANALYSIS_ROOT}"
cd "${REMOTE_ROOT}"

{
  printf 'task_id=%s\n' "${TASK_ID}"
  printf 'remote_root=%s\n' "${REMOTE_ROOT}"
  printf 'start_utc=%s\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
  printf 'start_local=%s\n' "$(date '+%Y-%m-%dT%H:%M:%S%z')"
  printf 'mode=raw_public_collection_only\n'
  printf 'alignment_policy=skip_on_aws_defer_to_amdserver\n'
} > "${STATUS_LOG}"

for sample_id in "${SAMPLES[@]}"; do
  sample_dir="${ANALYSIS_ROOT}/cross_exchange_public_sample_${sample_id}"
  sample_log="${LOG_DIR}/${sample_id}.log"
  printf '\n[%s] sample_start sample_id=%s\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" "${sample_id}" | tee -a "${STATUS_LOG}"

  if "${PYTHON_BIN}" examples/hyperliquid/synchronized_public_collection.py collect \
      --output-dir "${sample_dir}" \
      --duration-seconds 1800 \
      --binance-symbol BTCUSDT \
      --hyperliquid-coin BTC \
      --hyperliquid-l2book-fast \
      --task-id "${TASK_ID}" \
      --clean-output \
      --skip-alignment > "${sample_log}" 2>&1; then
    rc=0
  else
    rc=$?
  fi

  printf '[%s] sample_done sample_id=%s rc=%s log=%s\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" "${sample_id}" "${rc}" "${sample_log}" | tee -a "${STATUS_LOG}"
  if [ "${rc}" -ne 0 ]; then
    exit "${rc}"
  fi

  "${PYTHON_BIN}" -m json.tool "${sample_dir}/run_manifest.json" > /dev/null
  "${PYTHON_BIN}" -m json.tool "${sample_dir}/sample_manifest.json" > /dev/null
  "${PYTHON_BIN}" -m json.tool "${sample_dir}/synchronization_quality_summary.json" > /dev/null
  "${PYTHON_BIN}" - "${sample_dir}/run_manifest.json" <<'PY'
import json
import sys
path = sys.argv[1]
with open(path, encoding="utf-8") as fh:
    payload = json.load(fh)
if payload.get("alignment_status") != "skipped":
    raise SystemExit(f"{path}: alignment_status is not skipped")
if payload.get("alignment_execution_host") != "macmini_or_amdserver":
    raise SystemExit(f"{path}: unexpected alignment_execution_host")
if payload.get("binance_alignment", {}).get("status") != "skipped":
    raise SystemExit(f"{path}: binance_alignment status is not skipped")
if payload.get("hyperliquid_alignment", {}).get("status") != "skipped":
    raise SystemExit(f"{path}: hyperliquid_alignment status is not skipped")
if "raw_collection_only" in payload and payload.get("raw_collection_only") is not True:
    raise SystemExit(f"{path}: raw_collection_only is present but not true")
PY
done

{
  printf '\nend_utc=%s\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
  printf 'end_local=%s\n' "$(date '+%Y-%m-%dT%H:%M:%S%z')"
  printf 'status=completed\n'
} >> "${STATUS_LOG}"
