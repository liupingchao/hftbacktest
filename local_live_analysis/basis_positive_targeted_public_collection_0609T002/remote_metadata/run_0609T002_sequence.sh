#!/usr/bin/env bash
set -u
WORKTREE=/home/admin/hft_live/worktrees/0609T002-basis-positive-targeted-public
RUN_ROOT=/home/admin/hft_live/runs/0609T002-basis-positive-targeted-public
PY=/home/admin/hft_live/venv/bin/python
DURATION=1800
TASK_ID=0609T002
SAMPLES=(xemm_0609_normal_a xemm_0609_normal_b xemm_0609_active_a xemm_0609_active_b)
cd "$WORKTREE"
printf "sample_id\tstart_time\tend_time\texit_code\toutput_dir\n" > "$RUN_ROOT/sequence_status.tsv"
for sample in "${SAMPLES[@]}"; do
  out="$RUN_ROOT/$sample"
  start=$(date -Is)
  echo "START $sample $start" | tee -a "$RUN_ROOT/sequence.log"
  "$PY" examples/hyperliquid/synchronized_public_collection.py collect \
    --output-dir "$out" \
    --duration-seconds "$DURATION" \
    --task-id "$TASK_ID" \
    --clean-output \
    > "$RUN_ROOT/${sample}.stdout.log" \
    2> "$RUN_ROOT/${sample}.stderr.log"
  rc=$?
  end=$(date -Is)
  printf "%s\t%s\t%s\t%s\t%s\n" "$sample" "$start" "$end" "$rc" "$out" >> "$RUN_ROOT/sequence_status.tsv"
  echo "END $sample $end rc=$rc" | tee -a "$RUN_ROOT/sequence.log"
done
