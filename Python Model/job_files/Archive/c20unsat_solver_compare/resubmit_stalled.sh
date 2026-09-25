#!/bin/bash
set -e
cd /projects/anth4580/Bayesian/job_files/c20unsat_solver_compare

NEW_IDS_FILE="job_ids_retry.txt"
> "$NEW_IDS_FILE"

resubmit_chain() {
  local K="$1"; local start_index="$2"
  local prev_id=""
  for i in $(seq "$start_index" 9); do
    if [ -z "$prev_id" ]; then
      out=$(sbatch -M blanca --job-name="c20cmp_${K}_${i}_retry" solve_one_condition.sbatch "$K" "$i")
    else
      out=$(sbatch -M blanca --dependency=afterok:$prev_id --job-name="c20cmp_${K}_${i}_retry" solve_one_condition.sbatch "$K" "$i")
    fi
    job_id=$(echo "$out" | awk '{print $4}')
    echo "$K sweep$i -> job $job_id (dep: ${prev_id:-none})"
    echo "$K sweep$i $job_id" >> "$NEW_IDS_FILE"
    prev_id="$job_id"
  done
}

resubmit_chain A 3
resubmit_chain B 5
resubmit_chain C 5
resubmit_chain D 6
resubmit_chain E 6

echo "=== new final job IDs per chain ==="
grep " sweep9 " "$NEW_IDS_FILE"
