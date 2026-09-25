#!/bin/bash
set -e
cd /projects/anth4580/Bayesian/job_files/c20unsat_solver_compare
JOBIDS_FILE="job_ids.txt"
> "$JOBIDS_FILE"
for K in A B C D E; do
  prev_id=""
  for i in 1 2 3 4 5 6 7 8 9; do
    if [ -z "$prev_id" ]; then
      out=$(sbatch -M blanca --job-name="c20cmp_${K}_${i}" solve_one_condition.sbatch "$K" "$i")
    else
      out=$(sbatch -M blanca --dependency=afterok:$prev_id --job-name="c20cmp_${K}_${i}" solve_one_condition.sbatch "$K" "$i")
    fi
    job_id=$(echo "$out" | awk '{print $4}')
    echo "$K sweep$i -> job $job_id (dep: ${prev_id:-none})"
    echo "$K sweep$i $job_id" >> "$JOBIDS_FILE"
    prev_id="$job_id"
  done
done
echo "Submitted 45 jobs across 5 chains. IDs in $JOBIDS_FILE"
