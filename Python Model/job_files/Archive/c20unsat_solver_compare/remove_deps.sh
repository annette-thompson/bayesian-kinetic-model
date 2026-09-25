#!/bin/bash
cd /projects/anth4580/Bayesian/job_files/c20unsat_solver_compare

DEPENDENT_JOBS="28087634 28087635 28087636 28087637 28087638 28087639 28087641 28087642 28087643 28087644 28087650 28087651 28087652 28087653 28087655 28087656 28087657 28087659 28087660 28087661"

echo "cancelling pending dependent jobs..."
scancel -M blanca $DEPENDENT_JOBS 2>&1

declare -A KI_MAP=(
  [A]="4 5 6 7 8 9"
  [B]="6 7 8 9"
  [C]="6 7 8 9"
  [D]="7 8 9"
  [E]="7 8 9"
)

NEW_IDS_FILE="job_ids_nodep.txt"
> "$NEW_IDS_FILE"

for K in A B C D E; do
  for i in ${KI_MAP[$K]}; do
    out=$(sbatch -M blanca --job-name="c20cmp_${K}_${i}_nodep" solve_one_condition.sbatch "$K" "$i")
    job_id=$(echo "$out" | awk '{print $4}')
    echo "$K sweep$i -> job $job_id (no dependency)"
    echo "$K sweep$i $job_id" >> "$NEW_IDS_FILE"
  done
done

echo "=== new final (sweep9) job IDs per chain ==="
grep " sweep9 " "$NEW_IDS_FILE"
