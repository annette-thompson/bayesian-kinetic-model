#!/bin/bash
# Submits the full 4 PID x 3 rtol x 3 atol = 36-combination C20+unsat grid,
# each as an independent GPU job (no dependency chains -- these are all
# single-system runs, not ladder sweeps). Run this ON Blanca (ssh curc).
set -e
JOBDIR="/projects/anth4580/Bayesian/job_files/c20unsat_solver_compare"
cd "$JOBDIR"

PIDS=("0.4 0.3 0.0" "0.3 0.3 0.0" "0.2 0.4 0.0" "0.1 0.3 0.0")
RTOLS=(1e-4 1e-5 1e-6)
ATOLS=(1e-6 1e-7 1e-8)

: > job_ids_gpu_grid.txt
for PID in "${PIDS[@]}"; do
  read -r P I D <<< "$PID"
  for RTOL in "${RTOLS[@]}"; do
    for ATOL in "${ATOLS[@]}"; do
      LABEL="p${P}_i${I}_rtol${RTOL}_atol${ATOL}"
      JID=$(sbatch -M blanca --job-name="gpu_${LABEL}" --parsable generate_and_score_gpu.sbatch "$P" "$I" "$D" "$RTOL" "$ATOL" | cut -d';' -f1)
      echo "label=$LABEL job=$JID" | tee -a job_ids_gpu_grid.txt
    done
  done
done
