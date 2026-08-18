#!/bin/bash
# Submit the C4 matrix on Alpine: one job per device, same axes as nate.
#
# Run this ON an Alpine login node (it calls sbatch). Partition/qos/gres are given
# here rather than in the worker so a single worker script serves both devices.
#
# Usage: ./submit_chain_c4_alpine.sh
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 1

AXES=(--configs "Chain C4 - a2"
      --tolerances "1e-4:1e-8,1e-5:1e-8,1e-6:1e-8,1e-4:1e-10,1e-6:1e-10"
      --precisions "32,64"
      --floors "0,0.001"
      --timeout 1800
      --max-eval-seconds 600)

gpu_id=$(sbatch --parsable -p aa100 -q gpu-normal --gres=gpu:a100-40gb:1 \
  "Bayesian Inference/run_benchmark_matrix_alpine.sh" gpu "${AXES[@]}")
echo "  submitted GPU job $gpu_id"

cpu_id=$(sbatch --parsable -p acpu -q cpu-normal \
  "Bayesian Inference/run_benchmark_matrix_alpine.sh" cpu "${AXES[@]}")
echo "  submitted CPU job $cpu_id"

echo
echo "  watch:   squeue -u \$USER"
echo "  logs:    /projects/anth4580/Bayesian/job_files/benchmx.<jobid>.out"
