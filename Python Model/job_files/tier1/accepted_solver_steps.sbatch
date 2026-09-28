#!/bin/bash
#SBATCH --job-name=tier1_accepted_steps
#SBATCH --partition=blanca
#SBATCH --qos=preemptable
#SBATCH --gres=gpu:1
#SBATCH --constraint=A100|(h100&xeon)
#SBATCH --time=03:00:00
#SBATCH --mem=48G
#SBATCH --output=tier1/accepted_steps.%j.out
# Solver steps at every Tier-1 run's accepted states (accepted_solver_steps.py), one process
# per model group. Submit from job_files/:  sbatch tier1/accepted_solver_steps.sbatch
PY=/projects/anth4580/software/anaconda/envs/Bayesian/bin/python   # the Bayesian env, without modules
export PYTHONUNBUFFERED=1
export JAX_COMPILATION_CACHE_DIR=/projects/anth4580/Bayesian/.jax_cache
nvidia-smi --query-gpu=name --format=csv,noheader
cd tier1
OUT=../../Results/Tier1/figures/accepted_solver_steps.json
rm -f "$OUT"
"$PY" accepted_solver_steps.py --groups 2>/dev/null | grep "^GROUP" | while IFS=$'\t' read -r _ key n; do
  "$PY" accepted_solver_steps.py --group "$key" --json "$OUT" 2>&1 | grep -v Warning
done
echo "==> done $(date '+%a %H:%M')"
