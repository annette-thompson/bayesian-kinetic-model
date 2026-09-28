#!/bin/bash
#SBATCH --job-name=tier1_fig78
#SBATCH --partition=blanca
#SBATCH --qos=preemptable
#SBATCH --gres=gpu:1
#SBATCH --constraint=A100|(h100&xeon)
#SBATCH --time=06:00:00
#SBATCH --mem=48G
#SBATCH --output=tier1/fig78.%j.out
# Figs 7 and 8 (outline 3.4-3.5): posterior-integrated enzyme sensitivity (posterior_morris.py)
# and the chain-length response to the FabF/FabB : TesA ratio (posterior_ratio_response.py),
# forward solves over R2's posterior at the scripts' defaults (50 draws; r 20; 9 ratios, grid 5).
# On the laptop CPU a batch of 20 solves took 13 s (2026-09-27), ~3 h for both at these sizes.
# Submit from job_files/ once R2 has finalized:  sbatch tier1/posterior_figs.sbatch
# (tier1/wait_then_fig78.sbatch waits for it and submits this).
PY=/projects/anth4580/software/anaconda/envs/Bayesian/bin/python   # the Bayesian env, without modules
export PYTHONUNBUFFERED=1
export JAX_COMPILATION_CACHE_DIR=/projects/anth4580/Bayesian/.jax_cache
nvidia-smi --query-gpu=name --format=csv,noheader
cd tier1
R2="/projects/anth4580/Bayesian/Results/Tier1/Tier1 C14+unsat - a1c3a2"
"$PY" -c "import sys; from tier1_status import is_finalized; sys.exit(0 if is_finalized(sys.argv[1]) else 1)" "$R2" \
  || { echo "R2 has not finalized; nothing run"; exit 1; }
POST="Results/Tier1/Tier1 C14+unsat - a1c3a2/posterior_samples_pm.nc"
echo "==> Fig 7 $(date '+%a %H:%M')"
time "$PY" posterior_morris.py --posterior "$POST" --params a1,c3,a2 2>&1 | grep -v Warning
echo "==> Fig 8 $(date '+%a %H:%M')"
time "$PY" posterior_ratio_response.py --posterior "$POST" --params a1,c3,a2 2>&1 | grep -v Warning
echo "==> done $(date '+%a %H:%M')"
