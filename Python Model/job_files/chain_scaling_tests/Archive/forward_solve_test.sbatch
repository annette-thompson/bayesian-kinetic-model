#!/bin/bash -l
#SBATCH --job-name=mask_fwd_test
#SBATCH --partition=blanca
#SBATCH --qos=preemptable
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --constraint="A100"
# gpu:a100:1 rather than gpu:1 -- a bare gpu:1 can be satisfied by a MIG slice
# on the h100 nodes (gpu:h100_3g.40gb / h100_2g.20gb is 3/7 or 2/7 of a card),
# and C20+unsat silently ran two chunks on one. Naming the type pins a whole
# physical GPU; no a100 node on blanca is MIG-partitioned. Pool: biokem2,
# curc1-4, ivc2 = 18 cards.
#SBATCH --output=/projects/anth4580/Bayesian/job_files/chain_scaling_tests/%x.%j.out
#SBATCH --mail-type=ALL
#SBATCH --export=NONE

# Deterministic forward solves only (no MCMC) -- cheap, so CPU count and time
# budget are both much smaller than a real inference segment. Same GPU family
# constraint as run_inference_segment_gpu_blanca.sh: fp64 throughput matters,
# Blanca's fleet is mostly A40/L40/T4/RTX6000 which are fast for fp32/ML but
# bad for this float64-heavy codebase.

module purge
module load miniforge
mamba activate Bayesian

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export NUMEXPR_NUM_THREADS=${SLURM_CPUS_PER_TASK}

echo "==> Host: $(hostname)"
echo "==> GPU model: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo unknown)"
echo "==> Python executable: $(command -v python)"

time python -u /projects/anth4580/Bayesian/job_files/chain_scaling_tests/forward_solve_test.py
