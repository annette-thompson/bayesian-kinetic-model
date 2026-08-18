#!/bin/bash
#SBATCH --job-name=lmod_test
#SBATCH --partition=acpu
#SBATCH --qos=cpu-normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:10:00
#SBATCH --output=/projects/anth4580/Bayesian/job_files/%x.%j.out
#SBATCH --account=ucb634_asc2

# Minimal repro for the intermittent "module load anaconda" / lmod failure
# seen on the GPU segment worker -- isolates just the environment setup
# (the actual suspect) from the expensive ODE/inference work, on the much
# faster CPU queue instead of waiting in the GPU queue to watch it fail at
# the same line. Doesn't call inference_runner.py at all; if you want that
# too, submit run_inference_segment.sh directly instead (same preamble,
# identical failure surface, plus a real (short) inference segment) --
# see the two-line note at the bottom of this file.
#
# Run interactively a few times to catch the flake:
#   for i in 1 2 3 4 5; do sbatch "Bayesian Inference/test_lmod_cpu.sh"; done
#   squeue -u "$USER" --name=lmod_test
#   (once done) grep -L "OK: import" /projects/anth4580/Bayesian/job_files/lmod_test.*.out

echo "----------------------------------------------------------"
echo "==> Host: $(hostname)"
echo "==> SLURM job: ${SLURM_JOB_ID:-unset}   node: ${SLURM_NODELIST:-unset}"
echo "==> Date: $(date)"
echo "----------------------------------------------------------"

echo "==> MODULEPATH: ${MODULEPATH:-unset}"
echo "==> type module (before lmod.sh): $(type module 2>&1)"

source /etc/profile.d/lmod.sh
echo "==> sourced /etc/profile.d/lmod.sh, exit=$?"
echo "==> type module (after lmod.sh): $(type module 2>&1)"
module --version 2>&1 | head -5

echo "----------------------------------------------------------"
echo "==> Attempt 1: module load anaconda"
module load anaconda
STATUS1=$?
echo "==> module load anaconda exit=$STATUS1"

if [[ $STATUS1 -ne 0 ]]; then
  echo "==> Attempt 1 failed -- retrying after a short sleep (tests the "
  echo "    'transient at node boot / module cache not yet visible' theory)"
  sleep 5
  module load anaconda
  STATUS2=$?
  echo "==> Attempt 2 (after sleep) exit=$STATUS2"
fi

echo "==> module list:"
module list 2>&1

echo "----------------------------------------------------------"
echo "==> conda activate Bayesian"
conda activate Bayesian
echo "==> conda activate exit=$?"
echo "==> which python: $(command -v python)"
echo "==> which conda: $(command -v conda)"
python --version 2>&1

echo "----------------------------------------------------------"
echo "==> Import sanity check (no inference run, just confirms the env"
echo "    inference_runner.py would actually get is usable):"
python - <<'PY'
import importlib.metadata as md
packages = ["arviz", "blackjax", "diffrax", "equinox", "jax", "jaxlib", "numpy", "pymc", "pytensor", "zarr"]
for package in packages:
    try:
        print(f"OK: import {package} == {md.version(package)}")
    except md.PackageNotFoundError:
        print(f"MISSING: {package}")
PY

echo "----------------------------------------------------------"
echo "==> Done. If this failed at 'module load anaconda' the same way the"
echo "    GPU job did, it reproduces here -- check for node-specific"
echo "    patterns via SLURM_NODELIST across a few repeated submissions."

# For a fuller repro that also exercises inference_runner.py (short, cheap
# config, tiny --max_hours) on the same CPU queue instead of this
# environment-only check:
#   sbatch "Bayesian Inference/run_inference_segment.sh" \
#     "Results/GPU Scaling Tests/Test FabD FabH FabG - a2/solver_params.json" 0.05
