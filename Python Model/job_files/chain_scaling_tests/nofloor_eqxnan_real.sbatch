#!/bin/bash -l
#SBATCH --job-name=nfeqx
#SBATCH --partition=blanca
#SBATCH --qos=preemptable
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=12:15:00
# No GPU request here: submit through job_files/gpu_submit.sh, which queues this
# script on Blanca (any full A100/V100/H100) and Alpine (full A100) at once; the
# first twin to start runs it and cancels the other (claim below). A bare sbatch
# gets no GPU and stops at the GPU check.
#SBATCH --output=/projects/anth4580/Bayesian/job_files/chain_scaling_tests/%x.%j.out
#SBATCH --mail-type=ALL
#SBATCH --export=NONE

# Real production tune/draws (unchanged from each system's own "Chain X - a1"
# config), no-floor + EQX_ON_ERROR=nan, writing to a fresh
# "Chain X - a1 nofloor-eqxnan" results dir so the original floor-based
# checkpoint/progress is untouched for comparison.
#   ../gpu_submit.sh --time 12:15:00 -- nofloor_eqxnan_real.sbatch C4_NoFB
#   ../gpu_submit.sh --time 12:15:00 -- nofloor_eqxnan_real.sbatch C6
#   ../gpu_submit.sh --time 12:15:00 -- nofloor_eqxnan_real.sbatch C8
#   ../gpu_submit.sh --time 12:15:00 -- nofloor_eqxnan_real.sbatch C10
#   ../gpu_submit.sh --time 12:15:00 -- nofloor_eqxnan_real.sbatch C12
# Optional 2nd arg: prior-width label (default "tightest"), e.g.
#   ../gpu_submit.sh --time 12:15:00 -- nofloor_eqxnan_real.sbatch C10 narrowest
# 12h segment budget (15 min margin under the 12:15 SLURM limit) -- checkpoints
# and exits cleanly if not done; resubmit_if_needed.sh queues the next segment.

_die() { echo "Error: $*" >&2; exit 1; }
SYSTEM="${1:-}"
WIDTH="${2:-tightest}"
MAXH="${3:-12.0}"
[[ -n "$SYSTEM" ]] || _die "Usage: sbatch $0 {C4_NoFB|C6|C8|C10|C12} [prior_width]"
SOLVER_PARAMS="/projects/anth4580/Bayesian/Results/Chain Scaling Tests/Chain $SYSTEM - a1 $WIDTH nofloor-eqxnan/solver_params.json"
[[ -r "$SOLVER_PARAMS" ]] || _die "Config not found: $SOLVER_PARAMS"
# First of the Blanca/Alpine twins to start runs; the other exits here.
source /projects/anth4580/Bayesian/job_files/gpu_twin_claim.sh || exit 0
[[ -n "$GPU_MODEL" ]] || _die "no GPU visible -- submit through job_files/gpu_submit.sh"
# Record this job in its set's jobids file. warmup_status.py discovers systems
# ONLY through that file, which used to be written solely by a successful
# resubmission -- and none ever succeeded, so a manually submitted run never
# showed up in it. Idempotent: resubmit_if_needed.sh may have logged this ID
# already at submit time, and a duplicate would double its counted compute.
JOBIDS_FILE="/projects/anth4580/Bayesian/job_files/inference_a1_${WIDTH// /_}_nofloor_eqxnan_jobids.txt"
# Alpine job IDs are written "alpine:<id>" so warmup_status.py asks the right cluster.
_JID="${SLURM_JOB_ID:-}"
[[ -n "$_JID" && "${SLURM_CLUSTER_NAME:-blanca}" != blanca ]] && _JID="${SLURM_CLUSTER_NAME}:$_JID"
if [[ -n "$_JID" ]] && ! grep -qx "$SYSTEM $_JID" "$JOBIDS_FILE" 2>/dev/null; then
    echo "$SYSTEM $_JID" >> "$JOBIDS_FILE"
fi

module purge
module load miniforge
mamba activate Bayesian

export PYTHONUNBUFFERED=1
export BAYESIAN_BATCH_STRATEGY=vmap
export JAX_COMPILATION_CACHE_DIR="/projects/anth4580/Bayesian/.jax_cache"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export NUMEXPR_NUM_THREADS=${SLURM_CPUS_PER_TASK}

export MASK_TEST_MODE=nofloor
export MASK_TEST_SOLVER_PARAMS="$SOLVER_PARAMS"
export MASK_TEST_MAX_HOURS="${MAXH}"
export EQX_ON_ERROR=nan


# ---- self-resubmission -------------------------------------------------- #
# The successor is submitted from here rather than pre-chained with
# --dependency=afterany, so it exists only when the work is actually
# unfinished: resubmit_if_needed.sh checks the checkpoint phase and the
# per-chain acceptance first. The TERM trap covers preemption -- on this QOS a
# job can be requeued out from under us, and SIGTERM is the only warning.
RESUB="/projects/anth4580/Bayesian/job_files/chain_scaling_tests/resubmit_if_needed.sh"
# Not "$(basename "${BASH_SOURCE[0]}")": under SLURM the batch script runs
# from a spool copy named "slurm_script", so resubmit_if_needed.sh built
# ".../masking_check/slurm_script" and EVERY automatic resubmission failed
# with "Unable to open file" -- a run that hit its segment budget just
# stopped, silently. Found 2026-09-14 after six runs sat idle ~2.5 days.
_SCRIPT_NAME="nofloor_eqxnan_real.sbatch"
_resubmitted=0
_do_resubmit() {
    [[ $_resubmitted -eq 1 ]] && return 0
    _resubmitted=1
    bash "$RESUB" "$_SCRIPT_NAME" "$SYSTEM" "$WIDTH" "$MAXH" "${1:-normal}"
}
trap '_do_resubmit preempted; exit 143' TERM

echo "==> System: $SYSTEM  Prior width: $WIDTH"
echo "==> Host: $(hostname)"
echo "==> GPU model: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo unknown)"

time python -u /projects/anth4580/Bayesian/job_files/chain_scaling_tests/truncated_warmup_test.py

_do_resubmit normal
