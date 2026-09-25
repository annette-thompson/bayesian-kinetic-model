#!/bin/bash -l
#SBATCH --job-name=me1refit
#SBATCH --partition=blanca
#SBATCH --qos=preemptable
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=06:15:00
# No GPU request here: submit through job_files/gpu_submit.sh, which queues this on Blanca
# and Alpine at once; the first twin to start runs it and cancels the other.
#SBATCH --output=/projects/anth4580/Bayesian/job_files/%x.%j.out
#SBATCH --export=NONE

# Refit the five originally-fitted ME1 parameters against dataset D, on a GPU.
#   ./gpu_submit.sh --time 06:15:00 --name me1refit -- me1_refit_gpu.sbatch [evals]
#
# Why a GPU: the objective solves seven initial-rate conditions that differ only in their
# initial state, which me1_model.solve_batch already expresses as one jitted vmap. On CPU
# that batch is cost-neutral -- vmap runs the stiff solves in lockstep, so it moves at the
# pace of its slowest member. On a GPU the members run genuinely in parallel, so the batch
# should approach the cost of a single solve and make a few hundred objective evaluations
# practical rather than an overnight job.

_die() { echo "Error: $*" >&2; exit 1; }
EVALS="${1:-250}"
SETS="${2:-C20+unsat C20+unsat+FBinit}"   # which reaction sets to refit
BASE="/projects/anth4580/Bayesian"
cd "$BASE/ME1_Refit" || _die "no ME1_Refit directory"

source "$BASE/job_files/gpu_twin_claim.sh" || exit 0
[[ -n "$GPU_MODEL" ]] || _die "no GPU visible -- submit through job_files/gpu_submit.sh"

module purge
module load miniforge
mamba activate Bayesian

export PYTHONUNBUFFERED=1
export JAX_COMPILATION_CACHE_DIR="$BASE/.jax_cache"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export EQX_ON_ERROR=nan

echo "==> node=$(hostname) gpu=$GPU_MODEL cluster=${SLURM_CLUSTER_NAME:-blanca} evals=$EVALS"

# Confirm JAX actually sees the GPU and measure one objective evaluation before committing
# to a few hundred of them -- a silent CPU fallback would otherwise look like a slow fit.
python - "$SETS" <<'PY'
import time, sys
sys.path.insert(0, ".")
import jax
print("jax devices:", jax.devices(), flush=True)
import me1_config as cfg, me1_model as mm, me1_objective as mo
# Build the FIRST requested set, not the config default: the default points at
# C20+unsat+FBinit, and a missing reaction folder there would abort the timing check
# before the refits even start (it did, the first time this ran).
first = sys.argv[1].split()[0] if len(sys.argv) > 1 else "C20+unsat"
rx = cfg.PROJECT / "Reactions" / "EC_FAS_ME1" / first
t0 = time.time(); M = mm.ME1Model(reactions_dir=rx); print(f"build {first} {time.time()-t0:.1f}s", flush=True)
data = mo.load_data(); p = dict(cfg.PUBLISHED)
t0 = time.time(); mo.evaluate(M, p, data=data); print(f"first eval (compiles) {time.time()-t0:.1f}s", flush=True)
t0 = time.time(); r = mo.evaluate(M, p, data=data); print(f"warm eval {time.time()-t0:.2f}s", flush=True)
print(f"obj1={r.obj1:.4f} obj2={r.obj2:.3f} obj3={r.obj3:.3f}", flush=True)
PY

for RX in $SETS; do
    echo "================ refit to D: $RX ================"
    time python -u refit_to_D.py --reactions "$RX" --evals "$EVALS"
done
echo "ALL REFITS DONE"
