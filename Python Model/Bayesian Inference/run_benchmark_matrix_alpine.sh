#!/bin/bash -l
#SBATCH --job-name=benchmx
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --time=20:00:00
#SBATCH --output=/projects/anth4580/Bayesian/job_files/%x.%j.out
#SBATCH --export=NONE
#SBATCH --account=ucb634_asc2
#
# Stage-1 timing matrix on Alpine. ONE worker for both devices: partition, qos and
# gres are supplied at submit time (sbatch CLI flags override in-file #SBATCH), so
#   sbatch -p aa100 -q gpu-normal --gres=gpu:a100-40gb:1 ... run_benchmark_matrix_alpine.sh gpu
#   sbatch -p acpu  -q cpu-normal                        ... run_benchmark_matrix_alpine.sh cpu
# rather than maintaining a fourth near-identical SLURM script.
#
# `-l` and `--export=NONE` are load-bearing: sbatch defaults to --export=ALL, which
# drags the submitting shell's MODULEPATH (lacking /curc/sw/alpine-modules/*) into
# the job and makes `module load miniforge` fail with "unknown module".
#
# --max-hours leaves headroom under the wall limit so the driver stops between
# cells and the JSONL stays consistent, rather than being killed mid-measurement.
# Everything after DEVICE is forwarded verbatim to benchmark_matrix.py, so
# --configs/--tolerances/--precisions/--floors are reachable without editing this
# file. Only machine/device/env are fixed here.
set -uo pipefail
DEVICE="${1:?usage: run_benchmark_matrix_alpine.sh gpu|cpu [benchmark_matrix.py args...]}"
shift
cd /projects/anth4580/Bayesian || exit 1
module purge && module load miniforge && mamba activate Bayesian

# The jax 0.10.2 env, matching nate's stack, so machine differences are hardware
# rather than library version (the older jax 0.7.0 env needs rtol 1e-6 to give
# finite gradients at all).
PY=/projects/anth4580/software/anaconda/envs/Bayesian_jaxgpu/bin/python
[ -x "$PY" ] || PY=python

echo "==> $(hostname)  device=$DEVICE  $(date '+%F %T')"
"$PY" -c "import jax,importlib.metadata as md;print('STACK jax',md.version('jax'),'diffrax',md.version('diffrax'),jax.devices())"
"$PY" -u Utilities/benchmark_matrix.py --stage timing --machine alpine \
  --devices "$DEVICE" --python "$PY" --max-hours 18 "$@"
echo "==> alpine $DEVICE pass finished $(date '+%F %T')"
