#!/bin/bash
# Stage-1 timing matrix on nate. Run ON nate (submit_benchmark_matrix.sh --detach
# nohups it there), or directly for a foreground pass.
#
# GPU and CPU cells run STRICTLY SEQUENTIALLY. nate has one GPU and shared cores,
# and a CPU cell saturating all 16 threads while a GPU cell is being timed would
# corrupt both numbers -- that exact contention invalidated an earlier sweep here.
#
# Resumable: benchmark_matrix.py skips cells already in the JSONL, so re-running
# after an interruption is a fast no-op over completed work.
#
# All arguments are forwarded verbatim to benchmark_matrix.py, so every axis it
# supports (--configs, --tolerances, --precisions, --floors, --max-hours) is
# reachable without editing this script. Only machine and device ordering are fixed
# here, because those are properties of nate rather than of the experiment.
#
# Usage: run_benchmark_matrix_nate.sh [benchmark_matrix.py args...]
#   run_benchmark_matrix_nate.sh --configs "Chain C4 - a2" \
#       --tolerances 1e-4:1e-8,1e-6:1e-8 --precisions 32,64 --floors 0,0.001
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 1
source "$HOME/miniforge3/etc/profile.d/conda.sh"; conda activate Bayesian

# Never share the GPU: refuse if a real inference or another copy is already here.
if pgrep -f "[i]nference_runner.py" >/dev/null; then
  echo "Error: an inference run is active on nate -- refusing to start (timings would be garbage)" >&2
  exit 1
fi
# PID lockfile, NOT pgrep. `pgrep -cf run_benchmark_matrix_nate.sh` also matches
# the ssh wrapper whose command line contains that same script name, so launching
# over ssh always looked like a second copy. The bracket trick does not help here:
# the wrapper genuinely contains the plain string, unlike the case where the
# searcher matches its own pattern.
LOCK="$HOME/Bayesian/job_files/benchmark_matrix.lock"
mkdir -p "$(dirname "$LOCK")"
if [ -r "$LOCK" ] && kill -0 "$(cat "$LOCK" 2>/dev/null)" 2>/dev/null; then
  echo "Error: benchmark matrix already running on nate (pid $(cat "$LOCK"))" >&2
  exit 1
fi
echo $$ > "$LOCK"
trap 'rm -f "$LOCK"' EXIT

ARGS=(--stage timing --machine nate --devices gpu,cpu "$@")

echo "=================================================="
echo "==> Stage-1 timing matrix on $(hostname)  $(date '+%F %T')"
echo "==> args: ${ARGS[*]}"
echo "=================================================="
python -u Utilities/benchmark_matrix.py "${ARGS[@]}"
echo "==> nate matrix pass finished $(date '+%F %T')"
