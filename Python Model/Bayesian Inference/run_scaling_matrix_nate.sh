#!/bin/bash
# Run the whole reaction-count scaling matrix ON nate, sequentially, with no
# Mac in the loop. This is the detached counterpart to submit_scaling_matrix.sh
# --nate: that one drives the sequence from your Mac (which then has to stay
# awake for days to hand off between configs), this one hands the entire
# sequence to nate so your Mac can sleep, disconnect, or go home.
#
# Normally launched by `submit_scaling_matrix.sh --nate --detach`, which pushes
# the repo and nohups this over ssh. Can also be run directly on nate.
#
# Every config runs to its draw target (or MAX_HOURS, if given) before the next
# one starts -- nate has one RTX 3080, and two runs sharing it would contend for
# the GPU and spoil both timings, which is the whole measurement here.
#
# Usage (on nate, from the project directory):
#   "Bayesian Inference/run_scaling_matrix_nate.sh" [MAX_HOURS]
#
# Progress is written to job_files/scaling_matrix.status (one "state<TAB>label"
# line per config, rewritten after every config) and job_files/scaling_matrix.log;
# each config's own output goes to job_files/<label with _ for spaces>.log, the
# same convention submit_inference_nate.sh --status reads.
#
# Deliberately no `set -e`: one config failing must not strand the rest.
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
WORKER="$SCRIPT_DIR/run_inference_nate.sh"
JOB_FILES="$PROJECT_DIR/job_files"
STATUS_FILE="$JOB_FILES/scaling_matrix.status"
MAX_HOURS="${1:-}"

mkdir -p "$JOB_FILES"

[[ -r "$WORKER" ]] || { echo "Error: worker not found: $WORKER" >&2; exit 1; }

# Never share the GPU: refuse to start if anything is already sampling here,
# including a second copy of this matrix.
if pgrep -f "inference_runner.py" >/dev/null; then
  echo "Error: an inference run is already active on nate (pid $(pgrep -f inference_runner.py | head -n1))." >&2
  echo "       Stop it first -- the single GPU can only host one run at a time." >&2
  exit 1
fi
if [[ "$(pgrep -cf "run_scaling_matrix_nate.sh")" -gt 1 ]]; then
  echo "Error: another scaling matrix is already running on nate." >&2
  exit 1
fi

CONFIGS=()
while IFS= read -r cfg; do
  CONFIGS+=("$cfg")
done < <(find "$PROJECT_DIR/Results/GPU Scaling Tests" -mindepth 2 -maxdepth 2 -name solver_params.json | sort)
[[ ${#CONFIGS[@]} -gt 0 ]] || { echo "Error: no configs under Results/GPU Scaling Tests/*/solver_params.json" >&2; exit 1; }

STATES=()
for _ in "${CONFIGS[@]}"; do STATES+=("pending"); done

_label() { basename "$(dirname "$1")"; }

_write_status() {
  local i
  for i in "${!CONFIGS[@]}"; do
    printf '%s\t%s\n' "${STATES[$i]}" "$(_label "${CONFIGS[$i]}")"
  done > "$STATUS_FILE"
}

# Same completion test compare_sampler_benchmarks.py applies: both artifacts
# present means finalized. A checkpoint alone means it stopped short (MAX_HOURS,
# or an interrupted run) and re-running this script resumes it.
_state_of() {
  local dir="$1"
  if [[ -f "$dir/timing.json" && -f "$dir/posterior_samples_pm.nc" ]]; then
    echo "done"
  elif [[ -d "$dir/checkpoint" ]]; then
    echo "INCOMPLETE"
  else
    echo "NO-OUTPUT"
  fi
}

echo "=================================================="
echo "==> Scaling matrix on nate (sequential, one GPU)"
echo "==> Host: $(hostname)   PID: $$   Started: $(date '+%F %T')"
echo "==> Configs: ${#CONFIGS[@]}   Per-config max hours: ${MAX_HOURS:-unbounded}"
echo "==> Status file: $STATUS_FILE"
echo "=================================================="

_write_status
n_bad=0
n_incomplete=0
matrix_start=$SECONDS

for i in "${!CONFIGS[@]}"; do
  cfg="${CONFIGS[$i]}"
  label="$(_label "$cfg")"
  log="$JOB_FILES/${label// /_}.log"

  STATES[$i]="running"
  _write_status
  echo ""
  echo "=== [$((i + 1))/${#CONFIGS[@]}] $label -- started $(date '+%F %T') ==="
  echo "    log: $log"

  start=$SECONDS
  if "$WORKER" "$cfg" "$MAX_HOURS" > "$log" 2>&1; then
    STATES[$i]="$(_state_of "$(dirname "$cfg")")"
  else
    STATES[$i]="FAILED"
  fi
  elapsed=$((SECONDS - start))

  case "${STATES[$i]}" in
    done)       ;;
    INCOMPLETE) n_incomplete=$((n_incomplete + 1)) ;;
    *)          n_bad=$((n_bad + 1)) ;;
  esac

  printf '=== %s -> %s after %02d:%02d:%02d ===\n' \
    "$label" "${STATES[$i]}" $((elapsed / 3600)) $((elapsed % 3600 / 60)) $((elapsed % 60))
  [[ "${STATES[$i]}" == "FAILED" || "${STATES[$i]}" == "NO-OUTPUT" ]] && tail -n 15 "$log"
  _write_status
done

total=$((SECONDS - matrix_start))
echo ""
echo "=================================================="
printf '==> Matrix finished in %02d:%02d:%02d at %s\n' \
  $((total / 3600)) $((total % 3600 / 60)) $((total % 60)) "$(date '+%F %T')"
for i in "${!CONFIGS[@]}"; do
  printf '    %-11s %s\n' "${STATES[$i]}" "$(_label "${CONFIGS[$i]}")"
done
(( n_incomplete )) && echo "==> $n_incomplete incomplete -- re-run this script to resume them"
(( n_bad )) && echo "==> $n_bad need attention -- see the per-config logs in $JOB_FILES"
echo "==> Pull results to your Mac with:  Sync/sync_from_nate.sh"
echo "=================================================="

[[ "$n_bad" -eq 0 ]]
