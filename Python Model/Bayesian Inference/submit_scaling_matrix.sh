#!/bin/bash
# Submit the whole reaction-count-axis scaling matrix in one command: one
# independent inference run per "Results/GPU Scaling Tests/*/solver_params.json"
# config (the batch-B configs, single free param `a2`, which isolate pure
# reaction-count effect on run time and convergence -- tests 1+2 of the scaling
# test plan). Configs and their Data/*.csv come from
# generate_scaling_test_data.ipynb.
#
# Alpine (default): run on a login node from the "Python Model" directory, after
# Sync/sync_to_cluster.sh. Each config gets its own dependent job chain
# (submit_inference_chain.sh --gpu), so configs run in parallel.
#
# --nate: run from your Mac instead, against the lab RTX 3080 workstation. One
# GPU means configs run ONE AT A TIME (submit_inference_nate.sh --wait), and
# --segments/--time are ignored (no SLURM wall-clock kill there). Each config is
# checked for real output on nate before moving on, and a config that fails
# doesn't strand the ones behind it -- see the per-config summary at the end.
#
# --detach: same sequence, but run BY nate instead of driven from your Mac (see
# run_scaling_matrix_nate.sh). Returns as soon as it is launched, so the Mac can
# sleep or go home -- which plain --nate cannot do, since there the Mac holds the
# loop and polls between configs. Use --status / --stop to check on or end it.
#
# Usage:
#   Bayesian\ Inference/submit_scaling_matrix.sh [--segments N] [--max-hours H]
#       [--time HH:MM:SS] [--nate] [--dry-run]
set -euo pipefail

_die() { echo "Error: $*" >&2; exit 1; }

# Print instead of running, under --dry-run. %q keeps paths with spaces pasteable.
_run() {
  if (( DRY_RUN )); then
    printf '  (dry run)'; printf ' %q' "$@"; printf '\n'
  else
    "$@"
  fi
}

# What a finished nate run actually left behind, in its results dir on nate.
# submit_inference_nate.sh --wait only knows the process exited -- it can't tell
# a completed run from one that died on startup -- so check for the two files
# compare_sampler_benchmarks.py needs before calling a config done. A checkpoint
# but no outputs means it stopped early (--max-hours, or an interrupted run) and
# re-running the same command resumes it; neither means it never really ran.
SSH_OPTS=(-o ConnectTimeout=10 -o BatchMode=yes)

# Is anything sampling on nate right now (any config, not just ours)? The [i] is
# load-bearing -- see the MATCH comment in submit_inference_nate.sh: without it
# this pattern matches the ssh shell that carries it, so it would always say yes.
_nate_busy() {
  ssh "${SSH_OPTS[@]}" "$NATE_HOST" "pgrep -f '[i]nference_runner.py' >/dev/null" 2>/dev/null
}

_nate_run_state() {
  local rel_dir="$1" out
  out="$(ssh "${SSH_OPTS[@]}" "$NATE_HOST" "
    d='$NATE_PROJECT/$rel_dir'
    if [ -f \"\$d/timing.json\" ] && [ -f \"\$d/posterior_samples_pm.nc\" ]; then echo done
    elif [ -d \"\$d/checkpoint\" ]; then echo incomplete
    else echo missing; fi" 2>/dev/null)" || out=unreachable
  printf '%s' "$out"
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

SEGMENTS=2
MAX_HOURS=""
TIME="24:00:00"
DRY_RUN=0
NATE=0
ACTION=run

usage() {
  cat >&2 <<EOF
Usage: $0 [options]

Options:
  --segments N       dependent segments per config, Alpine only (default $SEGMENTS)
  --max-hours H      per-segment wall budget before checkpoint (default: --time
                     minus 15 minutes; see submit_inference_chain.sh). On --nate,
                     the optional per-config budget; omit for unbounded.
  --time HH:MM:SS    SBATCH --time per job, Alpine only (default $TIME)
  --nate             run the matrix on nate instead of Alpine (sequential,
                     one config at a time -- see submit_inference_nate.sh).
                     Drives the sequence from this Mac, which must stay awake.
  --detach           like --nate, but hand the whole sequence to nate and return
                     immediately, so your Mac can sleep/disconnect. Implies --nate.
  --status           report detached-matrix progress on nate, then exit
  --stop             stop a detached matrix on nate, then exit (checkpoints kept)
  --dry-run          print what would be submitted without calling sbatch/ssh
  -h, --help         show this help
EOF
  exit 1
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --segments)  SEGMENTS="$2";  shift 2 ;;
    --max-hours) MAX_HOURS="$2"; shift 2 ;;
    --time)      TIME="$2";      shift 2 ;;
    --nate)      NATE=1;         shift ;;
    --detach)    NATE=1; ACTION=detach; shift ;;
    --status)    NATE=1; ACTION=status; shift ;;
    --stop)      NATE=1; ACTION=stop;   shift ;;
    --dry-run)   DRY_RUN=1;      shift ;;
    -h|--help)   usage ;;
    *) echo "Unknown argument: $1" >&2; usage ;;
  esac
done

if (( NATE )); then
  SUBMIT="$SCRIPT_DIR/submit_inference_nate.sh"
  NEEDS_CMD=ssh
  ARGS=(--no-sync --wait)   # matrix-level sync happens once, below
  # Defines NATE_HOST/NATE_PROJECT for the completion check. Source it only
  # after SUBMIT is set -- it overwrites SCRIPT_DIR with its own location.
  source "$PROJECT_DIR/Sync/nate_sync_config.sh"
else
  SUBMIT="$SCRIPT_DIR/submit_inference_chain.sh"
  NEEDS_CMD=sbatch
  ARGS=(--gpu --segments "$SEGMENTS" --time "$TIME")
fi
[[ -z "$MAX_HOURS" ]] || ARGS+=(--max-hours "$MAX_HOURS")

# --- detached mode: the sequence runs ON nate, no Mac in the loop ------------
# The [r] does the same job as the [i] above: keep the pattern from matching the
# ssh shell that carries it.
MATRIX_MATCH='[r]un_scaling_matrix_nate.sh'
REMOTE_MATRIX="Bayesian Inference/run_scaling_matrix_nate.sh"
REMOTE_STATUS="$NATE_PROJECT/job_files/scaling_matrix.status"
REMOTE_MLOG="$NATE_PROJECT/job_files/scaling_matrix.log"

if [[ "$ACTION" == "status" ]]; then
  ssh "${SSH_OPTS[@]}" "$NATE_HOST" "
    if pgrep -f '$MATRIX_MATCH' >/dev/null; then echo '==> Matrix RUNNING on nate'
    else echo '==> Matrix not running on nate'; fi
    echo '--- progress ---'
    cat '$REMOTE_STATUS' 2>/dev/null || echo '(no status file yet)'
    echo '--- last matrix log lines ---'
    tail -n 15 '$REMOTE_MLOG' 2>/dev/null; true" || _die "could not reach nate"
  exit 0
fi

if [[ "$ACTION" == "stop" ]]; then
  # Driver first, so it can't start the next config after we kill the current
  # run. The interrupted config keeps its checkpoint and resumes on re-launch.
  ssh "${SSH_OPTS[@]}" "$NATE_HOST" \
    "pkill -f '$MATRIX_MATCH'; pkill -f '[i]nference_runner.py'; true" || _die "could not reach nate"
  echo "==> Stopped. Checkpoints are kept -- re-run with --detach to resume."
  exit 0
fi

if [[ "$ACTION" == "detach" ]]; then
  if (( DRY_RUN )); then
    echo "(dry run) ssh $NATE_HOST nohup '$REMOTE_MATRIX' '$MAX_HOURS'"
    exit 0
  fi
  echo "==> Syncing to nate (Sync/sync_to_nate.sh)"
  "$PROJECT_DIR/Sync/sync_to_nate.sh" > /dev/null
  pid="$(ssh "${SSH_OPTS[@]}" "$NATE_HOST" "cd '$NATE_PROJECT' && mkdir -p job_files && (nohup '$REMOTE_MATRIX' '$MAX_HOURS' > 'job_files/scaling_matrix.log' 2>&1 < /dev/null & echo \$!)")" \
    || _die "could not launch the matrix on nate"
  # run_scaling_matrix_nate.sh refuses to start if the GPU is already busy, and
  # that refusal only shows up in the log -- so confirm it is actually alive
  # rather than reporting a pid that died a second later.
  if ssh "${SSH_OPTS[@]}" "$NATE_HOST" "sleep 3; pgrep -f '$MATRIX_MATCH' >/dev/null"; then
    echo "==> Matrix running detached on nate (pid $pid)."
    echo "==> Your Mac is free now -- sleep it, close it, disconnect."
    echo "==> Check on it:  $0 --status"
    echo "==> Stop it:      $0 --stop"
    echo "==> When done:    Sync/sync_from_nate.sh && python \"Bayesian Inference/compare_sampler_benchmarks.py\""
  else
    echo "Error: the matrix exited immediately on nate. Log:" >&2
    ssh "${SSH_OPTS[@]}" "$NATE_HOST" "tail -n 20 '$REMOTE_MLOG'" >&2
    exit 1
  fi
  exit 0
fi
# -----------------------------------------------------------------------------

[[ -x "$SUBMIT" ]] || _die "$(basename "$SUBMIT") not found (or not executable) next to this script"
(( DRY_RUN )) || command -v "$NEEDS_CMD" >/dev/null 2>&1 ||
  _die "$NEEDS_CMD not found (Alpine mode needs a login node; --nate runs from your Mac) -- or pass --dry-run"

# Read with while/read, not `mapfile`: --nate runs this under macOS's bash 3.2.
CONFIGS=()
while IFS= read -r cfg; do
  CONFIGS+=("$cfg")
done < <(find "$PROJECT_DIR/Results/GPU Scaling Tests" -mindepth 2 -maxdepth 2 -name solver_params.json | sort)
[[ ${#CONFIGS[@]} -gt 0 ]] || _die "No configs found under Results/GPU Scaling Tests/*/solver_params.json"

echo "Reaction-count-axis scaling matrix: ${#CONFIGS[@]} config(s)"
echo "  $(basename "$SUBMIT") ${ARGS[*]}"
echo ""

if (( NATE && ! DRY_RUN )); then
  echo "==> Syncing to nate once for the whole matrix (Sync/sync_to_nate.sh)"
  "$PROJECT_DIR/Sync/sync_to_nate.sh"
  echo ""
fi

# A config that fails must not strand the ones behind it: on nate the matrix is
# a sequential multi-day run, so aborting at config 3 of 7 overnight would waste
# the rest of the night. Record each outcome and keep going. (Plain strings, not
# arrays: expanding an empty array trips `set -u` on macOS's bash 3.2.)
n_bad=0
n_incomplete=0
summary=""

for cfg in "${CONFIGS[@]}"; do
  label="$(basename "$(dirname "$cfg")")"
  echo "=== $label ==="

  if ! _run "$SUBMIT" --solver-params "$cfg" "${ARGS[@]}"; then
    status="FAILED"
    n_bad=$((n_bad + 1))
    # Carrying on is only safe while nothing else holds the GPU. A launch that
    # failed because a run is still active on nate (submit_inference_nate.sh's
    # already-running guard, or a leftover from an interrupted matrix) means the
    # next config would share the 3080 with it -- which is exactly what the
    # sequential design exists to prevent, and it would spoil both timings.
    if (( NATE && ! DRY_RUN )) && _nate_busy; then
      _die "$label failed while another run is still active on nate. Continuing would double-book the GPU; check it with submit_inference_nate.sh --status (--stop to kill), then re-run this script to resume."
    fi
    echo "  !! $label failed -- continuing with the remaining config(s)"
  elif (( DRY_RUN )); then
    status="dry run"
  elif (( NATE )); then
    case "$(_nate_run_state "$(dirname "${cfg#"$PROJECT_DIR"/}")")" in
      done)       status="done" ;;
      incomplete) status="INCOMPLETE"; n_incomplete=$((n_incomplete + 1)) ;;
      unreachable) status="UNCHECKED"; n_bad=$((n_bad + 1)) ;;
      *)          status="NO OUTPUT"; n_bad=$((n_bad + 1)) ;;
    esac
    echo "  -> $status"
  else
    status="submitted"
  fi

  summary+="$status"$'\t'"$label"$'\n'
  echo ""
done

echo "Summary:"
printf '%s' "$summary" | while IFS=$'\t' read -r status label; do
  printf '  %-11s %s\n' "$status" "$label"
done
echo ""

if (( n_incomplete )); then
  echo "$n_incomplete config(s) INCOMPLETE: checkpointed but short of the draw target."
  echo "Re-run this script to resume them (finished configs are a fast no-op)."
fi
if (( n_bad )); then
  echo -n "$n_bad config(s) need attention -- check the output above"
  if (( NATE )); then echo " and nate's job_files/<config>.log"; else echo ""; fi
fi

# --wait means nate's runs are already done here; Alpine's are only queued.
if (( NATE )); then
  echo "Results are on nate -- pull them down first:  Sync/sync_from_nate.sh"
else
  echo "Once these finish (squeue, or poll for Results/GPU Scaling Tests/*/timing.json):"
fi

cat <<'EOF'

Aggregate the run-time-vs-#reactions curve (test 1) with:

    python "Bayesian Inference/compare_sampler_benchmarks.py"

For convergence vs #reactions (test 2), re-read r-hat/ESS off the same
checkpoint at growing draw counts -- no re-running needed:

    python Utilities/finalize_window.py \
      --solver_params_file "Results/GPU Scaling Tests/<config>/solver_params.json" \
      --burn_in <N> --list

Picking the winner (test 3): among configs that actually converge, take the
lowest total wall-time-to-converged-ESS. That system becomes the fixed base
network for tests 4-7.
EOF

[[ "$n_bad" -eq 0 ]] || exit 1
