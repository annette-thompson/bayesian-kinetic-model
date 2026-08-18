#!/bin/bash
# Submit a chain of dependent Alpine jobs that resume ONE checkpointed BlackJAX
# inference run across the 24h wall-clock cap. Run this on a login node.
#
# Each job runs a segment worker (CPU by default, GPU with --gpu), which samples
# within --max-hours then checkpoints and exits. Jobs are chained with
# --dependency=afterany so a segment that hits the SLURM time limit STILL
# triggers the next one, which resumes from <results_save_dir>/checkpoint/. Once
# the draw target is met the run finalizes (writes netcdf) and any remaining
# queued segments become a fast no-op.
#
# Run longer later: just submit more segments (optionally with --extra-draws on
# the first of the new batch to raise the target).
set -euo pipefail

_die() { echo "Error: $*" >&2; exit 1; }

# Default per-segment budget from SLURM's [D-]HH:MM:SS --time: that many hours
# minus a 15-minute margin to checkpoint in (or the full --time if it is <=15m).
_default_max_hours() {
  awk -F'[-:]' '{
    if      (NF == 4) t = $1*24 + $2 + $3/60 + $4/3600
    else if (NF == 3) t = $1 + $2/60 + $3/3600
    else if (NF == 2) t = $1/60 + $2/3600
    else              t = $1/3600
    printf "%.4f", (t > 0.25) ? t - 0.25 : t
  }' <<< "$1"
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CPU_WORKER="$SCRIPT_DIR/run_inference_segment.sh"
GPU_WORKER="$SCRIPT_DIR/run_inference_segment_gpu.sh"
WORKER=""
USE_GPU=0
MAX_HOURS=""
TIME="24:00:00"
EXTRA_DRAWS=""
SOLVER_PARAMS=""
SEGMENTS=""

usage() {
  cat >&2 <<EOF
Usage: $0 --solver-params FILE --segments N [options]

Required:
  --solver-params FILE   solver_params.json/yaml
  --segments N           number of dependent jobs to submit

Options:
  --gpu                  use the GPU segment worker (aa100/1 GPU) instead of CPU
  --max-hours H          per-segment wall budget before checkpoint (default: --time
                         minus 15 minutes, or the full --time if it is <=15 minutes)
  --time HH:MM:SS        SBATCH --time per job (default $TIME)
  --extra-draws N        raise the draw target; applied to the FIRST submitted
                         segment only (use to run longer than the config's draws)
  --worker PATH          explicit segment sbatch script (overrides --gpu; default:
                         $CPU_WORKER)
  -h, --help             show this help
EOF
  exit 1
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --solver-params) SOLVER_PARAMS="$2"; shift 2 ;;
    --segments)      SEGMENTS="$2";      shift 2 ;;
    --gpu)           USE_GPU=1;          shift ;;
    --max-hours)     MAX_HOURS="$2";     shift 2 ;;
    --time)          TIME="$2";          shift 2 ;;
    --extra-draws)   EXTRA_DRAWS="$2";   shift 2 ;;
    --worker)        WORKER="$2";        shift 2 ;;
    -h|--help)       usage ;;
    *) echo "Unknown argument: $1" >&2; usage ;;
  esac
done

[[ -n "$SOLVER_PARAMS" && -n "$SEGMENTS" ]] || usage
if [[ -z "$WORKER" ]]; then
  if (( USE_GPU )); then WORKER="$GPU_WORKER"; else WORKER="$CPU_WORKER"; fi
fi
[[ -n "$MAX_HOURS" ]] || MAX_HOURS="$(_default_max_hours "$TIME")"

[[ -r "$SOLVER_PARAMS" ]] || _die "Solver params file not readable: $SOLVER_PARAMS"
[[ -r "$WORKER" ]] || _die "Worker script not found: $WORKER"
[[ "$SEGMENTS" =~ ^[0-9]+$ && "$SEGMENTS" -ge 1 ]] || _die "--segments must be a positive integer"
command -v sbatch >/dev/null 2>&1 || _die "sbatch not found (run this on an Alpine login node)"

echo "Submitting $SEGMENTS dependent segment(s):"
echo "  solver params : $SOLVER_PARAMS"
echo "  worker        : $WORKER"
echo "  per-job --time: $TIME   per-segment --max_hours: $MAX_HOURS"
[[ -n "$EXTRA_DRAWS" ]] && echo "  extra draws   : $EXTRA_DRAWS (first segment only)"
echo ""

first_job=""
prev=""
for ((i = 1; i <= SEGMENTS; i++)); do
  cmd=(sbatch --parsable --time="$TIME")
  dep_note="no dependency"
  if [[ -n "$prev" ]]; then
    cmd+=(--dependency="afterany:$prev")
    dep_note="afterany:$prev"
  fi
  cmd+=("$WORKER" "$SOLVER_PARAMS" "$MAX_HOURS")
  [[ "$i" -eq 1 && -n "$EXTRA_DRAWS" ]] && cmd+=("$EXTRA_DRAWS")

  jid=$("${cmd[@]}")
  jid="${jid%%;*}"  # strip any ";cluster" suffix from --parsable

  echo "  segment $i: job $jid  ($dep_note)"
  first_job="${first_job:-$jid}"
  prev="$jid"
done

job_name=$(grep -m1 '^#SBATCH --job-name=' "$WORKER" | cut -d= -f2) || job_name=bayes_seg
echo ""
echo "Submitted. Monitor with:  squeue -u \"$USER\" --name=$job_name"
echo "First job: $first_job   Last job: $prev"
echo "To run longer after these finish: re-run with more --segments"
echo "(add --extra-draws N if the draw target was already met)."
