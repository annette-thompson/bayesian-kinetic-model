#!/bin/bash -l
# Stage 2 of Notes/tier1_experiment_plan.md section 3: R1, R3, R5, R8 and the R4 pilot, 23 runs,
# each one tier1.sbatch job queued on Blanca and Alpine through gpu_submit.sh.
#
#   tier1/submit_stage2.sh --dry_run          # what would be submitted, and why anything is skipped
#   tier1/submit_stage2.sh                    # submit all of it
#   tier1/submit_stage2.sh --only R1,R3       # submit some of the groups
#
# Guard: the plan holds Stage 2 until R0 has been inspected, so this refuses to submit until R0
# ("Tier1 C8 - a1c3") has finalized (posterior_samples_pm.nc exists). --force skips that check.
# A run is skipped when it has already finalized, when its config is missing, or when a job with
# its name is already queued or running on either cluster, so rerunning this resubmits only the
# runs that stopped at the wall clock (they resume from their checkpoints).
set -uo pipefail
type module >/dev/null 2>&1 || source /etc/profile >/dev/null 2>&1
BASE=/projects/anth4580/Bayesian
RESULTS="$BASE/Results/Tier1"
R0_RUN="Tier1 C8 - a1c3"

declare -A GROUP
GROUP[R1]="Tier1 C14+unsat - a1c3"
GROUP[R3]="Tier1 C8 - d1d2|Tier1 C8 - d1d2 - dense|Tier1 C14+unsat - d1d2"
GROUP[R5]="Tier1 C8_noise5 - a1c3|Tier1 C8_noise20 - a1c3|Tier1 C8_noise40 - a1c3|Tier1 C8 - a1c3 - prior+1sd|Tier1 C8 - a1c3 - prior+2sd|Tier1 C8 - a1c3 - prior+3sd|Tier1 C8 - a1c3 - prior+4sd"
GROUP[R8]="Tier1 C8 - a1c3 - ta0.95|Tier1 C8 - a1c3 - rtol1e-5"
GROUP[R4]="$(for i in $(seq 0 9); do printf 'Tier1 C8_sbc%03d - a1c3|' "$i"; done)"
ORDER="R1 R3 R5 R8 R4"

DRY=0; FORCE=0; ONLY=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry_run) DRY=1; shift ;;
    --force) FORCE=1; shift ;;
    --only) ONLY="$2"; shift 2 ;;
    *) echo "submit_stage2.sh: unknown option $1" >&2; exit 2 ;;
  esac
done
if [[ -n "$ONLY" ]]; then
  for g in ${ONLY//,/ }; do
    [[ -n "${GROUP[$g]+x}" ]] || { echo "submit_stage2.sh: no group $g (have: $ORDER)" >&2; exit 2; }
  done
  ORDER="${ONLY//,/ }"
fi

if [[ ! -e "$RESULTS/$R0_RUN/posterior_samples_pm.nc" ]]; then
  if (( FORCE )); then
    echo "R0 has not finalized; submitting anyway (--force)"
  elif (( DRY )); then
    echo "R0 has not finalized: a real submission would stop here (--force overrides)"
  else
    echo "R0 (\"$R0_RUN\") has not finalized, and the plan holds Stage 2 until it has been inspected." >&2
    echo "Check it, then rerun; --force skips this check." >&2
    exit 1
  fi
fi

# Job names on both clusters (gpu_submit.sh names both twins after --name). Same cluster switch
# as gpu_submit.sh: the module alone does not move SLURM_CONF.
QUEUED=""
for c in blanca alpine; do
  module unload slurm >/dev/null 2>&1; module load "slurm/$c" >/dev/null 2>&1
  QUEUED+=$'\n'"$(SLURM_CONF="/curc/slurm/$c/etc/slurm.conf" squeue -h -u "$USER" -o %j 2>/dev/null)"
done

cd "$BASE/job_files" || exit 1
n_sub=0; n_skip=0
for g in $ORDER; do
  IFS='|' read -r -a runs <<< "${GROUP[$g]}"
  for run in "${runs[@]}"; do
    [[ -n "$run" ]] || continue
    slug="tier1_$(echo "${run#Tier1 }" | sed 's/ - /_/g; s/ /_/g')"
    if [[ -e "$RESULTS/$run/posterior_samples_pm.nc" ]]; then
      echo "skip  $g  $run  (finalized)"; n_skip=$((n_skip + 1)); continue
    fi
    if [[ ! -r "$RESULTS/$run/solver_params.json" ]]; then
      echo "skip  $g  $run  (no solver_params.json: sync or rebuild the config)"; n_skip=$((n_skip + 1)); continue
    fi
    if grep -qxF "$slug" <<< "$QUEUED"; then
      echo "skip  $g  $run  (a job named $slug is already queued or running)"; n_skip=$((n_skip + 1)); continue
    fi
    if (( DRY )); then
      echo "would $g  $run  -> ./gpu_submit.sh --time 12:15:00 --name $slug -- tier1/tier1.sbatch \"$run\""
    else
      echo "==>   $g  $run"
      ./gpu_submit.sh --time 12:15:00 --name "$slug" -- tier1/tier1.sbatch "$run" || { echo "submission failed: $run" >&2; exit 1; }
      sleep 1
    fi
    n_sub=$((n_sub + 1))
  done
done
echo "$( (( DRY )) && echo "would submit" || echo "submitted" ) $n_sub, skipped $n_skip"
