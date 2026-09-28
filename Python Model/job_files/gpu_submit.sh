#!/bin/bash -l
# Submit one GPU job to Blanca and Alpine at once; whichever copy starts first runs it.
# Only cards at least as fast as an A100 in double precision, which is what the ODE solves
# run in:
#
#   Blanca: preemptable, a full A100 or H100 (the Intel H100 node -- the AMD H100 nodes are
#           split into MIG slices, which a bare gpu:1 could land on). No V100 (0.67-0.75x).
#   Alpine: three copies, one per card type: a full 40 GB A100 and a full 80 GB A100 on
#           aa100, and a full H200 on ah200. Typed requests keep off the MIG slices; the
#           gpu-normal QOS allows 6, 3 and 4 of them per user.
#
# The job script must source gpu_twin_claim.sh before doing any work: the first copy to
# start claims the work and cancels the others; a copy that starts later exits.
#
# Time. --time is used as-is on both clusters: right for resumable segment jobs, whose
# limit is a budget they checkpoint within. --a100_hours H is for work of a fixed size
# measured on an A100: each cluster's limit becomes H / (slowest card it can land on),
# using the measured speed factors in resumable_sampler.GPU_SPEED_VS_A100 -- 1.00 (A100)
# on both, now that V100s are excluded. Alpine's gpu-normal QOS caps at 24 h.
#
# Usage:
#   gpu_submit.sh [--time HH:MM:SS | --a100_hours H] [--cpus N] [--mem M] [--name NAME]
#                 [--nice N] [--blanca_only | --alpine_only] [--a100_only] -- <sbatch script> [args...]
# --nice N lowers the job's priority by N (sbatch --nice) on both clusters: for bulk work
# that should wait behind this user's other jobs, e.g. the SBC replicates.
# Prints "blanca <jobid>" / "alpine <jobid>" and records them in gpu_claims/<token>.jobs.
set -uo pipefail
# Called from inside jobs too, where the module function may not be inherited.
type module >/dev/null 2>&1 || source /etc/profile >/dev/null 2>&1
CLAIMS=/projects/anth4580/Bayesian/job_files/gpu_claims
ALPINE_ACCOUNT=ucb634_asc2
BLANCA_SLOWEST=1.00     # A100; with --a100_only too
ALPINE_SLOWEST=1.00
ALPINE_MAX_S=$((24 * 3600))

TIME=""; A100_H=""; CPUS=8; MEM=""; NAME=""; NICE=""; WHERE=both; A100_ONLY=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --time) TIME="$2"; shift 2 ;;
    --a100_hours) A100_H="$2"; shift 2 ;;
    --cpus) CPUS="$2"; shift 2 ;;
    --mem) MEM="$2"; shift 2 ;;
    --name) NAME="$2"; shift 2 ;;
    --nice) NICE="$2"; shift 2 ;;
    --blanca_only) WHERE=blanca; shift ;;
    --a100_only) A100_ONLY=1; shift ;;
    --alpine_only) WHERE=alpine; shift ;;
    --) shift; break ;;
    *) echo "gpu_submit.sh: unknown option $1" >&2; exit 2 ;;
  esac
done
[[ $# -ge 1 ]] || { echo "gpu_submit.sh: no sbatch script given" >&2; exit 2; }
[[ -n "$TIME" || -n "$A100_H" ]] || { echo "gpu_submit.sh: give --time or --a100_hours" >&2; exit 2; }

# Switching clusters means switching SLURM_CONF, and the module alone cannot be trusted
# to do it: "module load slurm/alpine" over a loaded slurm/blanca reports a swap but
# leaves SLURM_CONF alone, and inside a job the node's own /etc/slurm/slurm.conf wins
# regardless (which silently sent every in-job Alpine submission to Blanca, where aa100
# does not exist). Set it explicitly; both clusters share one sbatch binary.
_use_cluster() {
  module unload slurm >/dev/null 2>&1; module load "slurm/$1" >/dev/null 2>&1
  export SLURM_CONF="/curc/slurm/$1/etc/slurm.conf"
}
_hms() { local s=$1; printf "%02d:%02d:%02d" $((s / 3600)) $((s % 3600 / 60)) $((s % 60)); }
_limit() {  # $1 slowest speed factor, $2 optional max seconds
  if [[ -n "$A100_H" ]]; then
    local s; s=$(python3 -c "import math,sys; print(int(math.ceil(float(sys.argv[1]) * 3600 / float(sys.argv[2]))))" "$A100_H" "$1")
    [[ -n "${2:-}" && $s -gt $2 ]] && s=$2
    _hms "$s"
  else
    echo "$TIME"
  fi
}

TOKEN="$(date +%Y%m%d-%H%M%S)-$$-$RANDOM"
mkdir -p "$CLAIMS"
COMMON=(--parsable --nodes=1 --ntasks=1 --cpus-per-task="$CPUS" --comment="twin:$TOKEN")
[[ -n "$MEM" ]] && COMMON+=(--mem="$MEM")
[[ -n "$NAME" ]] && COMMON+=(--job-name="$NAME")
[[ -n "$NICE" ]] && COMMON+=(--nice="$NICE")

# Test hook: GPU_SUBMIT_PROFILE=cpu_test swaps in CPU partitions so the claim/cancel
# logic can be exercised without waiting on a GPU queue.
if [[ "${GPU_SUBMIT_PROFILE:-gpu}" == cpu_test ]]; then
  BLANCA=(--partition=blanca --qos=preemptable)
  ALPINE=(--partition=acpu --qos=cpu-normal --account="$ALPINE_ACCOUNT")
else
  # --a100_only: for timing comparisons that must not mix card types -- A100s only, and on
  # Alpine only the 40 GB kind.
  if [[ $A100_ONLY -eq 1 ]]; then
    BLANCA=(--partition=blanca --qos=preemptable --gres=gpu:a100:1 --constraint=A100)
  else
    BLANCA=(--partition=blanca --qos=preemptable --gres=gpu:1 --constraint="A100|(h100&xeon)")
  fi
  ALPINE_BASE=(--qos=gpu-normal --account="$ALPINE_ACCOUNT")
  ALPINE_A100_40=("${ALPINE_BASE[@]}" --partition=aa100 --gres=gpu:a100-40gb:1)
  ALPINE_A100_80=("${ALPINE_BASE[@]}" --partition=aa100 --gres=gpu:a100_80gb:1)
  ALPINE_H200=("${ALPINE_BASE[@]}" --partition=ah200 --gres=gpu:h200:1)
fi

# The job id is stdout's last line (--parsable); Alpine prints submission warnings on
# stderr, which must not be mistaken for a failure.
_submit() {  # $1 cluster name, rest: sbatch args
  local cl=$1; shift
  local err; err=$(mktemp)
  # Job-step variables inherited from a running job confuse a submission aimed at
  # another cluster, so drop them for the call.
  local id; id=$(env -u SLURM_JOB_ID -u SLURM_JOBID -u SLURM_CLUSTER_NAME -u SLURM_NODELIST \
                     -u SLURM_JOB_PARTITION -u SLURM_JOB_QOS -u SLURM_JOB_ACCOUNT \
                     sbatch "$@" 2>"$err" | tail -1)
  if [[ "$id" =~ ^[0-9]+$ ]]; then echo "$cl $id" | tee -a "$CLAIMS/$TOKEN.jobs.tmp"
  else echo "gpu_submit.sh: $cl sbatch failed: $(cat "$err")" >&2; fi
  rm -f "$err"
}
: > "$CLAIMS/$TOKEN.jobs.tmp"
if [[ "$WHERE" != alpine ]]; then
  _use_cluster blanca
  _submit blanca "${COMMON[@]}" "${BLANCA[@]}" --time="$(_limit "$BLANCA_SLOWEST")" "$@"
fi
if [[ "$WHERE" != blanca ]]; then
  _use_cluster alpine
  if [[ "${GPU_SUBMIT_PROFILE:-gpu}" == cpu_test ]]; then
    _submit alpine "${COMMON[@]}" "${ALPINE[@]}" --time="$(_limit "$ALPINE_SLOWEST" "$ALPINE_MAX_S")" "$@"
  else
    variants=(ALPINE_A100_40)
    [[ $A100_ONLY -eq 1 ]] || variants+=(ALPINE_A100_80 ALPINE_H200)
    for v in "${variants[@]}"; do
      declare -n args="$v"
      _submit alpine "${COMMON[@]}" "${args[@]}" --time="$(_limit "$ALPINE_SLOWEST" "$ALPINE_MAX_S")" "$@"
      unset -n args
    done
  fi
fi
mv "$CLAIMS/$TOKEN.jobs.tmp" "$CLAIMS/$TOKEN.jobs"
echo "$(date +%FT%T) $TOKEN $(tr '\n' ' ' < "$CLAIMS/$TOKEN.jobs")-- $*" >> "$CLAIMS/submissions.log"
[[ -s "$CLAIMS/$TOKEN.jobs" ]]
