#!/bin/bash
# Decide, from inside a finishing job, whether a successor is warranted -- and
# submit it if so.
#
# Replaces the pre-chained "sbatch --dependency=afterany" pattern. That one
# queues a successor unconditionally at submit time, so a run that converges
# still leaves a job behind to start up, discover there is nothing to do, and
# exit. Deciding here instead means a successor exists only when the work is
# genuinely unfinished.
#
# Called three ways:
#   resubmit_if_needed.sh <script> <system> <width> <maxh> normal
#   resubmit_if_needed.sh <script> <system> <width> <maxh> preempted       (SIGTERM trap)
#   resubmit_if_needed.sh <script> <system> <width> <maxh> watcher_cancel  (see below)
#
# Refuses to resubmit when:
#   * checkpoint phase is "done"            -- the run finished or converged
#   * every chain sits at zero acceptance   -- diverging on every proposal;
#     more wall time cannot help, and this is the case that burned 24 h on C10
#   * the segment cap is reached            -- stops a broken run looping forever
#   * another caller already claimed this segment -- see the lock below
#   * total A100-EQUIVALENT compute across the run's segments exceeds its max_total_hours (from
#     its solver_params.json, else RESUB_MAX_TOTAL_HOURS, default 24) -- a hard
#     ceiling so a slow system (C18+unsat, C20+unsat) cannot keep quietly
#     re-queueing itself for days without a human looking at it
#
# 2026-09-09: a job's own TERM trap is not reliable as the *only* path here. Bash
# only runs a pending trap once its foreground `wait()` on the python child
# returns, and if python does not exit before SLURM's kill escalation
# (SIGTERM -> SIGKILL) the whole process group is killed before the trap can
# call this script -- lost 3 of 9 jobs to exactly this race in one rotation.
# Anything that deliberately cancels a job (cancel_and_resubmit.sh) now calls
# this script itself right after `scancel`, instead of trusting the trap. Both
# paths can still fire for the same cancellation (trap wins the race
# sometimes), so the segment lock below makes double-submission impossible
# rather than just unlikely.

set -uo pipefail
SCRIPT="${1:?script}"; SYSTEM="${2:?system}"; WIDTH="${3:?width}"
MAXH="${4:?maxh}";     WHY="${5:-normal}"
MC="/projects/anth4580/Bayesian/job_files/masking_check"
BASE="/projects/anth4580/Bayesian/Results/Chain Scaling Tests"
WALL="${RESUB_WALL:-12:15:00}"
MAX_SEGMENTS="${RESUB_MAX_SEGMENTS:-12}"
MAX_TOTAL_HOURS="${RESUB_MAX_TOTAL_HOURS:-24}"

# WIDTH is normally a bare label ("tightest"/"narrowest") wrapped in the old
# "a1 $WIDTH [nofloor-eqxnan]" template. The 2026-09-09 rename moved to
# prior-value-based directory names and, for multi-parameter configs, a
# $PARAMSET label (e.g. "a1c2_no_floor") that's already the FULL suffix on its
# own -- detected by already ending in _floor/_no_floor, so it's used as-is
# rather than double-wrapped.
case "$WIDTH" in
  *_floor|*_no_floor) RUN_DIR="$BASE/Chain $SYSTEM - $WIDTH" ;;
  *)
    case "$SCRIPT" in
      *nofloor*) RUN_DIR="$BASE/Chain $SYSTEM - a1 $WIDTH nofloor-eqxnan" ;;
      *)         RUN_DIR="$BASE/Chain $SYSTEM - a1 $WIDTH" ;;
    esac
    ;;
esac
CKPT="$RUN_DIR/checkpoint"
COUNTER="$CKPT/.segments"

_say() { echo "==> resubmit[$SYSTEM/$WIDTH]: $*"; }

[[ -d "$CKPT" ]] || { _say "no checkpoint dir; not resubmitting"; exit 0; }

phase=$(python3 -c "
import json,sys
try:
    print(json.load(open(sys.argv[1]))['phase'])
except Exception:
    print('unknown')
" "$CKPT/checkpoint_meta.json" 2>/dev/null)

if [[ "$phase" == "done" ]]; then
    _say "phase=done -- finished, not resubmitting"; exit 0
fi

# Total-compute ceiling: compute banked across every segment, as the sampler
# records it in checkpoint_meta.json at each checkpoint. Until 2026-09-14 this
# was wall-clock since the first checkpoint, which counted queue waits and
# outages as spent work -- runs left idle by a broken resubmission read as far
# past a 24h cap they had barely touched. The cap comes from the run's own
# solver_params.json when set, so this script and the sampler can't disagree.
# A checkpoint from before the counter existed reads as 0 and is let through,
# matching the sampler; an unreadable one is let through rather than blocked
# on a false unknown.
time_check=$(python3 -c "
import json, sys
meta, cfg, fallback = sys.argv[1], sys.argv[2], float(sys.argv[3])
try:
    m = json.load(open(meta))
    # A100-equivalent (wall x card speed factor) when the sampler recorded it.
    used = float(m.get('a100_equiv_seconds', m.get('compute_seconds', 0.0))) / 3600.0
except Exception:
    print('unknown'); sys.exit()
cap = fallback
try:
    v = json.load(open(cfg)).get('posterior_sampling', {}).get('max_total_hours', fallback)
    cap = None if v is None else float(v)
except Exception:
    pass
if cap is None:
    print('OK %.1f unbounded' % used)
else:
    print(('STOP' if used >= cap else 'OK') + ' %.1f %g' % (used, cap))
" "$CKPT/checkpoint_meta.json" "$RUN_DIR/solver_params.json" "$MAX_TOTAL_HOURS" 2>/dev/null)
if [[ "$time_check" == STOP* ]]; then
    read -r _ used cap <<< "$time_check"
    _say "total compute ${used} h >= cap ${cap}h -- stopping for review, not resubmitting"
    exit 0
fi

# All-chains-dead check. Mirrors the sampler-side guard so a preemption that
# skips the sampler's own exit path still cannot resurrect a dead run.
dead=$(/projects/anth4580/software/anaconda/envs/Bayesian/bin/python - "$CKPT/draws.zarr" <<'PY' 2>/dev/null
import sys, os
try:
    import zarr, numpy as np
    p = sys.argv[1]
    if not os.path.exists(p):
        print("unknown"); raise SystemExit
    ar = np.asarray(zarr.open(p, mode="r")["warmup_stats/acceptance_rate"][:])
    if ar.size == 0:
        print("unknown"); raise SystemExit
    if ar.ndim == 2 and ar.shape[0] < ar.shape[1]:
        ar = ar.T
    n = ar.shape[1] if ar.ndim == 2 else 1
    d = sum(1 for c in range(n) if ar[:, c].mean() < 1e-6)
    print("dead" if d >= n else "alive")
except Exception:
    print("unknown")
PY
)
if [[ "$dead" == "dead" ]]; then
    _say "every chain at zero acceptance -- not resubmitting (see warmup_status STATUS=DIED)"
    exit 0
fi

n=$(cat "$COUNTER" 2>/dev/null || echo 0)
if (( n >= MAX_SEGMENTS )); then
    _say "segment cap $MAX_SEGMENTS reached -- not resubmitting"; exit 0
fi

# Atomic dedup: mkdir is the one filesystem op POSIX guarantees only one
# concurrent caller can win. Keyed on the segment about to be created, so the
# trap and an external cancel_and_resubmit.sh racing for the SAME transition
# never both submit -- whichever loses the mkdir exits quietly instead. Lock
# dirs are left behind deliberately, as a small audit trail of who resubmitted
# what and when (mtime + WHY on each).
LOCK="$CKPT/.resubmit_lock_seg_$((n + 1))"
if ! mkdir "$LOCK" 2>/dev/null; then
    _say "segment $((n+1)) already claimed by another caller (reason=$WHY) -- skipping, not a failure"
    exit 0
fi
echo "$WHY" > "$LOCK/reason" 2>/dev/null || true
echo $((n + 1)) > "$COUNTER"

# Twin submission: queued on Blanca and Alpine at once; the first to start runs the
# segment and cancels the other (gpu_twin_claim.sh). The job registers its own ID in
# the set's jobids file when it wins, so nothing is registered here -- a twin that
# never runs must not be counted.
out=$(bash /projects/anth4580/Bayesian/job_files/gpu_submit.sh --time "$WALL" -- \
      "$MC/$SCRIPT" "$SYSTEM" "$WIDTH" "$MAXH" 2>&1)
if grep -qE '^(blanca|alpine) [0-9]+$' <<< "$out"; then
    _say "queued $(grep -E '^(blanca|alpine) [0-9]+$' <<< "$out" | tr '\n' ' ')(segment $((n+1))/$MAX_SEGMENTS, reason=$WHY, phase=$phase, wall=$WALL)"
    # A cluster that refused the job still matters -- without this its error was
    # swallowed and the pair silently became a single-cluster submission.
    grep -E '^gpu_submit\.sh:' <<< "$out" | while read -r line; do _say "$line"; done
else
    _say "gpu_submit FAILED: $out"
fi
