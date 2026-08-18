#!/bin/bash
# Launch (or check on) a BlackJAX inference run on nate -- a local lab Ubuntu
# workstation (Xeon Silver 4208 + RTX 3080) reached via `ssh nate` -- as the
# non-Alpine counterpart to submit_inference_chain.sh. Run this from your Mac.
#
# nate has no SLURM: no queue, no wall-clock kill, and one GPU. So there's no
# segment/dependency chaining here -- this pushes the repo to nate (see
# Sync/sync_to_nate.sh), then launches ONE run_inference_nate.sh process in
# the background over ssh (nohup + fully redirected stdio, so it survives
# your ssh session dropping), and returns immediately unless --wait is given.
#
# Usage:
#   Bayesian\ Inference/submit_inference_nate.sh --solver-params FILE [options]
set -euo pipefail

_die() { echo "Error: $*" >&2; exit 1; }

# Fail fast (instead of hanging) if nate is unreachable -- e.g. off the VPN
# or the machine is asleep. ControlMaster/ControlPersist in ~/.ssh/config
# (if set for the "nate" host) keeps reconnects in --wait cheap regardless.
SSH_OPTS=(-o ConnectTimeout=10 -o BatchMode=yes)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
source "$PROJECT_DIR/Sync/nate_sync_config.sh"   # NATE_HOST, NATE_PROJECT, LOCAL_BASE
SYNC_TO="$PROJECT_DIR/Sync/sync_to_nate.sh"
WORKER_REL="Bayesian Inference/run_inference_nate.sh"
POLL_SECONDS=30

SOLVER_PARAMS=""
MAX_HOURS=""
EXTRA_DRAWS=""
NO_SYNC=0
WAIT=0
ACTION="launch"

usage() {
  cat >&2 <<EOF
Usage: $0 --solver-params FILE [options]

Required:
  --solver-params FILE   solver_params.json/yaml (local path, inside "$LOCAL_BASE")

Options:
  --max-hours H     optional wall-clock budget before checkpoint+exit (default:
                     unbounded -- nate has no wall-clock kill, so the run just
                     samples to the draw target)
  --extra-draws N   raise the draw target before running
  --no-sync         skip the Sync/sync_to_nate.sh push (assumes already synced)
  --wait            block here, polling nate every ${POLL_SECONDS}s until the run exits
  --status          report whether this config is running on nate + tail its log, then exit
  --stop            SIGTERM the running process for this config on nate, then exit
  -h, --help        show this help
EOF
  exit 1
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --solver-params) SOLVER_PARAMS="$2"; shift 2 ;;
    --max-hours)     MAX_HOURS="$2";     shift 2 ;;
    --extra-draws)   EXTRA_DRAWS="$2";   shift 2 ;;
    --no-sync)       NO_SYNC=1;          shift ;;
    --wait)          WAIT=1;             shift ;;
    --status)        ACTION="status";    shift ;;
    --stop)          ACTION="stop";      shift ;;
    -h|--help)       usage ;;
    *) echo "Unknown argument: $1" >&2; usage ;;
  esac
done

[[ -n "$SOLVER_PARAMS" ]] || usage

SOLVER_PARAMS_ABS="$(cd "$(dirname "$SOLVER_PARAMS")" && pwd)/$(basename "$SOLVER_PARAMS")"
case "$SOLVER_PARAMS_ABS" in
  "$LOCAL_BASE"/*) REL_PATH="${SOLVER_PARAMS_ABS#"$LOCAL_BASE"/}" ;;
  *) _die "Solver params file must be inside $LOCAL_BASE (so it can be synced to nate): $SOLVER_PARAMS" ;;
esac
REMOTE_SOLVER_PARAMS="$NATE_PROJECT/$REL_PATH"
# The leading [i] is load-bearing. ssh runs its remote command as `bash -c
# '<command>'`, so the pattern text appears verbatim in that shell's own command
# line -- and `inference_runner.py.*<path>` matches it, since regex `.*` happily
# matches the literal ".*". Every pgrep here would then find the shell asking the
# question and report a run that doesn't exist. Writing it as [i]nference makes
# the pattern unable to match its own text while still matching the real python
# process.
#
# Match on REL_PATH, not REMOTE_SOLVER_PARAMS: the launcher cds into the project
# directory first, so the path the python process actually carries on its command
# line is project-relative, without the "$NATE_PROJECT/" prefix.
MATCH="[i]nference_runner.py.*${REL_PATH}"
LABEL="$(basename "$(dirname "$REL_PATH")")"
LABEL_SAFE="${LABEL// /_}"
REMOTE_LOG="$NATE_PROJECT/job_files/${LABEL_SAFE}.log"

# pgrep's exit code is swallowed by the `| head -n1` pipe (head exits 0
# whether or not it got input), so ssh's own exit status here is 255 iff ssh
# itself couldn't connect/authenticate -- never "no process matched". That
# lets us tell "confirmed not running" apart from "couldn't check", which
# matters: silently treating "unreachable" as "not running" could double-
# launch a job that's actually already running on nate.
_remote_pid() {
  local out rc
  set +e
  out="$(ssh "${SSH_OPTS[@]}" "$NATE_HOST" "pgrep -f '$MATCH' | head -n1" 2>&1)"
  rc=$?
  set -e
  [[ $rc -ne 255 ]] || _die "Could not reach nate over ssh (host: $NATE_HOST) -- check VPN/network/that it's awake ($out)"
  printf '%s' "$out"
}

_tail_log() {
  ssh "${SSH_OPTS[@]}" "$NATE_HOST" "tail -n 20 '$REMOTE_LOG' 2>/dev/null"
}

if [[ "$ACTION" == "status" ]]; then
  pid="$(_remote_pid)"
  if [[ -n "$pid" ]]; then
    echo "==> $LABEL: RUNNING on nate (pid $pid)"
  else
    echo "==> $LABEL: not running on nate"
  fi
  echo "--- last log lines ($REMOTE_LOG) ---"
  _tail_log || true
  exit 0
fi

if [[ "$ACTION" == "stop" ]]; then
  pid="$(_remote_pid)"
  [[ -n "$pid" ]] || _die "$LABEL: no matching run found on nate"
  echo "==> Stopping $LABEL (pid $pid) on nate"
  ssh "${SSH_OPTS[@]}" "$NATE_HOST" "kill $pid"
  exit 0
fi

[[ -r "$SOLVER_PARAMS" ]] || _die "Solver params file not readable: $SOLVER_PARAMS"

if [[ "$NO_SYNC" -eq 0 ]]; then
  echo "==> Syncing to nate first (Sync/sync_to_nate.sh; use --no-sync to skip)"
  "$SYNC_TO"
  echo ""
fi

existing_pid="$(_remote_pid)"
[[ -z "$existing_pid" ]] || _die "$LABEL already running on nate (pid $existing_pid) -- stop it first with --stop, or use --status/--wait"

echo "Launching on nate:"
echo "  solver params : $SOLVER_PARAMS"
echo "  remote path   : $REMOTE_SOLVER_PARAMS"
echo "  max hours     : ${MAX_HOURS:-unbounded}"
[[ -n "$EXTRA_DRAWS" ]] && echo "  extra draws   : $EXTRA_DRAWS"
echo "  remote log    : $REMOTE_LOG"
echo ""

# Everything after the cd must be PROJECT-relative, not home-relative: the log
# and the solver params both live under $NATE_PROJECT, so prefixing them with it
# again would look for $NATE_PROJECT/$NATE_PROJECT/... The redirect is what fails
# first, which kills the launch outright while still reporting a pid from $!.
REMOTE_CMD="cd '$NATE_PROJECT' && mkdir -p job_files && (nohup '$WORKER_REL' '$REL_PATH' '$MAX_HOURS' '$EXTRA_DRAWS' > 'job_files/${LABEL_SAFE}.log' 2>&1 < /dev/null & echo \$!)"
launch_pid="$(ssh "${SSH_OPTS[@]}" "$NATE_HOST" "$REMOTE_CMD")"

echo "==> Launched (pid ${launch_pid:-unknown} on nate)"
echo "==> Monitor:  ssh $NATE_HOST tail -f '$REMOTE_LOG'"
echo "==> Or:       $0 --solver-params \"$SOLVER_PARAMS\" --status"
echo "==> Pull results when done:  Sync/sync_from_nate.sh"

if [[ "$WAIT" -eq 1 ]]; then
  echo ""
  echo "==> --wait: polling every ${POLL_SECONDS}s until the run exits..."
  # Unlike _remote_pid, a transient unreachable nate here just warns and
  # keeps polling instead of aborting -- a multi-hour run shouldn't be
  # killed by one flaky network blip.
  while true; do
    set +e
    pid="$(ssh "${SSH_OPTS[@]}" "$NATE_HOST" "pgrep -f '$MATCH' | head -n1" 2>&1)"
    rc=$?
    set -e
    if [[ $rc -eq 255 ]]; then
      echo "    (nate unreachable, will retry: $pid)" >&2
      sleep "$POLL_SECONDS"
      continue
    fi
    [[ -n "$pid" ]] || break
    sleep "$POLL_SECONDS"
  done
  echo "==> $LABEL finished (no longer running on nate)"
fi
