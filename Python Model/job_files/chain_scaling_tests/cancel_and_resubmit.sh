#!/bin/bash
# Cancel a job and submit its successor from here, rather than trusting the
# doomed job's own TERM trap to run before SLURM's kill escalation reaps it.
#
# 2026-09-09: a checkpoint-then-cancel rotation of 9 jobs lost 3 successors to
# exactly that race (bash only runs a pending trap once its foreground `wait`
# on the python child returns; SLURM's SIGKILL can beat python to exiting).
# Any script that deliberately cancels a run to rotate it onto new code
# should call THIS instead of a bare `scancel`. resubmit_if_needed.sh's own
# segment lock makes it safe even if the trap also fires for the same
# cancellation -- whichever caller loses the race just skips, not a failure.
#
# Usage: cancel_and_resubmit.sh <jobid> <sbatch_script> <system> <width> <maxh>
#   e.g. cancel_and_resubmit.sh 28209601 nofloor_eqxnan_real.sbatch C16+unsat tightest 12.0
set -uo pipefail
JOBID="${1:?jobid}"; SCRIPT="${2:?sbatch_script}"; SYSTEM="${3:?system}"
WIDTH="${4:?width}"; MAXH="${5:?maxh}"
MC="/projects/anth4580/Bayesian/job_files/masking_check"

scancel "$JOBID"
bash "$MC/resubmit_if_needed.sh" "$SCRIPT" "$SYSTEM" "$WIDTH" "$MAXH" watcher_cancel
