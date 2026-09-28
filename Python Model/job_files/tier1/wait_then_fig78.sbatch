#!/bin/bash
#SBATCH --job-name=tier1_fig78_wait
#SBATCH --partition=blanca
#SBATCH --qos=preemptable
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=18:00:00
#SBATCH --output=tier1/fig78_wait.%j.out
# Waits for R2 to finalize, then submits tier1/posterior_figs.sbatch (Figs 7 and 8).
#   sbatch tier1/wait_then_fig78.sbatch <epoch>     (from job_files/)
# <epoch>: R2's posterior file must be newer than this. A reopened run keeps its earlier
# posterior until the new one is written, and its checkpoint reads "done" for the hour of
# finalizing before that, so the checkpoint alone would start the figures on the old posterior.
# An argument rather than the job's start time, so a requeue after preemption keeps it.
AFTER="${1:?epoch that the new R2 posterior must postdate}"
R2="/projects/anth4580/Bayesian/Results/Tier1/Tier1 C14+unsat - a1c3a2"
cd /projects/anth4580/Bayesian/job_files || exit 1
r2_done() {
  python3 -c "
import os, sys
sys.path.insert(0, 'tier1')
from tier1_status import is_finalized
run, after = sys.argv[1], float(sys.argv[2])
nc = os.path.join(run, 'posterior_samples_pm.nc')
sys.exit(0 if is_finalized(run) and os.path.getmtime(nc) > after else 1)" "$R2" "$AFTER"
}
until r2_done; do sleep 600; done
echo "==> R2 finalized $(date '+%a %H:%M'); submitting Figs 7 and 8"
module load slurm/blanca >/dev/null 2>&1
env -u SLURM_JOB_ID -u SLURM_JOBID sbatch tier1/posterior_figs.sbatch
