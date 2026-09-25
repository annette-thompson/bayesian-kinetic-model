#!/bin/bash
# ONE check-and-fix cycle for C20+unsat (tightest_nofloor) -- run this
# repeatedly from a local polling loop, not as its own long-lived process.
# Prints status; exits 0 once converged, 1 if DIED (stop polling, needs a
# human), 2 otherwise (keep polling). If stalled, reapplies the
# RESUB_MAX_TOTAL_HOURS override each normal auto-resubmit won't have (see
# keep_c20_unsat_alive.sh's removed docstring for why the plain 24h default
# won't let this run finish on its own).
ml slurm/blanca 2>/dev/null
PY=/projects/anth4580/software/anaconda/envs/Bayesian/bin/python
WS=/projects/anth4580/Bayesian/job_files/warmup_status.py
RESUB=/projects/anth4580/Bayesian/job_files/chain_scaling_tests/resubmit_if_needed.sh

status=$("$PY" "$WS" tightest_nofloor --parsable 2>/dev/null | awk -F'|' '$2=="C20+unsat"{print $3}')
echo "$(date '+%Y-%m-%d %H:%M:%S') C20+unsat status=[$status]"
case "$status" in
  converged) echo "CONVERGED -- done"; exit 0 ;;
  DIED)      echo "DIED -- stopping, needs a human look"; exit 1 ;;
  stalled)   echo "stalled -- reapplying ceiling override and resubmitting"
             RESUB_MAX_TOTAL_HOURS=200 bash "$RESUB" nofloor_eqxnan_real.sbatch C20+unsat tightest 12.0 keep_alive_watcher
             exit 2 ;;
  *)         exit 2 ;;
esac
