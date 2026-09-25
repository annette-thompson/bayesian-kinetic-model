#!/bin/bash
# Trigger check for the multi-param pilot's first submission: true once the
# plot-regen job (28214750) has left the queue AND C20 (tightest_nofloor) has
# converged. C20+unsat and the other still-warming systems are deliberately
# NOT part of this condition -- they have far to go and will run to their own
# 24h cap independently.
ml slurm/blanca 2>/dev/null
regen_running=$(squeue -j 28214750 -h -o '%T' 2>/dev/null)
c20_status=$(/projects/anth4580/software/anaconda/envs/Bayesian/bin/python \
    /projects/anth4580/Bayesian/job_files/warmup_status.py tightest_nofloor --parsable 2>/dev/null \
    | awk -F'|' '$2=="C20"{print $3}')
echo "$(date '+%Y-%m-%d %H:%M:%S') regen_running=[$regen_running] c20_status=[$c20_status]"
[[ -z "$regen_running" && "$c20_status" == "converged" ]]
