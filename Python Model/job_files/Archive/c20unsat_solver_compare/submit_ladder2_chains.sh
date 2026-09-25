#!/bin/bash
# Submits, for each of the 4 candidates B/C/D/E, a chain of 13 dependent SLURM
# jobs (one per non-C20+unsat ladder rung) via --dependency=afterok, so each
# system only starts once the previous one has PASSED (exit 0). A rung that
# fails the min_total_kept>=10 check makes generate_and_score_ladder2.py exit
# nonzero, which SLURM's afterok treats as a failed dependency -- every later
# job in that candidate's chain is then never run (shows as
# DependencyNeverSatisfied and gets cancelled). Run this ON Blanca (ssh curc).
set -e
JOBDIR="/projects/anth4580/Bayesian/job_files/c20unsat_solver_compare"
cd "$JOBDIR"

SYSTEMS=(C4_NoFB C6 C8 C10 C12 "C12+unsat" C14 "C14+unsat" C16 "C16+unsat" C18 "C18+unsat" C20)
# Candidates to submit chains for come from argv, defaulting to B C D E so old
# invocations (no args) behave exactly as before. Pass e.g. "Q" to add just
# one more candidate's chain without resubmitting the others.
if [[ $# -gt 0 ]]; then
  CANDIDATES=("$@")
else
  CANDIDATES=(B C D E)
fi

: >> job_ids_ladder2.txt
for CAND in "${CANDIDATES[@]}"; do
  PREV=""
  for SYS in "${SYSTEMS[@]}"; do
    if [[ -z "$PREV" ]]; then
      JID=$(sbatch -M blanca --job-name="ladder_${CAND}" --parsable \
            generate_and_score_ladder2.sbatch "$SYS" "$CAND" | cut -d';' -f1)
    else
      JID=$(sbatch -M blanca --job-name="ladder_${CAND}" --parsable \
            --dependency=afterok:"$PREV" \
            generate_and_score_ladder2.sbatch "$SYS" "$CAND" | cut -d';' -f1)
    fi
    echo "candidate=$CAND system=$SYS job=$JID prev=$PREV" | tee -a job_ids_ladder2.txt
    PREV="$JID"
  done
done
