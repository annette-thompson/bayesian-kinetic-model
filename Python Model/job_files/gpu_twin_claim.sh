# Sourced by a job script submitted through gpu_submit.sh, before it does any work:
#
#   source /projects/anth4580/Bayesian/job_files/gpu_twin_claim.sh || exit 0
#
# The first twin to start claims the work (an atomic mkdir on the shared /projects
# filesystem) and cancels the twin still queued on the other cluster. A twin that
# starts second finds the claim taken and returns 1, so its script exits without
# running. A requeued job (Blanca preemption keeps the job ID) recognises its own
# claim and carries on. A job not submitted as a twin (no twin: comment) is untouched.
# Also exports GPU_MODEL for resumable_sampler's A100-equivalent time accounting.

_gtc_claims=/projects/anth4580/Bayesian/job_files/gpu_claims
export GPU_MODEL="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"
_gtc_tok=$(scontrol show job "${SLURM_JOB_ID:-0}" 2>/dev/null | grep -o 'Comment=twin:[^ ]*' | head -1)
_gtc_tok=${_gtc_tok#Comment=twin:}
if [[ -n "$_gtc_tok" ]]; then
  _gtc_me="${SLURM_CLUSTER_NAME:-unknown} ${SLURM_JOB_ID}"
  if mkdir "$_gtc_claims/$_gtc_tok.lock" 2>/dev/null; then
    echo "$_gtc_me" > "$_gtc_claims/$_gtc_tok.lock/owner"
    echo "==> twin claim: $_gtc_me runs this work (token $_gtc_tok)"
  else
    for _gtc_i in 1 2 3 4 5 6 7 8 9 10; do
      [[ -s "$_gtc_claims/$_gtc_tok.lock/owner" ]] && break; sleep 1
    done
    _gtc_owner=$(cat "$_gtc_claims/$_gtc_tok.lock/owner" 2>/dev/null)
    if [[ "$_gtc_owner" != "$_gtc_me" ]]; then
      echo "==> twin claim: already running as '$_gtc_owner'; this job ($_gtc_me) exits without running"
      return 1
    fi
    echo "==> twin claim: requeued job, still holds the claim"
  fi
  # gpu_submit.sh writes the pair file right after its second sbatch; wait for it
  # in case this job started within seconds of submission.
  for _gtc_i in $(seq 1 24); do
    [[ -s "$_gtc_claims/$_gtc_tok.jobs" ]] && break; sleep 5
  done
  while read -r _gtc_cl _gtc_id; do
    [[ -z "$_gtc_id" || "$_gtc_cl $_gtc_id" == "$_gtc_me" ]] && continue
    # SLURM_CONF set explicitly: the module alone does not switch it, and inside a job
    # the node's own /etc/slurm/slurm.conf wins.
    if ( module unload slurm >/dev/null 2>&1; module load "slurm/$_gtc_cl" >/dev/null 2>&1
         export SLURM_CONF="/curc/slurm/$_gtc_cl/etc/slurm.conf"
         scancel "$_gtc_id" ); then
      echo "==> twin claim: cancelled twin $_gtc_cl $_gtc_id"
    else
      echo "==> twin claim: could not cancel $_gtc_cl $_gtc_id; it will exit when it starts"
    fi
  done < "$_gtc_claims/$_gtc_tok.jobs"
fi
return 0
