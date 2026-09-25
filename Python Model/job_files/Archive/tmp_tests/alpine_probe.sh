#!/bin/bash -l
#SBATCH --output=/projects/anth4580/Bayesian/tmp_tests/alpine_probe_%j.out
#SBATCH --export=NONE
echo "== inside job on $(hostname); SLURM_CLUSTER_NAME=$SLURM_CLUSTER_NAME SLURM_JOB_ID=$SLURM_JOB_ID"
echo "SLURM_CONF=$SLURM_CONF"
printf '#!/bin/bash\ntrue\n' > /tmp/anth_probe_payload.sh
echo "--- A: module unload/load then sbatch --test-only aa100"
( module unload slurm >/dev/null 2>&1; module load slurm/alpine >/dev/null 2>&1; echo "   SLURM_CONF now=$SLURM_CONF; sbatch=$(command -v sbatch)"
  sbatch --test-only --partition=aa100 --qos=gpu-normal --account=ucb634_asc2 --gres=gpu:a100-40gb:1 --time=00:10:00 /tmp/anth_probe_payload.sh 2>&1 | head -3 )
echo "--- B: same, but with the job's own SLURM_* variables cleared"
( module unload slurm >/dev/null 2>&1; module load slurm/alpine >/dev/null 2>&1
  env -u SLURM_JOB_ID -u SLURM_JOBID -u SLURM_CLUSTER_NAME -u SLURM_NODELIST -u SLURM_JOB_PARTITION -u SLURM_JOB_QOS -u SLURM_JOB_ACCOUNT -u SLURM_SUBMIT_DIR \
    sbatch --test-only --partition=aa100 --qos=gpu-normal --account=ucb634_asc2 --gres=gpu:a100-40gb:1 --time=00:10:00 /tmp/anth_probe_payload.sh 2>&1 | head -3 )
echo "--- C: blanca control (should work)"
( module unload slurm >/dev/null 2>&1; module load slurm/blanca >/dev/null 2>&1
  sbatch --test-only --partition=blanca --qos=preemptable --time=00:10:00 /tmp/anth_probe_payload.sh 2>&1 | head -2 )
