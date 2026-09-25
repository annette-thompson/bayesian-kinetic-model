#!/bin/bash -l
#SBATCH --output=/projects/anth4580/Bayesian/tmp_tests/alpine_probe2_%j.out
#SBATCH --export=NONE
echo "== on $(hostname), node SLURM_CONF=$SLURM_CONF"
printf '#!/bin/bash\ntrue\n' > /tmp/anth_probe_payload.sh
( module unload slurm >/dev/null 2>&1; module load slurm/alpine >/dev/null 2>&1
  export SLURM_CONF=/curc/slurm/alpine/etc/slurm.conf
  echo "   explicit SLURM_CONF=$SLURM_CONF"
  env -u SLURM_JOB_ID -u SLURM_JOBID -u SLURM_CLUSTER_NAME -u SLURM_NODELIST -u SLURM_JOB_PARTITION -u SLURM_JOB_QOS -u SLURM_JOB_ACCOUNT \
    sbatch --test-only --partition=aa100 --qos=gpu-normal --account=ucb634_asc2 --gres=gpu:a100-40gb:1 --time=00:10:00 /tmp/anth_probe_payload.sh 2>&1 | head -3 )
