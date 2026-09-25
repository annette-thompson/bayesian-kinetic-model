#!/bin/bash -l
#SBATCH --partition=blanca
#SBATCH --qos=preemptable
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:a100:1
#SBATCH --time=00:05:00
#SBATCH --output=/projects/anth4580/Bayesian/job_files/c20unsat_solver_compare/logs/gputest.%j.out
#SBATCH --export=NONE
module purge
module load miniforge
mamba activate Bayesian
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
python -c "
import jax
print('devices:', jax.devices())
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
x = jnp.array([1.0], dtype=jnp.float64)
print('dtype:', x.dtype, 'device:', x.devices())
"
