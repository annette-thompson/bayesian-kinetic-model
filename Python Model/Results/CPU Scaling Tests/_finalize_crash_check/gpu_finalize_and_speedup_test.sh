#!/bin/bash
#SBATCH --job-name=gpu_diag
#SBATCH --partition=aa100
#SBATCH --qos=gpu-normal
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --gres=gpu:a100-40gb:1
#SBATCH --time=01:00:00
#SBATCH --output=gpu_diag.%j.out
#SBATCH --account=ucb634_asc2

set -x  # echo every command, so the .out file shows exactly what ran and when

module load anaconda
conda activate Bayesian
cd /projects/anth4580/Bayesian

echo "=== confirming GPU is visible to JAX ==="
python -c "import jax; print(jax.devices())"

echo "=== ensuring the finalize-crash-check config exists ==="
python - <<'PY'
import json, pathlib
src = json.load(open("Results/Test FabD - a1/solver_params.json"))
src["prior_sampling"]["draws"] = 200
src["posterior_sampling"] = {"draws": 150, "tune": 150, "chains": 4,
                             "target_accept": 0.9, "checkpoint_every_steps": 300, "random_seed": 0}
src["output_paths"]["results_save_dir"] = "Results/_finalize_crash_check"
d = pathlib.Path("Results/_finalize_crash_check"); d.mkdir(parents=True, exist_ok=True)
json.dump(src, open(d/"solver_params.json", "w"), indent=4)
print("wrote", d/"solver_params.json")
PY

echo "=== crash-frequency check (4 attempts) ==="
CRASH_COUNT=0
for i in 1 2 3 4; do
  rm -rf "Results/_finalize_crash_check/checkpoint" "Results/_finalize_crash_check"/*.nc
  echo "--- GPU attempt $i ---"
  python -u Utilities/inference_runner.py \
    --solver_params_file "Results/_finalize_crash_check/solver_params.json" > "gpu_attempt_$i.log" 2>&1
  if grep -qi "recursive_mutex\|system_error\|abort\|Traceback" "gpu_attempt_$i.log"; then
    echo ">>> attempt $i CRASHED"
    CRASH_COUNT=$((CRASH_COUNT+1))
  else
    echo ">>> attempt $i OK"
  fi
done
echo "=== crash tally: $CRASH_COUNT / 4 ==="

echo "=== speedup probe (bigger network, pure compute) ==="
python -u Utilities/benchmark_throughput.py \
  --solver_params_file "Results/Test FabD FabH FabG - a1/solver_params.json" --grad-only --grad-evals 25
cp "Results/Test FabD FabH FabG - a1/gradient_probe.json" gpu_gradient_probe.json

echo "=== DONE ==="
