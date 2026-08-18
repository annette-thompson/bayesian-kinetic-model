#!/bin/bash
# Provisions the "Bayesian" conda env on THIS machine. Meant to run on nate,
# streamed via setup_nate_env.sh (`ssh nate bash -s -- [--recreate] <
# provision_nate_env.sh`) rather than copied over by hand.
#
# Installs Miniforge (conda + mamba) if no conda/mamba is found, creates (or
# rebuilds, with --recreate) the "Bayesian" env from the package list in
# environment.yml (the same env name Alpine's job scripts activate), then
# installs GPU-enabled JAX via `pip install jax[cuda12]` -- JAX's cuda12 pip
# wheels bundle their own CUDA runtime, so no system CUDA toolkit or
# driver-matched conda build is needed, just a recent enough NVIDIA driver
# (this mirrors how Alpine's GPU jobs get GPU jax without `module load
# cuda` -- see the "cuda/gcc unneeded" note for that cluster).
set -euo pipefail

RECREATE=0
[[ "${1:-}" == "--recreate" ]] && RECREATE=1

JAX_VERSION="0.7.0"
# Everything Utilities/ and Calculation Files/ import directly is listed here,
# even where another package would drag it in anyway -- sympy and seaborn were
# missing for exactly that reason (nothing else pulled them), and the runner only
# found out by dying on `import sympy` a few seconds into a GPU job.
CONDA_PACKAGES=(
  python=3.12 numba=0.65.1 llvmlite pytensor arviz blackjax diffrax
  numpyro nutpie preliz "zarr>=3" pymc equinox jaxtyping ipykernel
  "libsqlite>=3.53" "sqlite>=3.53"
  sympy seaborn matplotlib pandas pyyaml
)

echo "=================================================="
echo "==> Provisioning nate for Bayesian inference"
echo "==> Host: $(hostname)   User: $(whoami)   Date: $(date)"
echo "=================================================="

echo ""
echo "--- GPU check ---"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
else
  echo "WARNING: nvidia-smi not found -- is the NVIDIA driver installed?" >&2
fi

echo ""
echo "--- conda/mamba ---"
for base in "$HOME/miniforge3" "$HOME/miniconda3" "$HOME/anaconda3" "$HOME/mambaforge" "/opt/conda" "/opt/miniconda3" "/opt/miniforge3"; do
  if [[ -x "$base/bin/conda" ]]; then
    export PATH="$base/bin:$PATH"
    break
  fi
done

if ! command -v conda >/dev/null 2>&1; then
  echo "No conda found -- installing Miniforge3 to \$HOME/miniforge3"
  INSTALLER="$HOME/miniforge_installer.sh"
  curl -fsSL -o "$INSTALLER" "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh"
  bash "$INSTALLER" -b -p "$HOME/miniforge3"
  rm -f "$INSTALLER"
  export PATH="$HOME/miniforge3/bin:$PATH"
  conda init bash || true
else
  echo "Found conda: $(command -v conda)"
fi

source "$(conda info --base)/etc/profile.d/conda.sh"

SOLVER=conda
command -v mamba >/dev/null 2>&1 && SOLVER=mamba
echo "Using $SOLVER for env creation"

ENV_EXISTS=0
conda env list | awk '{print $1}' | grep -qx "Bayesian" && ENV_EXISTS=1

if [[ "$ENV_EXISTS" -eq 1 && "$RECREATE" -eq 1 ]]; then
  echo "--recreate: removing existing Bayesian env"
  conda env remove -n Bayesian -y
  ENV_EXISTS=0
fi

if [[ "$ENV_EXISTS" -eq 0 ]]; then
  echo ""
  echo "--- Creating 'Bayesian' env (python=3.12 + conda-forge packages) ---"
  "$SOLVER" create -n Bayesian -c conda-forge -y "${CONDA_PACKAGES[@]}"
else
  echo "'Bayesian' env already exists -- reusing (pass --recreate to rebuild from scratch)"
fi

conda activate Bayesian

echo ""
echo "--- GPU-enabled JAX ---"
# conda-forge's pymc/numpyro/blackjax now pull in a CUDA-capable jax themselves
# (jax-cuda12-pjrt and friends), so the env is often GPU-ready already. Forcing
# the pinned version in with pip then FAILS the whole provision: pip cannot
# uninstall a conda-installed jax-cuda12-pjrt (no RECORD file), so it aborts. Only
# reach for pip when conda's jax genuinely can't see the GPU.
if python -c "import jax, sys; sys.exit(0 if any(d.platform == 'gpu' for d in jax.devices()) else 1)" 2>/dev/null; then
  echo "conda-forge jax already sees the GPU -- skipping the pip install"
else
  echo "conda jax sees no GPU -- installing jax[cuda12]==${JAX_VERSION} (self-contained CUDA runtime)"
  pip install --upgrade "jax[cuda12]==${JAX_VERSION}" "jaxlib==${JAX_VERSION}"
fi

echo ""
echo "--- Verifying ---"
python - <<'PY'
import importlib.metadata as md
import platform

packages = ["arviz", "blackjax", "diffrax", "equinox", "jax", "jaxlib", "numpy", "pymc", "pytensor", "zarr"]
print(f"Python version: {platform.python_version()}")
for package in packages:
    try:
        print(f"  {package}: {md.version(package)}")
    except md.PackageNotFoundError:
        print(f"  {package}: NOT INSTALLED")

import jax
print(f"JAX devices: {jax.devices()}")
if not any(d.platform == "gpu" for d in jax.devices()):
    print(
        "WARNING: JAX does not see a GPU device -- check the NVIDIA driver version "
        "(cuda12 wheels need a fairly recent driver) and that CUDA_VISIBLE_DEVICES "
        "isn't restricting it."
    )
PY

echo ""
echo "==> Done. On nate: conda activate Bayesian"
