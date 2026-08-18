#!/bin/bash
# Bootstrap nate for Bayesian inference: installs Miniforge (if missing) and
# creates the "Bayesian" conda env with GPU-enabled JAX. Run this from your
# Mac, once, before the first sync/submit to nate. It streams
# Sync/provision_nate_env.sh to nate over ssh and runs it there
# (`ssh nate bash -s -- ... < provision_nate_env.sh`) -- nothing needs to be
# copied over by hand first.
#
# Usage: Sync/setup_nate_env.sh [--recreate]
#   --recreate   drop and rebuild the "Bayesian" env from scratch
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/nate_sync_config.sh"   # NATE_HOST
PROVISION_SCRIPT="$SCRIPT_DIR/provision_nate_env.sh"
SSH_OPTS=(-o ConnectTimeout=10 -o BatchMode=yes)

[[ -r "$PROVISION_SCRIPT" ]] || { echo "Error: $PROVISION_SCRIPT not found" >&2; exit 1; }

echo "==> Provisioning $NATE_HOST (installing Miniforge/packages can take several minutes on first run)..."
echo ""
ssh "${SSH_OPTS[@]}" "$NATE_HOST" bash -s -- "$@" < "$PROVISION_SCRIPT"

echo ""
echo "==> Provisioning complete. Next steps:"
echo "    Sync/sync_to_nate.sh                                  # push the repo"
echo "    Bayesian\\ Inference/submit_inference_nate.sh --solver-params FILE   # launch a run"
