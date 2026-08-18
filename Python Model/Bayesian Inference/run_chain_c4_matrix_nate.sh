#!/bin/bash
# The C4 matrix on nate: every axis that was previously confounded by dead species.
#
# C4 is the smallest chain-length rung -- 44 reactions, 59 species, and ZERO dead
# species (versus 78-81% dead on the old enzyme ladder). That makes it the right
# place to re-test tolerance, precision, device and floor: on the enzyme systems
# those axes were entangled with deadness, so a floor that "helped" the big system
# and "broke" the small ones was really just reporting how much inert network each
# one carried.
#
# Known going in: on a Mac CPU at rtol=1e-4/atol=1e-8 the C4 log-posterior is finite
# on all 4 chains (183.3, 201.8, 175.9, 181.0) but the gradient is NaN on all 4.
# So a NaN gradient is NOT explained by dead species. The tolerance ladder below is
# there to find where it goes finite.
#
# Usage: run_chain_c4_matrix_nate.sh [extra benchmark_matrix.py args...]
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 1

exec bash "Bayesian Inference/run_benchmark_matrix_nate.sh" \
  --configs "Chain C4 - a2" \
  --tolerances "1e-4:1e-8,1e-5:1e-8,1e-6:1e-8,1e-4:1e-10,1e-6:1e-10" \
  --precisions "32,64" \
  --floors "0,0.001" \
  --timeout 1800 \
  --max-eval-seconds 600 \
  "$@"
