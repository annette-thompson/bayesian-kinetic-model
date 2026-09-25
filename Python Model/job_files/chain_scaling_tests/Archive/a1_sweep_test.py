"""1-D sweep over a1 alone, everything else at nominal -- matching the ACTUAL
truncated_warmup_test.py config exactly (free_kinetic_params has only "a1";
every other scaling group is fixed at nominal). The extreme-draws test in
forward_solve_test.py perturbed all 13 scaling groups at once, which is not
what real warmup here actually explores -- NUTS can only ever move a1, since
it's the only free parameter. This finds where along a1's own LogNormal(0.001,
1000) prior range the ODE solve actually breaks, with and without the floor.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path("/projects/anth4580/Bayesian")
UTILITIES_DIR = PROJECT_ROOT / "Utilities"
JOB_DIR = PROJECT_ROOT / "job_files" / "chain_scaling_tests"
sys.path.insert(0, str(UTILITIES_DIR))
sys.path.insert(0, str(JOB_DIR))

import jax
jax.config.update("jax_enable_x64", True)

import forward_solve_test as fst
from generate_chain_data import ChainSystem, nominal_scaling_group_overrides
from reaction_model_builder import discover_scaling_groups, set_scaling_group_values

SYSTEM = "C4_NoFB"
# The prior is LogNormal with lower=0.001/upper=1000 as a 95%-mass CREDIBLE
# INTERVAL (a preliz-style maxent spec), not a hard truncation -- the
# distribution's actual support extends past both ends, and an unadapted early
# NUTS step (in the log-transformed space) can propose values out there. Sweep
# well beyond the stated interval so a failure point in the tails isn't missed.
A1_VALUES = np.geomspace(1e-6, 1e6, 80).tolist()


def main() -> None:
    rx = fst.reactions_for(SYSTEM)
    scaling_groups = discover_scaling_groups(rx)
    sys_ = ChainSystem(rx, fst.RTOL, fst.ATOL, fst.PCOEFF, fst.ICOEFF, fst.DCOEFF,
                      scaling_group_overrides=nominal_scaling_group_overrides(scaling_groups))
    y0 = sys_.y0()
    network_floor = sys_.network
    network_nofloor = fst.unclamp(sys_.network)

    print(f"=== {SYSTEM}: 1-D sweep over a1 alone, {len(A1_VALUES)} points over [0.001, 1000] ===", flush=True)
    print(f"{'a1':>12}  {'floor_result':<20}{'floor_steps':>12}  {'nofloor_result':<20}{'nofloor_steps':>12}", flush=True)

    jsonl_path = JOB_DIR / f"a1_sweep_{SYSTEM}.jsonl"
    with open(jsonl_path, "w") as jf:
        for a1 in A1_VALUES:
            theta_i = set_scaling_group_values(sys_.theta, sys_.params, {"a1": a1})
            r_f = fst.run_once(network_floor, theta_i, y0, f"a1={a1:.4g}/floor")
            r_nf = fst.run_once(network_nofloor, theta_i, y0, f"a1={a1:.4g}/no-floor")
            print(f"{a1:>12.4g}  {r_f['result_name']:<20}{r_f['num_steps']:>12}  "
                 f"{r_nf['result_name']:<20}{r_nf['num_steps']:>12}", flush=True)
            jf.write(json.dumps(dict(
                a1=a1,
                floor_result=r_f["result_name"], floor_steps=r_f["num_steps"],
                floor_min_conc=r_f["min_concentration"],
                nofloor_result=r_nf["result_name"], nofloor_steps=r_nf["num_steps"],
                nofloor_min_conc=r_nf["min_concentration"],
            )) + "\n")
            jf.flush()
    print(f"per-point detail written to {jsonl_path}", flush=True)


if __name__ == "__main__":
    main()
