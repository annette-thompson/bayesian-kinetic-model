"""Isolate why the sensitivity sweep reports identical d1 and d2 curves at C14.

The model is structurally capable of separating them: C14's TesA carries
1/exp(12*d1+d2) for chains 4-12 and 1/exp((2*14-12)*d1+d2) = coefficient 16 for
chain 14, and C14_FA is the largest observable in the dataset. So a d1-only and a
d2-only perturbation chosen to induce the SAME multiplier on the coefficient-12
reactions must induce DIFFERENT multipliers (m^1.333 vs m) on the coefficient-16
one, and the observables must differ.

This checks the one link the structural inspection could not: whether
set_scaling_group_values actually writes d1 and d2 as separate entries in theta,
and whether the built parameter list contains them at all. A silent no-op there
would explain identical curves while leaving the model correct.
"""
import json, sys
from pathlib import Path
sys.path.insert(0, "/projects/anth4580/Bayesian/Utilities")
import numpy as np
import generate_chain_data as gcd
from reaction_model_builder import set_scaling_group_values

ROOT = gcd.project_root()
SYSTEM = "C14"
cfg = json.loads((ROOT / "Results/Chain Scaling Tests" / f"Chain {SYSTEM} - a1 tightest"
                  / "solver_params.json").read_text())
sg = {k: float(v) for k, v in cfg["scaling_groups"].items()}
srcs = [ROOT / p for p in cfg["output_paths"]["reactions_source"]]
ctrl = cfg["ODE_stepsize_controller"]
sys_ = gcd.ChainSystem(srcs, rtol=ctrl["rtol"], atol=ctrl["atol"], pcoeff=ctrl["pcoeff"],
                       icoeff=ctrl["icoeff"], dcoeff=ctrl["dcoeff"],
                       scaling_group_overrides=sg)

params = list(sys_.params)
print(f"n params = {len(params)}")
for g in ("d1", "d2", "a1"):
    print(f"  '{g}' in params: {g in params}"
          f"{'  index ' + str(params.index(g)) if g in params else ''}")

m = 10.0
o1 = dict(sg); o1["d1"] = -np.log(m) / 12.0
o2 = dict(sg); o2["d2"] = -np.log(m)
t0 = np.asarray(sys_.theta)
t1 = np.asarray(set_scaling_group_values(sys_.theta, sys_.params, o1))
t2 = np.asarray(set_scaling_group_values(sys_.theta, sys_.params, o2))

print(f"\nintended: d1 = {o1['d1']:.6f} (d2=0)   vs   d2 = {o2['d2']:.6f} (d1=0)")
print(f"theta differs nominal vs d1-override: {not np.allclose(t0, t1)}"
      f"  ({int((t0 != t1).sum())} entries changed)")
print(f"theta differs nominal vs d2-override: {not np.allclose(t0, t2)}"
      f"  ({int((t0 != t2).sum())} entries changed)")
print(f"theta differs d1-override vs d2-override: {not np.allclose(t1, t2)}"
      f"  ({int((t1 != t2).sum())} entries changed)")
for g in ("d1", "d2"):
    if g in params:
        i = params.index(g)
        print(f"  theta[{g}]: nominal={t0[i]:.6f}  d1ovr={t1[i]:.6f}  d2ovr={t2[i]:.6f}")

print("\nVERDICT:", "sweep bug is NOT in set_scaling_group_values"
      if not np.allclose(t1, t2) else
      "set_scaling_group_values produced IDENTICAL theta -- this is the bug")
