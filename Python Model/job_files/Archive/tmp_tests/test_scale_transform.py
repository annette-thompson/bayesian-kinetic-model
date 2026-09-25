"""Verify inference_runner._ScaleTransform on the real C6 a1+c3+d1 model.

The reparameterization must change ONLY the coordinate NUTS walks in, never the
model. So, for the same physical point:
  1. log-density differs by exactly log|d(d1)/du| = -log(12)
  2. d logp / du  ==  (d logp / d d1) / 12        (chain rule)
  3. d1 comes back out as u / 12
and the reason for doing it at all:
  4. initial-point jitter lands d1 inside its prior window, not at TesA 25,000x
"""
import copy, json, math, sys
from pathlib import Path

sys.path.insert(0, "/projects/anth4580/Bayesian/Utilities")
import numpy as np
import inference_runner as ir
import resumable_sampler as rs

RUN = Path("/projects/anth4580/Bayesian/Results/Chain Scaling Tests/Chain C6 - a1c3d1_no_floor")
ORIG = RUN / "solver_params.json"
TEST = RUN / "solver_params_SCALETEST.json"   # same dir => relative paths resolve identically

results = []
def check(label, ok, detail=""):
    results.append(ok)
    print(("PASS  " if ok else "FAIL  ") + label + (f"   [{detail}]" if detail else ""), flush=True)

cfg = json.loads(ORIG.read_text())
scaled = copy.deepcopy(cfg)
for spec in scaled["free_kinetic_params"]:
    if spec["param_name"] == "d1":
        spec["prior_dist_params"]["sample_scale"] = 12.0
TEST.write_text(json.dumps(scaled, indent=2))

try:
    old = ir._build_model_bundle(ir.import_solver_params(ORIG)).pm_model
    new = ir._build_model_bundle(ir.import_solver_params(TEST)).pm_model
    old_vv = [v.name for v in old.value_vars]
    new_vv = [v.name for v in new.value_vars]
    print("old value vars:", old_vv)
    print("new value vars:", new_vv)
    check("free RV still named d1 (posteriors/plots unchanged)",
          [r.name for r in new.free_RVs] == ["a1", "c3", "d1"])
    check("NUTS coordinate is d1_x12__", new_vv == ["a1_log__", "c3_log__", "d1_x12__"])

    # Same physical point in both coordinates, inside the prior.
    a1l, c3l, d1 = 0.10, -0.20, 0.05
    p_old = {"a1_log__": a1l, "c3_log__": c3l, "d1": d1}
    p_new = {"a1_log__": a1l, "c3_log__": c3l, "d1_x12__": 12.0 * d1}

    lp_old = float(old.compile_logp()(p_old))
    lp_new = float(new.compile_logp()(p_new))
    diff = lp_new - lp_old
    check("logp_new - logp_old == -log(12)  (same model, only the Jacobian)",
          abs(diff - (-math.log(12.0))) < 1e-6,
          f"diff={diff:.8f}, expected {-math.log(12.0):.8f}; logp_old={lp_old:.4f}")

    g_old = np.asarray(old.compile_dlogp()(p_old))
    g_new = np.asarray(new.compile_dlogp()(p_new))
    check("d logp/d(a1_log__), d(c3_log__) unchanged",
          np.allclose(g_old[:2], g_new[:2], rtol=1e-6, atol=1e-8),
          f"old={g_old[:2]}, new={g_new[:2]}")
    ratio = g_new[2] / g_old[2] if g_old[2] != 0 else float("nan")
    check("d logp/du == (d logp/d d1) / 12", abs(ratio - 1.0 / 12.0) < 1e-6,
          f"ratio={ratio:.8f}, expected {1/12:.8f}")

    # The actual pipeline path: jittered initial points through prepare_from_pymc.
    br_old = rs.prepare_from_pymc(old, n_chains=8, random_seed=42)
    br_new = rs.prepare_from_pymc(new, n_chains=8, random_seed=42)
    d1_old = np.asarray(br_old.initial_positions[br_old.value_var_names.index("d1")]).ravel()
    u_new = np.asarray(br_new.initial_positions[br_new.value_var_names.index("d1_x12__")]).ravel()
    d1_new = u_new / 12.0
    lim = math.log(10) / 12.0
    fmt = lambda a: " ".join(f"{v:+.3f}" for v in a)
    print("initial d1, OLD:", fmt(d1_old), f" TesA x {math.exp(-12*d1_old.max()):.3g}..{math.exp(-12*d1_old.min()):.3g}")
    print("initial d1, NEW:", fmt(d1_new), f" TesA x {math.exp(-12*d1_new.max()):.3g}..{math.exp(-12*d1_new.min()):.3g}")
    check(f"new initial points all inside the prior window |d1| < {lim:.4f}",
          bool(np.all(np.abs(d1_new) < lim)), f"max |d1| = {np.abs(d1_new).max():.4f}")
finally:
    TEST.unlink(missing_ok=True)

print("\nALL PASS" if all(results) else "\nSOME FAILED", flush=True)
sys.exit(0 if all(results) else 1)
