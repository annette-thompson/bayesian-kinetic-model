"""Why does the strict-tolerance baseline probe fail at 720 s? Where does it stall, and
does the floor or a looser reference tolerance change that?"""
import json, sys, time
sys.path.insert(0, "/projects/anth4580/Bayesian/Utilities")
sys.path.insert(0, "/projects/anth4580/Bayesian/job_files/bench")
import numpy as np, jax.numpy as jnp, diffrax as dfrx
import generate_chain_data as gcd
import reaction_model_builder as rmb
from joint_feasibility import _unclamped_call
ROOT = gcd.project_root()
system = sys.argv[1]
cfg = json.loads((ROOT / "Results" / "Chain Scaling Tests" / f"Chain {system} - a1_0.1-10_no_floor" / "solver_params.json").read_text())
sg = {k: float(v) for k, v in cfg["scaling_groups"].items()}
srcs = [ROOT / p for p in cfg["output_paths"]["reactions_source"]]
c = cfg["ODE_stepsize_controller"]
floored_call = rmb.ReactionNetwork.__call__
checkpoints = np.array([150.0, 300.0, 450.0, 600.0, 720.0])

def probe(floor, rtol, atol, cap=200_000):
    rmb.ReactionNetwork.__call__ = floored_call if floor else _unclamped_call
    s = gcd.ChainSystem(srcs, rtol=rtol, atol=atol, pcoeff=c["pcoeff"], icoeff=c["icoeff"], dcoeff=c["dcoeff"], scaling_group_overrides=sg)
    t0 = time.time()
    sol = dfrx.diffeqsolve(dfrx.ODETerm(s.network), dfrx.Kvaerno5(), t0=0.0, t1=720.0, dt0=1e-6,
                           y0=jnp.asarray(s.y0()), args=s.theta, saveat=dfrx.SaveAt(ts=jnp.asarray(checkpoints)),
                           stepsize_controller=dfrx.PIDController(rtol=rtol, atol=atol, pcoeff=c["pcoeff"], icoeff=c["icoeff"], dcoeff=c["dcoeff"]),
                           max_steps=cap, throw=False)
    ys = np.asarray(sol.ys)
    reached = [float(t) for t, y in zip(checkpoints, ys) if np.all(np.isfinite(y))]
    steps = int(sol.stats["num_steps"]); rej = int(sol.stats["num_rejected_steps"])
    ok = bool(sol.result == dfrx.RESULTS.successful)
    last = ys[len(reached) - 1] if reached else None
    neg = None if last is None else float(last.min())
    print(f"  floor={floor!s:<5} rtol={rtol:g} atol={atol:g}: ok={ok} steps={steps} rejected={rej} "
          f"reached={reached[-1] if reached else 0:g}s  min conc at last reached={neg}  ({time.time()-t0:.0f}s)", flush=True)
    return s, sol, reached

print(f"=== {system}")
s, _, _ = probe(False, c["rtol"], c["atol"])           # working tolerance, no floor
for floor in (False, True):
    for rtol, atol in ((1e-10, 1e-12), (1e-9, 1e-11), (1e-8, 1e-10)):
        probe(floor, rtol, atol)
# which species are near zero at 150 s and 720 s (working tolerance, no floor)
rmb.ReactionNetwork.__call__ = _unclamped_call
sol = dfrx.diffeqsolve(dfrx.ODETerm(s.network), dfrx.Kvaerno5(), t0=0.0, t1=720.0, dt0=1e-6, y0=jnp.asarray(s.y0()), args=s.theta,
                       saveat=dfrx.SaveAt(ts=jnp.asarray([150.0, 720.0])),
                       stepsize_controller=dfrx.PIDController(rtol=c["rtol"], atol=c["atol"], pcoeff=c["pcoeff"], icoeff=c["icoeff"], dcoeff=c["dcoeff"]),
                       max_steps=20000, throw=False)
ys = np.asarray(sol.ys)
for name in ("C3_MalCoA", "C2_AcCoA", "NADPH", "NADH", "ACP"):
    i = s.index_of[name]; print(f"  {name:<10} 150 s: {ys[0, i]:.4g}   720 s: {ys[1, i]:.4g}")
fa = [n for n in s.species if n.endswith("_FA")]
print("  total FA (uM) 150 s: %.4g   720 s: %.4g" % (sum(ys[0, s.index_of[n]] for n in fa), sum(ys[1, s.index_of[n]] for n in fa)))
print("DONE")
