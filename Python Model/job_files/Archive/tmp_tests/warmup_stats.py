import numpy as np, zarr
from pathlib import Path
BASE = Path("/projects/anth4580/Bayesian/Results/Chain Scaling Tests")
for run in ["Chain C6 - a1c3_no_floor", "Chain C6 - a1c3d1_no_floor", "Archive/Chain C6 - a1c3d1_unscaled_no_floor",
            "Chain C10 - a1c3_no_floor", "Chain C10 - a1c3d1_no_floor"]:
    z = zarr.open(str(BASE / run / "checkpoint" / "draws.zarr"), mode="r")
    g = z["warmup_stats"]
    n = min(25, g["acceptance_rate"].shape[1])
    def s(name):
        return np.asarray(g[name][:, :n]) if name in g else None
    steps, eps, acc, div = s("n_steps"), s("step_size"), s("acceptance_rate"), s("diverging")
    print(f"{run[-40:]:<40} first {n} warmup draws: grad evals/draw median {np.median(steps):.0f} (max {steps.max()}), "
          f"step size median {np.median(eps):.3g}, accept {np.mean(acc):.2f}, divergent {np.mean(div):.1%}")
    per_chain = np.median(steps, axis=1)
    print(f"{'':<40} per-chain median grad evals: {per_chain.astype(int).tolist()}; "
          f"last 5 draws step size by chain: {np.round(np.median(eps[:, -5:], axis=1), 4).tolist()}")
    td = s("tree_depth"); lp = s("lp")
    print(f"{'':<40} tree depth median {np.median(td):.0f} max {td.max()}; lp at draw {n} by chain: {np.round(lp[:, -1], 1).tolist()}")
    vals = z["warmup"]
    for v in vals.array_keys():
        a = np.asarray(vals[v][:, n - 1])
        print(f"{'':<40} {v} at draw {n}: {np.round(a.ravel(), 3).tolist()}")
