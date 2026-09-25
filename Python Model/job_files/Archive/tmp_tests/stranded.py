"""What were the stranded chains doing? Per-chain position, log posterior, and sampler
behaviour for every run the lp-gap check flagged."""
import json, os
import numpy as np, zarr

BASE = "/projects/anth4580/Bayesian/Results/Chain Scaling Tests"
RUNS = ["Chain C8 - a1_0.1-10_no_floor", "Chain C10 - a1_0.05-20_no_floor",
        "Chain C6 - a1 narrowest nofloor-eqxnan"]
for run in RUNS:
    d = os.path.join(BASE, run)
    st = json.load(open(os.path.join(d, "checkpoint", "status.json")))
    cfg = json.load(open(os.path.join(d, "solver_params.json")))
    z = zarr.open(os.path.join(d, "checkpoint", "draws.zarr"), mode="r")
    stranded = set(st.get("stranded_chains", []))
    prior = cfg["free_kinetic_params"][0]["prior_dist_params"]
    print("\n=== %s   prior [%g, %g], target_accept %s, tune %s, draws %s" % (
        run, prior["lower"], prior["upper"], cfg["posterior_sampling"]["target_accept"],
        st["n_tune"], st["sampling_done"]))
    names = list(z["sampling"].array_keys())
    g = z["sampling_stats"]
    lp = np.asarray(g["lp"][:]); acc = np.asarray(g["acceptance_rate"][:])
    ss = np.asarray(g["step_size"][:]); div = np.asarray(g["diverging"][:])
    ns = np.asarray(g["n_steps"][:])
    vals = {n: np.asarray(z["sampling/" + n][:]) for n in names}
    for n, v in vals.items():
        if v.ndim == 3 and v.shape[2] == 1:
            vals[n] = v[:, :, 0]
    tail = slice(-min(200, lp.shape[1]), None)
    print("  %-6s %-9s %11s %11s %9s %9s %8s   %s" % (
        "chain", "state", "lp mean", "lp last200", "accept", "step", "diverg%",
        ", ".join("%s (exp -> value)" % n for n in names)))
    lp_tail = lp[:, tail].mean(axis=1)
    best = lp_tail.max()
    for c in range(lp.shape[0]):
        pos = ", ".join("%.4f -> %.4f" % (vals[n][c, tail].mean(), np.exp(vals[n][c, tail].mean()))
                        for n in names)
        print("  %-6d %-9s %11.2f %11.2f %9.3f %9.4f %8.2f   %s" % (
            c, "STRANDED" if c in stranded else "kept", lp[c].mean(), lp_tail[c],
            acc[c].mean(), np.median(ss[c]), 100 * div[c].mean(), pos))
    print("  lp gap worst kept vs stranded: %.1f nats (exclusion threshold 20)" % (
        best - min(lp_tail) if stranded else float("nan")))
    # when did they separate?
    if stranded:
        s = sorted(stranded)[0]
        keep = [c for c in range(lp.shape[0]) if c not in stranded]
        gap = lp[keep].mean(axis=0) - lp[s]
        first = np.argmax(gap > 20) if (gap > 20).any() else None
        print("  chain %d first fell >20 nats behind at sampling draw %s; median leapfrog steps: stranded %.0f vs kept %.0f" % (
            s, first if first is not None else "never", np.median(ns[s]), np.median(ns[keep])))
