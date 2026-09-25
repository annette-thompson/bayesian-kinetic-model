"""Refit the five originally-fitted parameters against dataset D instead of dataset A.

obj1's target is swapped from the seven fitted initial rates (A) to the six held-out ones
(D); obj2 (time course) and obj3 (profile) are unchanged. The question is how far the
parameters move when asked to explain the data they failed on -- which separates "the
held-out gap is a parameter problem" from "it is structural".

Usage: python -u refit_to_D.py [--evals 250]
"""
import argparse, json, sys, time
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, ".")
from scipy.optimize import minimize
import me1_config as cfg, me1_model as mm, me1_objective as mo

ap = argparse.ArgumentParser()
ap.add_argument("--evals", type=int, default=250)
ap.add_argument("--reactions", default="C20+unsat",
                help="reaction set to fit with (default C20+unsat, i.e. no FBinit)")
a = ap.parse_args()

rx = cfg.PROJECT / "Reactions" / "EC_FAS_ME1" / a.reactions
print(f"reaction set: {a.reactions}", flush=True)
M = mm.ME1Model(reactions_dir=rx); data = mo.load_data(); published = dict(cfg.PUBLISHED)
held = pd.read_csv(cfg.DATA / "heldout_initial_rates.csv")
Dkw = dict(rate_conditions=cfg.HELDOUT_CONDITIONS,
           rate_measured=held["measured_rate_uM_C16_per_min"].to_numpy())

base_A = mo.evaluate(M, published, data=data)
base_D = mo.evaluate(M, published, data=data, **Dkw)
print(f"published on A: obj1={base_A.obj1:8.4f} total={base_A.total:12.4g}")
print(f"published on D: obj1={base_D.obj1:8.4f} total={base_D.total:12.4g}", flush=True)

f = mo.make_scalar_objective(M, cfg.MATLAB_FITTED, published, data=data, **Dkw)
t0 = time.time()
res = minimize(f, f.pack(published), method="Nelder-Mead",
               options={"maxfev": a.evals, "maxiter": a.evals, "disp": True})
fitted = f.unpack(res.x)
print(f"\n{time.time()-t0:.0f} s, {len(f.history)} evaluations")

after_D = mo.evaluate(M, fitted, data=data, **Dkw)
after_A = mo.evaluate(M, fitted, data=data)
rows = [{"parameter": n, "published": published[n], "refit_to_D": fitted[n],
         "ratio": fitted[n] / published[n]} for n in cfg.MATLAB_FITTED]
print("\n" + pd.DataFrame(rows).to_string(index=False, float_format=lambda x: f"{x:.5g}"))
print(f"\n{'':<14}{'obj1':>10}{'obj2':>10}{'obj3':>10}{'total':>14}")
for lab, r in (("published/D", base_D), ("refit/D", after_D),
               ("published/A", base_A), ("refit/A", after_A)):
    print(f"{lab:<14}{r.obj1:>10.4f}{r.obj2:>10.3f}{r.obj3:>10.3f}{r.total:>14.4g}")
json.dump({"reactions": a.reactions, "published": published, "refit_to_D": fitted,
           "obj": {"published_D": base_D.total, "refit_D": after_D.total,
                   "published_A": base_A.total, "refit_A": after_A.total}},
          open(f"refit_to_D_{a.reactions.replace('+','_')}.json", "w"), indent=1)
pd.DataFrame(f.history).to_csv(f"refit_to_D_{a.reactions.replace('+','_')}_history.csv", index=False)
print(f"\nwrote refit_to_D_{a.reactions.replace('+','_')}.json and its history csv")

# ---- all four Figure S1 datasets, published vs refit-to-D -------------------------
rates, tc, prof = data
t_data = tc["time_min"].to_numpy() * 60.0
times = np.unique(np.concatenate([t_data, [cfg.ENDPOINT_S]]))


def snapshot(p):
    sol = M.solve(p, M.condition_y0(cfg.RATE_CONDITIONS[0]), times)
    at = {float(t): i for i, t in enumerate(times)}
    return {"A": M.initial_rates_c16(p) / 2.5,
            "D": M.initial_rates_c16(p, cfg.HELDOUT_CONDITIONS) / 2.5,
            "C": np.array([sol.c16_equivalents[at[float(t)]] for t in t_data]),
            "B": sol.profile[at[float(cfg.ENDPOINT_S)]]}


snap = {"published": snapshot(published), "refit to D": snapshot(fitted)}
COL = {"published": "#2f6f9f", "refit to D": "#3d7d54"}
fig, axes = plt.subplots(2, 2, figsize=(15, 9.5))
w = 0.8 / 3


def bars(ax, names, meas, se, key, title):
    x = np.arange(len(names))
    ax.bar(x - w, meas, w, yerr=se, capsize=3, color="#c4643a", label="measured")
    for i, (lab, sn) in enumerate(snap.items()):
        ax.bar(x + i * w, sn[key], w, color=COL[lab], label=lab)
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=35, ha="right", fontsize=7)
    ax.set_ylabel("initial rate (uM C16 eq/min)"); ax.set_title(title, fontsize=10)
    ax.grid(axis="y", alpha=0.3); ax.legend(fontsize=8)


bars(axes[0][0], [c["label"] for c in cfg.RATE_CONDITIONS],
     rates["measured_rate_uM_C16_per_min"].to_numpy(), rates["standard_error"].to_numpy(),
     "A", "A. Initial rates (was fitted)")
bars(axes[1][1], [c["label"] for c in cfg.HELDOUT_CONDITIONS],
     held["measured_rate_uM_C16_per_min"].to_numpy(), held["standard_error"].to_numpy(),
     "D", "D. Held-out rates (now the fit target)")

ax = axes[1][0]
ax.plot(tc["time_min"], tc["c16_equivalents_uM"], "o", color="#c4643a", ms=8, label="measured")
for lab, sn in snap.items():
    ax.plot(tc["time_min"], sn["C"], "s-", color=COL[lab], label=lab)
ax.set_xlabel("time (min)"); ax.set_ylabel("C16 equivalents (uM)")
ax.set_title("C. Reference time course", fontsize=10); ax.grid(alpha=0.3); ax.legend(fontsize=8)

ax = axes[0][1]
labels = [f"C{r.chain}{':1' if r.unsaturated else ''}" for r in prof.itertuples()]
x = np.arange(len(labels))
ax.bar(x - w, snap["published"]["B"].sum() * prof["mole_fraction"].to_numpy(), w,
       color="#c4643a", label="target shape")
for i, (lab, sn) in enumerate(snap.items()):
    ax.bar(x + i * w, sn["B"], w, color=COL[lab], label=lab)
ax.set_xticks(x); ax.set_xticklabels(labels, rotation=90, fontsize=7)
ax.set_ylabel("fatty acid (uM)"); ax.set_title("B. Product profile at 720 s", fontsize=10)
ax.grid(axis="y", alpha=0.3); ax.legend(fontsize=8)

fig.suptitle(f"{a.reactions}: published vs refit to dataset D, against all four datasets")
fig.tight_layout(rect=(0, 0, 1, 0.97))
out = f"refit_to_D_{a.reactions.replace('+','_')}.png"
fig.savefig(out, dpi=160); print(f"wrote {out}")
