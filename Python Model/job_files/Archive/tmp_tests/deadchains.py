"""Every run whose chains died: when, at what step size, and under which configuration.

A chain is "dead" when its acceptance is ~0 for the rest of the run: every proposal is
rejected, so its position cannot move. resumable_sampler aborts a run only when ALL chains
are dead; partial deaths are silent, so this scans for both.
"""
import glob, json, os, re
import numpy as np, zarr

EPS = 1e-6
ROOTS = ["/projects/anth4580/Bayesian/Results/Chain Scaling Tests",
         "/projects/anth4580/Bayesian/Results/Chain Scaling Tests/Archive",
         "/projects/anth4580/Bayesian/Results/Chain Count Test"]

def death_index(acc_row):
    """First index after which acceptance never recovers above EPS, or None."""
    alive = np.where(acc_row > EPS)[0]
    last = alive[-1] if alive.size else -1
    return last + 1 if last + 1 < acc_row.size else None

rows = []
for root in ROOTS:
    for d in sorted(glob.glob(os.path.join(root, "*"))):
        ck = os.path.join(d, "checkpoint")
        zp = os.path.join(ck, "draws.zarr")
        if not os.path.isdir(zp):
            continue
        try:
            z = zarr.open(zp, mode="r")
            cfg = json.load(open(os.path.join(d, "solver_params.json")))["posterior_sampling"]
            for phase in ("warmup", "sampling"):
                g = f"{phase}_stats"
                if g not in z or "acceptance_rate" not in z[g]:
                    continue
                acc = np.asarray(z[g]["acceptance_rate"][:])
                if acc.size == 0:
                    continue
                ss = np.asarray(z[g]["step_size"][:]) if "step_size" in z[g] else None
                lp = np.asarray(z[g]["lp"][:]) if "lp" in z[g] else None
                dead = []
                for c in range(acc.shape[0]):
                    i = death_index(acc[c])
                    if i is not None and (acc.shape[1] - i) >= 20:   # dead for >=20 draws, not a blip
                        dead.append((c, i))
                if dead:
                    rows.append(dict(run=os.path.basename(d), root=os.path.basename(root), phase=phase,
                                     chains=acc.shape[0], draws=acc.shape[1], n_dead=len(dead),
                                     first_death=min(i for _, i in dead),
                                     step_at_death=float(np.median([ss[c, max(i-1, 0)] for c, i in dead])) if ss is not None else None,
                                     step_alive=float(np.median(ss[:, -1])) if ss is not None else None,
                                     lp_at_death=float(np.median([lp[c, max(i-1, 0)] for c, i in dead])) if lp is not None else None,
                                     lp_alive=float(np.median(lp[:, -1])) if lp is not None else None,
                                     target_accept=cfg.get("target_accept"), tune=cfg.get("tune")))
        except Exception as e:
            rows.append(dict(run=os.path.basename(d), root=os.path.basename(root), phase="error",
                             chains=None, draws=None, n_dead=None, first_death=None,
                             step_at_death=None, step_alive=None, lp_at_death=None, lp_alive=None,
                             target_accept=None, tune=None, err=f"{type(e).__name__}: {e}"))

print("%-44s %-9s %-8s %5s %6s %6s %7s %11s %11s %9s" % (
    "run", "where", "phase", "dead", "chains", "draws", "died@", "step@death", "step alive", "target"))
for r in sorted(rows, key=lambda r: (r.get("n_dead") or 0), reverse=True):
    if r["phase"] == "error":
        print("%-44s %-9s ERROR %s" % (r["run"][:44], r["root"][:9], r.get("err")))
        continue
    print("%-44s %-9s %-8s %5d %6d %6d %7d %11.2e %11.2e %9s" % (
        r["run"][:44], ("floor" if r["run"].endswith("floor") and "no_floor" not in r["run"] else
                        "no-floor" if "no_floor" in r["run"] or "nofloor" in r["run"] else "?"),
        r["phase"], r["n_dead"], r["chains"], r["draws"], r["first_death"],
        r["step_at_death"] or float("nan"), r["step_alive"] or float("nan"), r["target_accept"]))
if not rows:
    print("no run has a chain dead for 20+ consecutive draws")
