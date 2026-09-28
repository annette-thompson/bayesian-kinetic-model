"""Which scaling groups to free: a linear-Gaussian screen over all 18 at once, at the previous
(ME1) solution, for choosing a small subset that shows most of what the full model would.

What it computes, on one system with the Tier-1 data design (baseline time series at 72-720 s,
baseline profile at 720 s, initial rates under the five conditions at 150 s; sigma = 10% of the
noise-free value + floor, as every generated dataset uses):

1. Sensitivities. Every scaling group is nudged up and down by a small step on the coordinate
   the sampler uses (log for the LogNormal groups, the group itself for the additive d groups),
   one at a time, with all others at the solution: 1 + 2*18 = 37 parameter sets, each solved
   under the data conditions and the prediction conditions in one batch. Central differences
   give J (data x groups) and G (predictions x groups).
2. Fisher information and the all-free posterior. F = J^T W J with W = 1/sigma^2, the prior
   precision P = diag(1/prior sd^2) (the Tier-1 priors), and Sigma = (F + P)^-1: the Gaussian
   approximation of the posterior with all 18 groups free, i.e. the fit this screen avoids.
3. Predictions that matter, and how uncertain the data leave them. At the Tier-1 baseline, from
   the 720 s fatty acids: the three objectives (total production, average chain length,
   unsaturated fraction); each objective's local sensitivity to each of the nine enzymes
   (d objective / d log10 [enzyme], a local stand-in for Fig 7's Morris mu*); and the ratio
   strategy's local slope (d avg chain length / d log10 R, FabF and FabB up by sqrt(R), TesA
   down by it, Fig 8). Their posterior variance with all groups free is diag(G Sigma G^T).
4. Which groups carry that uncertainty. For each group i, how much each prediction's variance
   would fall if i were known exactly (conditioning Sigma on i), as a share of its variance.
5. Every subset S of 3, 4 or 5 groups, the others held at the solution. Its conditional
   posterior Sigma_S = (F_SS + P_SS)^-1 and the prediction variance it shows, diag(G_S Sigma_S
   G_S^T), as a fraction of the all-free variance ("captured"), averaged over the predictions;
   plus each subset's own identifiability (smallest contraction, variance left along its
   loosest direction: how strong a trade-off a point estimate would hide inside it).

6. Blocks: groups joined, directly or through each other, by an all-free posterior correlation
   above 0.2 (and 0.5). For each block: its variance with the other blocks held at the solution
   against its variance with everything free (near 1 means it can be fit on its own without
   hiding anything), its largest correlation with a group outside it, and the share of the
   prediction uncertainty it carries. This is the pairwise coupling Morris sigma/mu* cannot give:
   sigma/mu* says a group's effect depends on the others, not on which ones, and it is measured
   on the objectives over a 100-fold range rather than on the data near the solution.

Limits: linear-Gaussian, so it cannot see separate peaks or curved ridges (the a1+c2 kind); a
subset fit also cannot show a trade-off with a group it holds fixed, which is what step 4 is for.

  python parameter_screen.py                          # C20+unsat, all 18 groups
  python parameter_screen.py --system C8 --groups d1,d2   # quick check against predict_identifiability.py

Writes job_files/tier1/parameter_screen_<system>.json (the results) and .npz (J, G, sigma, F and
the prior sds, for reweighting or larger subsets without re-solving).
"""
import argparse
import itertools
import json
import math
import re
import sys
import time
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import numpy as np

import expected_information_grid as eig
from forward_model import CONDITIONS, ENZYMES, FLOOR_CONC, NOISE_FRAC, ForwardModel

_spec = spec_from_file_location("morris_screen", HERE.parent / "chain_system_sensitivity_analysis" / "morris_screen.py")
morris = module_from_spec(_spec)
_spec.loader.exec_module(morris)          # the objectives Figs 7 and 8 use

LOGNORMAL_SD = math.log(10.0) / 1.959964          # prior sd of log(x): 95% in [0.1, 10]
D_SAMPLE_SCALE = {"d1": 12.0, "d2": 1.0}          # build_tier1_configs.py
H = 1e-3            # parameter step: in log for LogNormal groups, in prior sd for d groups
DELTA = 0.05        # log10 step for the enzyme and ratio sensitivities (~12%)
FEASIBILITY = {"c2": "a1+c2 pilot never converged, even with the dense metric"}


def prior_sd(group):
    return LOGNORMAL_SD / D_SAMPLE_SCALE[group] if group in D_SAMPLE_SCALE else LOGNORMAL_SD


def prediction_rows(fm):
    """Baseline, each enzyme down and up by DELTA in log10, and the ratio down and up."""
    y0 = fm.y0()
    rows, labels = [y0.copy()], ["baseline"]
    for e in ENZYMES:
        for sign in (-1, 1):
            y = y0.copy()
            y[fm.index[e]] *= 10.0 ** (sign * DELTA)
            rows.append(y); labels.append(f"{e}{'+' if sign > 0 else '-'}")
    for sign in (-1, 1):
        y = y0.copy()
        for e in ("FabF", "FabB"):
            y[fm.index[e]] *= 10.0 ** (sign * DELTA / 2)
        y[fm.index["TesA"]] /= 10.0 ** (sign * DELTA / 2)
        rows.append(y); labels.append(f"ratio{'+' if sign > 0 else '-'}")
    return np.stack(rows), labels


def predictions(obj):
    """obj: (rows, n_obj) objectives in prediction_rows order -> named prediction vector."""
    names, vals = [], []
    n_obj = obj.shape[1]
    objn = OBJ_NAMES[:n_obj]
    for k, n in enumerate(objn):
        names.append(n); vals.append(obj[0, k])
    for j, e in enumerate(ENZYMES):
        lo, hi = obj[1 + 2 * j], obj[2 + 2 * j]
        for k, n in enumerate(objn):
            names.append(f"d {n} / d log10 {e}"); vals.append((hi[k] - lo[k]) / (2 * DELTA))
    lo, hi = obj[-2], obj[-1]
    names.append("d avg_chain_length / d log10 ratio"); vals.append((hi[1] - lo[1]) / (2 * DELTA))
    return names, np.array(vals)


OBJ_NAMES = ["total_production", "avg_chain_length", "unsat_fraction"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--system", default="C20+unsat")
    ap.add_argument("--groups", default=None, help="comma-separated subset to screen (default: every group)")
    ap.add_argument("--sizes", default="3,4,5", help="subset sizes to rank")
    a = ap.parse_args()
    t0 = time.time()

    times = sorted(set(eig.SERIES_TIMES) | {eig.RATE_TIME, eig.END_TIME})
    fm = ForwardModel(a.system, times=times)
    groups = a.groups.split(",") if a.groups else list(fm.groups)
    truth = dict(fm.nominal)                          # the ME1 solution: every group at its no-op
    species = fm.names(r"C\d+_FA(_unsat)? \(uM\)")
    targets = [s for s in fm.species if re.fullmatch(r"C\d+_FA(_unsat)?", s)]   # as posterior_morris.py
    objectives, obj_names = morris.make_objectives(fm._sys, targets, 1e-9)
    t = np.asarray(fm.times)
    si = [int(np.argmin(np.abs(t - s))) for s in eig.SERIES_TIMES]
    ei, ri = int(np.argmin(np.abs(t - eig.END_TIME))), int(np.argmin(np.abs(t - eig.RATE_TIME)))

    data_y0 = np.stack([fm.y0(changes) for _, changes in CONDITIONS])
    pred_y0, pred_labels = prediction_rows(fm)
    batch = np.concatenate([data_y0, pred_y0])
    n_data = len(data_y0)

    def evaluate(values):
        ys, ok = fm.run(batch, fm.theta(values))
        if not ok.all():
            bad = [i for i, o in enumerate(ok) if not o]
            raise RuntimeError(f"solve failed at {values} for rows {bad}")
        obs = fm.observe(ys[:n_data], [eig.TOTAL, eig.RATE] + species)
        data = np.concatenate([obs[eig.TOTAL][0, si], np.array([obs[n][0, ei] for n in species]), obs[eig.RATE][:, ri]])
        obj = np.stack([objectives(y[ei]) for y in ys[n_data:]])
        pnames, pred = predictions(obj)
        return data, pnames, pred

    base_data, pnames, base_pred = evaluate(truth)
    print(f"==> {a.system}: {len(groups)} groups, {len(base_data)} data points, {len(pnames)} predictions; "
          f"baseline solved in {time.time() - t0:.0f} s", flush=True)
    J, G = [], []
    for i, g in enumerate(groups):
        step = H * (prior_sd(g) if g in D_SAMPLE_SCALE else 1.0)
        if g in D_SAMPLE_SCALE:
            up, dn = truth[g] + step, truth[g] - step
        else:
            up, dn = truth[g] * math.exp(step), truth[g] * math.exp(-step)
        du, _, pu = evaluate({**truth, g: up})
        dd, _, pd_ = evaluate({**truth, g: dn})
        J.append((du - dd) / (2 * step)); G.append((pu - pd_) / (2 * step))
        print(f"  {g} ({i + 1}/{len(groups)}, {time.time() - t0:.0f} s)", flush=True)
    J, G = np.stack(J, axis=1), np.stack(G, axis=1)

    floors = np.concatenate([np.full(len(si), FLOOR_CONC), np.full(len(species), FLOOR_CONC),
                             np.full(len(CONDITIONS), eig.FLOOR_RATE)])
    sigma = NOISE_FRAC * np.abs(base_data) + floors
    F = J.T @ (J / sigma[:, None] ** 2)
    sd0 = np.array([prior_sd(g) for g in groups])
    P = np.diag(1.0 / sd0 ** 2)
    Sigma = np.linalg.inv(F + P)
    contraction = 1 - np.diag(Sigma) / sd0 ** 2
    std = Sigma / np.outer(sd0, sd0)
    lam, vec = np.linalg.eigh(std)
    corr = Sigma / np.sqrt(np.outer(np.diag(Sigma), np.diag(Sigma)))

    v_prior = np.einsum("pi,ij,pj->p", G, np.diag(sd0 ** 2), G)
    v_full = np.einsum("pi,ij,pj->p", G, Sigma, G)
    keep = v_full > 1e-12 * max(v_full.max(), 1e-300)          # predictions the groups move at all

    # Share of each prediction's all-free variance that group i carries: the fall in variance
    # if i were known exactly (Sigma conditioned on i), over the all-free variance.
    carries = {}
    for i, g in enumerate(groups):
        s_i = Sigma[:, i]
        cond = Sigma - np.outer(s_i, s_i) / Sigma[i, i]
        v_cond = np.einsum("pi,ij,pj->p", G, cond, G)
        carries[g] = float(np.mean(((v_full - v_cond) / v_full)[keep]))

    def subset_stats(S):
        idx = [groups.index(g) for g in S]
        FS = F[np.ix_(idx, idx)] + np.diag(1.0 / sd0[idx] ** 2)
        SS = np.linalg.inv(FS)
        v = np.einsum("pi,ij,pj->p", G[:, idx], SS, G[:, idx])
        stdS = SS / np.outer(sd0[idx], sd0[idx])
        return {"groups": list(S), "captured_mean": float(np.mean((v / v_full)[keep])),
                "captured_median": float(np.median((v / v_full)[keep])),
                "min_contraction": float(1 - np.diag(stdS).min()),
                "loosest_variance_left": float(np.linalg.eigvalsh(stdS).max()),
                "flags": [FEASIBILITY[g] for g in S if g in FEASIBILITY]}

    sizes = [int(k) for k in a.sizes.split(",") if int(k) <= len(groups)]
    ranked = {}
    for k in sizes:
        rows = [subset_stats(S) for S in itertools.combinations(groups, k)]
        rows.sort(key=lambda r: -r["captured_mean"])
        ranked[str(k)] = rows[:25]
    reference = {name: subset_stats(S) for name, S in
                 (("R2 trio a1+c3+a2", ("a1", "c3", "a2")), ("R9 quad a1+c3+a2+b3", ("a1", "c3", "a2", "b3")))
                 if all(g in groups for g in S)}
    with_trio = {}
    if all(g in groups for g in ("a1", "c3", "a2")):
        for k in (4, 5):
            extra = [g for g in groups if g not in ("a1", "c3", "a2")]
            rows = [subset_stats(("a1", "c3", "a2") + S) for S in itertools.combinations(extra, k - 3)]
            rows.sort(key=lambda r: -r["captured_mean"])
            with_trio[str(k)] = rows[:10]

    def v_known(idx):
        """Each prediction's variance if the groups idx were known exactly (Sigma conditioned)."""
        rest = [i for i in range(len(groups)) if i not in idx]
        if not rest:
            return np.zeros(len(pnames))
        S_rr = Sigma[np.ix_(rest, rest)] - Sigma[np.ix_(rest, idx)] @ np.linalg.solve(
            Sigma[np.ix_(idx, idx)], Sigma[np.ix_(idx, rest)])
        return np.einsum("pi,ij,pj->p", G[:, rest], S_rr, G[:, rest])

    # Blocks: groups joined, directly or through each other, by an all-free posterior correlation
    # above the threshold. A block is safe to fit on its own when its variance with the other
    # blocks held fixed matches its variance with everything free (ratio near 1): holding the rest
    # at the solution then hides nothing about it.
    def blocks_at(threshold):
        n, seen, comps = len(groups), set(), []
        for s in range(n):
            if s in seen:
                continue
            stack, comp = [s], []
            seen.add(s)
            while stack:
                i = stack.pop()
                comp.append(i)
                for j in range(n):
                    if j not in seen and abs(corr[i, j]) > threshold:
                        seen.add(j)
                        stack.append(j)
            comps.append(sorted(comp))
        rows = []
        for comp in sorted(comps, key=len, reverse=True):
            FS = F[np.ix_(comp, comp)] + np.diag(1.0 / sd0[comp] ** 2)
            ratio = np.diag(np.linalg.inv(FS)) / np.diag(Sigma)[comp]
            outside = [abs(corr[i, j]) for i in comp for j in range(n) if j not in comp]
            rows.append({"groups": [groups[i] for i in comp],
                         "held_fixed_over_all_free_variance_min": float(ratio.min()),
                         "largest_correlation_outside": float(max(outside, default=0.0)),
                         "prediction_share_carried": float(np.mean(((v_full - v_known(comp)) / v_full)[keep])),
                         "contraction_all_free": {groups[i]: float(contraction[i]) for i in comp}})
        return rows

    blocks = {str(th): blocks_at(th) for th in (0.2, 0.5)}

    out = {"system": a.system, "groups": groups, "truth": truth, "n_data": int(len(base_data)),
           "prior_sd": dict(zip(groups, sd0.tolist())),
           "all_free": {"contraction": dict(zip(groups, contraction.tolist())),
                        "correlation": corr.tolist(),
                        "eigen": [{"variance_left": float(l), "direction": dict(zip(groups, vec[:, j].tolist()))}
                                  for j, l in enumerate(lam)]},
           "predictions": [{"name": n, "value": float(v), "sd_prior": float(math.sqrt(vp)),
                            "sd_all_free": float(math.sqrt(vf)), "moved": bool(kp)}
                           for n, v, vp, vf, kp in zip(pnames, base_pred, v_prior, v_full, keep)],
           "carries": dict(sorted(carries.items(), key=lambda kv: -kv[1])),
           "blocks": blocks, "reference_subsets": reference, "best_with_trio": with_trio, "ranked": ranked,
           "seconds": round(time.time() - t0, 1)}
    path = HERE / f"parameter_screen_{a.system}.json"
    path.write_text(json.dumps(out, indent=1) + "\n")
    # The matrices themselves, for reweighting the predictions or scoring larger subsets later
    # without re-solving.
    np.savez(HERE / f"parameter_screen_{a.system}.npz", groups=np.array(groups), predictions=np.array(pnames),
             J=J, G=G, sigma=sigma, F=F, prior_sd=sd0, data=base_data, prediction_values=base_pred)
    print(f"\nall groups free: contraction " + ", ".join(f"{g} {c:.3f}" for g, c in zip(groups, contraction)))
    print("share of prediction uncertainty each group carries: "
          + ", ".join(f"{g} {v:.2f}" for g, v in out["carries"].items()))
    for th, rows in blocks.items():
        print(f"blocks at |r| > {th}: " + "; ".join(
            f"{'+'.join(b['groups'])} (fixed/free var {b['held_fixed_over_all_free_variance_min']:.2f}, "
            f"carries {b['prediction_share_carried']:.2f})" for b in rows))
    for name, r in reference.items():
        print(f"{name}: captures {r['captured_mean']:.2f} (median {r['captured_median']:.2f}); "
              f"loosest direction keeps {r['loosest_variance_left']:.3f}")
    for k, rows in ranked.items():
        print(f"best {k}-group subsets: " + "; ".join(f"{'+'.join(r['groups'])} {r['captured_mean']:.2f}" for r in rows[:5]))
    print(f"wrote {path} ({out['seconds']:.0f} s)")


if __name__ == "__main__":
    main()
