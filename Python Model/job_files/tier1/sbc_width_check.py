"""Are the SBC posteriors as wide as the likelihood implies, and are the errors as large?

For every finished SBC replicate, on log scale:
  - the Laplace (Fisher-information) posterior sd at its truth, from finite-difference
    sensitivities of the Tier-1 design (baseline time series, baseline profile, five initial
    rates) with the data's own noise model;
  - the sampled posterior sd, and its ratio to the Laplace sd;
  - the standardized error z = (posterior mean - truth) / posterior sd, and the Mahalanobis
    distance of the truth, whose sum is chi-square on 2 x replicates degrees of freedom when
    the posteriors are calibrated.

A ratio near 1 says the sampler gives the width the likelihood implies. Small z with ratios near
1 then points to chance, not to the fit (written 2026-09-27, when the 10-replicate pilot's
z-scores had sd 0.52).

    python sbc_width_check.py            # every finished replicate in sbc_manifest.json
"""
import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import arviz as az
import numpy as np
from scipy import stats

import expected_information_grid as eig
from forward_model import CONDITIONS, FLOOR_CONC, ForwardModel

RESULTS = HERE.parent.parent / "Results" / "Tier1"
PARAMS = ["a1", "c3"]
H = 1e-3


def main():
    man = json.loads((HERE / "sbc_manifest.json").read_text())
    reps = [(k, r) for k, r in sorted(man["replicates"].items(), key=lambda kv: int(kv[0]))
            if (RESULTS / r["run"] / "posterior_samples_pm.nc").exists()]
    times = sorted(set(eig.SERIES_TIMES) | {eig.RATE_TIME, eig.END_TIME})
    fm = ForwardModel(man["system"], times=times)
    species = fm.names(r"C\d+_FA(_unsat)? \(uM\)")
    names = sorted(set(species) | {eig.TOTAL, eig.RATE})
    t = np.asarray(fm.times)
    si = [int(np.argmin(np.abs(t - s))) for s in eig.SERIES_TIMES]
    ei, ri = int(np.argmin(np.abs(t - eig.END_TIME))), int(np.argmin(np.abs(t - eig.RATE_TIME)))

    ratios, zs, d2s = [], [], []
    for k, rep in reps:
        truth = rep["truth"]
        base = eig.simulate(fm, fm.theta(dict(truth)), names)
        J = {n: [] for n in names}
        for p in PARAMS:
            up = eig.simulate(fm, fm.theta({**truth, p: truth[p] * math.exp(H)}), names)
            dn = eig.simulate(fm, fm.theta({**truth, p: truth[p] * math.exp(-H)}), names)
            for n in names:
                J[n].append((up[n] - dn[n]) / (2 * H))
        J = {n: np.stack(v, axis=-1) for n, v in J.items()}
        Jt = np.concatenate([J[eig.TOTAL][0, si, :], np.stack([J[n][0, ei, :] for n in species]),
                             J[eig.RATE][:, ri, :]])
        vt = np.concatenate([base[eig.TOTAL][0, si], np.array([base[n][0, ei] for n in species]),
                             base[eig.RATE][:, ri]])
        ft = np.concatenate([np.full(len(si), FLOOR_CONC), np.full(len(species), FLOOR_CONC),
                             np.full(len(CONDITIONS), eig.FLOOR_RATE)])
        lap_sd, _ = eig.score(Jt, vt, ft)

        post = az.from_netcdf(RESULTS / rep["run"] / "posterior_samples_pm.nc").posterior
        x = np.column_stack([np.log(post[p].values.ravel()) for p in PARAMS])
        mu, cov = x.mean(0), np.cov(x.T)
        err = mu - np.log([truth[p] for p in PARAMS])
        sd = np.sqrt(np.diag(cov))
        z = err / sd
        d2 = float(err @ np.linalg.solve(cov, err))
        ratios.append(sd / lap_sd)
        zs.extend(z.tolist())
        d2s.append(d2)
        print(f"sbc{int(k):03d}  truth " + " ".join(f"{p} {truth[p]:5.2f}" for p in PARAMS)
              + "  | sampled/Laplace sd " + " ".join(f"{r:.2f}" for r in sd / lap_sd)
              + "  | z " + " ".join(f"{v:+.2f}" for v in z) + f"  | Mahalanobis^2 {d2:.2f}", flush=True)

    ratios, zs = np.array(ratios), np.array(zs)
    n = len(d2s)
    print(f"\n{n} replicates")
    print("sampled/Laplace sd ratio, median: " + ", ".join(f"{p} {np.median(ratios[:, j]):.2f}"
                                                         for j, p in enumerate(PARAMS)))
    print(f"log-scale z: sd {zs.std():.2f} (1 when calibrated), largest |z| {np.abs(zs).max():.2f}")
    S = sum(d2s)
    print(f"sum of Mahalanobis^2 {S:.2f} on {2 * n} df: P(lower) {stats.chi2.cdf(S, 2 * n):.4f}, "
          f"P(upper) {stats.chi2.sf(S, 2 * n):.4f}")


if __name__ == "__main__":
    main()
