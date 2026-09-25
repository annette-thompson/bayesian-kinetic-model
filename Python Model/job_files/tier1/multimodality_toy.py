"""Known-answer multimodality test for the sampler and its diagnostics (outline 3.1).

A product forms through two parallel first-order routes with the same yield:

    P(t) = A * (2 - exp(-k1 t) - exp(-k2 t))

The model is symmetric in k1 and k2, so with truth (k1, k2) = (0.3, 3) the posterior has two
mirror-image modes, (0.3, 3) and (3, 0.3), of exactly equal mass (both priors are the same
LogNormal, 95% in [0.1, 10], median 1). That is the textbook label-switching construction:
the right answer is known to be bimodal, so we can check what our pipeline reports.

Each replicate runs the production sampler (BlackJAX NUTS through resumable_sampler, 4
chains, 300 warmup steps, the production stranded-chain rule) with early stopping off and a
fixed 1,000 draws, then records per replicate:
  - how many chains ended in each mode, and whether any chain crossed between modes
  - rank-normalised r-hat on k1 (a split across modes should push it far above 1.01)
  - the stranded-chain log-posterior gap (equal-mass modes should NOT be flagged: the rule
    is meant to catch low-posterior traps, not genuine competing modes)

The expected, validating result: r-hat flags every replicate whose chains split, and the lp
gap flags none of them. If the gap rule excluded chains here, it would be silently deleting
a real mode, which is the failure the SI item has to rule out.

No ODE: runs on a laptop CPU in a few minutes (~20 s per replicate).

Usage: python multimodality_toy.py [--replicates 10] [--out multimodality_toy.json]
"""
import argparse
import json
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent / "Utilities"))

import arviz as az
import numpy as np
import preliz as pz
import pymc as pm

import resumable_sampler as rs

TRUTH = {"k1": 0.3, "k2": 3.0}    # 1/min
A = 5.0                           # uM per route
TIMES = np.linspace(0.25, 10.0, 12)
NOISE_FRAC, NOISE_FLOOR = 0.10, 0.01


def curve(k1, k2, t):
    return A * (2.0 - np.exp(-k1 * t) - np.exp(-k2 * t))


def build_model(data, sigma):
    prior = pz.maxent(pz.LogNormal(), lower=0.1, upper=10.0, mass=0.95,
                      fixed_stat=["median", 1.0], plot=False)
    mu, s = (float(v) for v in prior.params)
    with pm.Model() as model:
        k1 = pm.LogNormal("k1", mu, s)
        k2 = pm.LogNormal("k2", mu, s)
        pred = A * (2.0 - pm.math.exp(-k1 * TIMES) - pm.math.exp(-k2 * TIMES))
        pm.Normal("y", mu=pred, sigma=sigma, observed=data)
    return model


def replicate(seed, n_tune, n_draws, lp_nats):
    rng = np.random.default_rng(seed)
    clean = curve(TRUTH["k1"], TRUTH["k2"], TIMES)
    sigma = NOISE_FRAC * np.abs(clean) + NOISE_FLOOR
    data = clean + rng.normal(0.0, sigma)
    model = build_model(data, sigma)
    bridge = rs.prepare_from_pymc(model, n_chains=4, random_seed=seed, jitter=True)
    spec = rs.SamplerSpec(n_tune=n_tune, n_draws=n_draws, n_chains=4, target_accept=0.8,
                          lp_exclusion_nats=lp_nats, min_chains_for_convergence=3,
                          rhat_threshold=None, ess_threshold=None, random_seed=seed,
                          max_total_hours=None, checkpoint_every=100)
    with tempfile.TemporaryDirectory() as run_dir:
        sampler = rs.ResumableSampler(bridge.logdensity_fn, bridge.initial_positions,
                                      bridge.value_var_names, spec, run_dir, bridge.config_hash)
        sampler.run()
        draws = rs.load_draws(run_dir, "sampling")
        stats = rs.load_stats(run_dir, "sampling")
    # value vars are log(k); mode label = which rate is the slow one
    lk1 = next(v for k, v in draws.items() if k.startswith("k1"))
    lk2 = next(v for k, v in draws.items() if k.startswith("k2"))
    slow_first = lk1 < lk2                                     # (chains, draws)
    frac = slow_first.mean(axis=1)
    chain_mode = np.where(frac > 0.5, "k1 slow", "k2 slow")
    crossed = bool(np.any((frac > 0.02) & (frac < 0.98)))
    rhat = float(az.rhat(az.convert_to_dataset({"k1": np.exp(lk1)}))["k1"])
    lp = stats["lp"]
    gaps = (np.nanmax(lp.mean(axis=1)) - lp.mean(axis=1)).round(2)
    return {
        "seed": seed,
        "chains_per_mode": {m: int((chain_mode == m).sum()) for m in ("k1 slow", "k2 slow")},
        "any_chain_crossed_modes": crossed,
        "rhat_k1": round(rhat, 3),
        "lp_gap_nats": gaps.tolist(),
        "flagged_as_stranded": [int(c) for c in np.where(gaps > lp_nats)[0]],
        "divergences": int(np.asarray(stats.get("diverging", np.zeros(1))).sum()),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--replicates", type=int, default=10)
    ap.add_argument("--tune", type=int, default=300)
    ap.add_argument("--draws", type=int, default=1000)
    ap.add_argument("--lp_nats", type=float, default=20.0, help="production stranded-chain threshold")
    ap.add_argument("--out", default=str(HERE / "multimodality_toy.json"))
    a = ap.parse_args()
    rows = []
    for seed in range(a.replicates):
        r = replicate(seed, a.tune, a.draws, a.lp_nats)
        rows.append(r)
        split = min(r["chains_per_mode"].values()) > 0
        print(f"seed {seed}: modes {r['chains_per_mode']}  split={split}  r-hat {r['rhat_k1']:.3f}  "
              f"crossed={r['any_chain_crossed_modes']}  lp gaps {r['lp_gap_nats']}  "
              f"stranded {r['flagged_as_stranded']}")
    split = [r for r in rows if min(r["chains_per_mode"].values()) > 0]
    summary = {
        "truth": TRUTH, "replicates": len(rows),
        "replicates_with_chains_in_both_modes": len(split),
        "of_those_flagged_by_rhat_gt_1.01": sum(r["rhat_k1"] > 1.01 for r in split),
        "replicates_with_any_chain_flagged_stranded": sum(bool(r["flagged_as_stranded"]) for r in rows),
        "replicates_with_a_chain_crossing_modes": sum(r["any_chain_crossed_modes"] for r in rows),
        "rows": rows,
    }
    Path(a.out).write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if k != "rows"}, indent=2))


if __name__ == "__main__":
    main()
