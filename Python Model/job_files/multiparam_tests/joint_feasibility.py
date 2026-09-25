"""Does a proposed JOINT multi-parameter prior box actually solve, under no-floor?

info_vs_conditions.py calibrates a defensible bound for one parameter at a time,
holding everything else at nominal -- it never tests whether two independently-
reasonable marginal boxes combine into a region where the ODE stops solving. A
parameter can look fine alone and still combine with another into trouble; this is
the cheap check for that, run BEFORE any real sampling job, so a bad joint box gets
caught as a measured percentage instead of discovered as a dead run on the cluster.

Draws a Latin hypercube over the joint box (log-space, since these are LogNormal-ish
multiplicative parameters -- and d1/d2 additive-in-exp), reports the fraction that
fail to solve (hit max_steps or diverge) across every real endpoint condition PLUS
the baseline timeseries, under no-floor (the negative-concentration floor removed,
matching truncated_warmup_test.py's own "nofloor" monkeypatch) since that's the
condition being screened for. Decision rule, stated up front rather than judged
after the fact: >10% joint-box failure means shrink the box and re-check before
submitting a real sampling job.

Usage:
    python joint_feasibility.py --system C6 --params a1,c2 --bounds 0.1,10,0.087,1.35
    (bounds are lower1,upper1,lower2,upper2,... matching --params order)
    python joint_feasibility.py --system C6 --params a1 --bounds 0.1,10   # sanity
        check: single free parameter, should show ~0% failure (matches production)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, "/projects/anth4580/Bayesian/Utilities")
sys.path.insert(0, str(HERE))

import numpy as np
import jax.numpy as jnp
import diffrax as dfrx
import reaction_model_builder as rmb
from reaction_model_builder import set_scaling_group_values
import info_vs_conditions as ivc

N_DRAWS = 200          # matches the Morris r=200 base-point convention elsewhere
FAIL_THRESHOLD = 0.10  # >10% of the joint box failing to solve -> shrink and re-check


def _unclamped_call(self, t, y, args):
    """No-floor: identical to truncated_warmup_test.py's MASK_TEST_MODE=nofloor
    monkeypatch, minus the print -- removes the jnp.maximum(y, 0.0) floor from
    the reaction-network right-hand side."""
    if self.param_idx_arr.shape[0] == 0:
        return jnp.zeros_like(y)
    theta = jnp.asarray(args)
    reactant_conc = y[self.reactant_idx_arr]
    reactant_powers = jnp.where(
        self.reactant_mask_arr, reactant_conc ** self.reactant_stoich_arr, 1.0,
    )
    mass_action_terms = jnp.prod(reactant_powers, axis=1)
    scale_factors = jnp.stack([
        fn(theta) if fn is not None else jnp.ones(()) for fn in self.scale_fns
    ])
    rates = theta[self.param_idx_arr] * scale_factors * mass_action_terms
    return self.stoich_matrix @ rates


def latin_hypercube_log(bounds_pairs, n, seed=0):
    """(n, k) array, one column per (lower, upper) pair, log-uniform, via a
    standard LHS stratification (each column independently permuted)."""
    rng = np.random.default_rng(seed)
    k = len(bounds_pairs)
    cut = np.linspace(0, 1, n + 1)
    u = rng.uniform(size=(n, k)) * (cut[1] - cut[0]) + cut[:-1, None]
    for j in range(k):
        rng.shuffle(u[:, j])
    out = np.empty((n, k))
    for j, (lo, hi) in enumerate(bounds_pairs):
        out[:, j] = 10 ** (np.log10(lo) + u[:, j] * (np.log10(hi) - np.log10(lo)))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--system", required=True)
    ap.add_argument("--params", required=True, help="comma-separated, e.g. a1,c2")
    ap.add_argument("--bounds", required=True, help="lower1,upper1,lower2,upper2,... matching --params order")
    ap.add_argument("--n", type=int, default=N_DRAWS)
    a = ap.parse_args()

    params = a.params.split(",")
    vals = [float(x) for x in a.bounds.split(",")]
    if len(vals) != 2 * len(params):
        raise SystemExit(f"--bounds needs {2*len(params)} numbers for {len(params)} params, got {len(vals)}")
    bounds_pairs = [(vals[2 * i], vals[2 * i + 1]) for i in range(len(params))]

    print(f"=== joint feasibility: {a.system}, params={params}, bounds={bounds_pairs} ===")
    print("Applying no-floor monkeypatch (matches truncated_warmup_test.py MASK_TEST_MODE=nofloor)...")
    rmb.ReactionNetwork.__call__ = _unclamped_call

    sys_, targets, ctrl = ivc.build(a.system)
    sg = json.loads((ivc.CFG_ROOT / f"Chain {a.system} - a1_0.1-10_floor" / "solver_params.json").read_text())["scaling_groups"]
    sg = {k: float(v) for k, v in sg.items()}
    idx = [sys_.index_of[t] for t in targets]
    y0s, ep_obs, times, ts_obs, sig_ep, sig_ts = ivc.load_data(sys_, a.system, targets)

    draws = latin_hypercube_log(bounds_pairs, a.n)
    n_fail = 0
    fail_examples = []
    for row in draws:
        overrides = dict(sg)
        for p, v in zip(params, row):
            overrides[p] = ivc.group_value_for(p, v)
        theta = set_scaling_group_values(sys_.theta, sys_.params, overrides)

        ok = True
        # Every endpoint condition, final time only (cheap: SaveAt(t1=True)).
        for y0 in y0s:
            if ivc.solve(sys_, theta, y0, ctrl, [times[-1]], idx) is None:
                ok = False
                break
        # Plus the baseline timeseries across all its saved times.
        if ok and ivc.solve(sys_, theta, y0s[0], ctrl, times, idx) is None:
            ok = False

        if not ok:
            n_fail += 1
            if len(fail_examples) < 5:
                fail_examples.append(dict(zip(params, [float(v) for v in row])))

    frac = n_fail / len(draws)
    print(f"\n{n_fail}/{len(draws)} draws ({frac:.1%}) failed to solve under no-floor.")
    if fail_examples:
        print("Example failing points:")
        for ex in fail_examples:
            print(f"  {ex}")
    if frac > FAIL_THRESHOLD:
        print(f"\nFAIL: {frac:.1%} > {FAIL_THRESHOLD:.0%} threshold -- shrink the box "
              "(tighten toward nominal) and re-run before submitting a real sampling job.")
    else:
        print(f"\nOK: {frac:.1%} <= {FAIL_THRESHOLD:.0%} threshold -- box is safe to use as-is.")

    out = HERE / f"joint_feasibility_{a.system}_{'_'.join(params)}.json"
    out.write_text(json.dumps(dict(
        system=a.system, params=params, bounds=bounds_pairs, n=len(draws),
        n_fail=n_fail, frac_fail=frac, threshold=FAIL_THRESHOLD,
        passed=frac <= FAIL_THRESHOLD, fail_examples=fail_examples,
    ), indent=2))
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
