"""Morris elementary-effects screen over the SCALING GROUPS.

Follows the method Ruppe et al. (PNAS 2020) used on this model family, so the result
is directly comparable to the group's prior sensitivity work: mean elementary effects
via the Morris method, radial design, Latin hypercube base points, convergence checked
by averaging over random subsets of trajectories.

WHAT IS DIFFERENT FROM THE PNAS SCREEN, AND WHY

The PNAS analysis varied ENZYME CONCENTRATIONS (0.1-10 uM for essential enzymes,
0-10 uM for the partially redundant ones) and found FabF and TesA dominate chain
length. That answers "which enzyme should an engineer titrate". This screen varies the
KINETIC SCALING GROUPS instead, which answers "which parameter is worth inferring" --
a different question on the same model, and the one the prior work leaves open.

Two consequences follow for the sampling design:

  * Scaling groups are MULTIPLIERS on rate constants, so the natural sample space is
    uniform in log10 over [1/SPAN, SPAN] rather than uniform on a linear concentration
    axis. Uniform-linear would put 90% of its mass above 1.0 and barely probe
    reductions at all.
  * The d-prefixed groups enter additively inside exp() (TesA's 1/exp(12*d1+d2)), so a
    raw value is not comparable to a multiplier. They are sampled on the induced
    RATE multiplier and converted, exactly as in the sensitivity sweep, so every group
    is screened on one common axis.

SPAN=10 mirrors the PNAS 0.1-10 uM window. They reported that widening to 0.1-100
changed the elementary effects little; the same check is available here by re-running
with --span 100.

OUTPUTS
  mu_star  mean |elementary effect| -- overall influence. The ranking metric.
  sigma    spread of elementary effects -- high values mean the effect depends on
           where in parameter space you are, i.e. nonlinearity or interaction with
           other groups. A group with large sigma cannot be trusted to a one-at-a-time
           sweep, which is exactly the blind spot of everything measured so far.

Usage:
  python morris_screen.py --system C6 --r 200 [--span 10] [--obs-floor 1e-9]
"""
import argparse
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, "/projects/anth4580/Bayesian/Utilities")

import numpy as np
import pandas as pd
import jax.numpy as jnp
import diffrax as dfrx
import generate_chain_data as gcd
from reaction_model_builder import set_scaling_group_values

gcd.TIME_RANGE = (0.0, 150.0)
ROOT = gcd.project_root()
HARD_CAP = 20_000


def group_value_for(group, mult):
    """Value of `group` that induces rate multiplier `mult` (see module docstring)."""
    if group == "d1":
        return -np.log(mult) / 12.0
    if group == "d2":
        return -np.log(mult)
    return mult


def build(system):
    cfg = json.loads((ROOT / "Results/Chain Scaling Tests" / f"Chain {system} - a1 tightest"
                      / "solver_params.json").read_text())
    sg = {k: float(v) for k, v in cfg["scaling_groups"].items()}
    srcs = [ROOT / p for p in cfg["output_paths"]["reactions_source"]]
    ctrl = cfg["ODE_stepsize_controller"]
    sys_ = gcd.ChainSystem(srcs, rtol=ctrl["rtol"], atol=ctrl["atol"], pcoeff=ctrl["pcoeff"],
                           icoeff=ctrl["icoeff"], dcoeff=ctrl["dcoeff"],
                           scaling_group_overrides=sg)
    targets = sys_.targets(gcd.UNSAT_PATTERN if "+unsat" in system else gcd.SAT_PATTERN)
    return sys_, targets, sg, ctrl


def baseline_y0(sys_, system):
    """The unperturbed condition -- one solve per evaluation keeps the screen cheap."""
    df = pd.read_csv(ROOT / "Data" / f"Chain_{system}" / "init_vs_final_conc.csv")
    sweep = [n for n in gcd.SWEEP_SPECIES if n in sys_.index_of]
    y0 = np.array(sys_.y0(), dtype=float)
    r = df.iloc[0]
    for n in sweep:
        col = f"{n} (uM)"
        if col in df.columns:
            y0[sys_.index_of[n]] = float(r[col])
    return y0


def make_objectives(sys_, targets, obs_floor):
    """The objectives ACS and PNAS screened: total production, average chain length,
    and -- for systems that actually have an unsaturated branch -- unsaturated
    fraction. Reporting unsat fraction on a saturated system would be a constant zero
    and would produce meaningless elementary effects, so it is only added when
    unsaturated targets exist."""
    idx = [sys_.index_of[t] for t in targets]
    lengths = np.array([int("".join(ch for ch in t.split("_")[0] if ch.isdigit()))
                        for t in targets], dtype=float)
    is_unsat = np.array([t.endswith("_unsat") for t in targets], dtype=bool)
    has_unsat = bool(is_unsat.any())
    names = ["total_production", "avg_chain_length"]
    if has_unsat:
        names.append("unsat_fraction")

    def objectives(final):
        conc = np.array([float(final[i]) for i in idx])
        conc = np.where(conc > obs_floor, conc, 0.0)
        total = float(conc.sum())
        avg_cl = float((conc * lengths).sum() / total) if total > 0 else 0.0
        out = [total, avg_cl]
        if has_unsat:
            out.append(float(conc[is_unsat].sum() / total) if total > 0 else 0.0)
        return np.array(out)
    return objectives, names


def evaluate(sys_, sg, groups, log_mults, y0, ctrl, objectives, n_obj):
    """Solve once at the given log10 rate multipliers; NaN objectives on failure."""
    o = dict(sg)
    for g, lm in zip(groups, log_mults):
        o[g] = group_value_for(g, 10.0 ** lm)
    theta = set_scaling_group_values(sys_.theta, sys_.params, o)
    sol = dfrx.diffeqsolve(
        dfrx.ODETerm(sys_.network), dfrx.Kvaerno5(),
        t0=gcd.TIME_RANGE[0], t1=gcd.TIME_RANGE[1], dt0=1e-6,
        y0=jnp.asarray(y0, dtype=jnp.float64), args=theta,
        saveat=dfrx.SaveAt(t1=True),
        stepsize_controller=dfrx.PIDController(
            rtol=ctrl["rtol"], atol=ctrl["atol"], pcoeff=ctrl["pcoeff"],
            icoeff=ctrl["icoeff"], dcoeff=ctrl["dcoeff"]),
        max_steps=HARD_CAP, throw=False)
    if not bool(sol.result == dfrx.RESULTS.successful):
        # must match the objective count, which is 3 on +unsat systems
        return np.full(n_obj, np.nan)
    final = np.asarray(sol.ys[-1] if sol.ys.ndim == 2 else sol.ys)
    return objectives(final)


def lhs(n, k, rng):
    """Latin hypercube on the unit cube, matching the PNAS sampling choice."""
    out = np.empty((n, k))
    for j in range(k):
        out[:, j] = (rng.permutation(n) + rng.random(n)) / n
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--system", required=True)
    ap.add_argument("--r", type=int, default=200, help="radial base points (trajectories)")
    ap.add_argument("--span", type=float, default=10.0, help="sample space is [1/span, span]")
    ap.add_argument("--obs-floor", type=float, default=1e-9)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    sys_, targets, sg, ctrl = build(a.system)
    groups = sorted(sg)
    k = len(groups)
    y0 = baseline_y0(sys_, a.system)
    objectives, names = make_objectives(sys_, targets, a.obs_floor)
    rng = np.random.default_rng(a.seed)
    L = np.log10(a.span)

    n_eval = a.r * (k + 1)
    print(f"=== Morris screen: {a.system} ===")
    print(f"  {k} scaling groups: {groups}")
    print(f"  sample space: uniform in log10 over [{1/a.span:g}, {a.span:g}]")
    print(f"  r={a.r} radial base points -> N = r*(k+1) = {n_eval} model evaluations")
    print(f"  objectives: {', '.join(names)} "
          f"({len(targets)} observables, {sum(t.endswith(chr(95)+chr(117)+chr(110)+chr(115)+chr(97)+chr(116)) for t in targets)} unsaturated)\n", flush=True)

    # Radial design: a base point and an auxiliary point per trajectory; each parameter
    # is moved from base to auxiliary one at a time (Campolongo et al. radial variant).
    base = (lhs(a.r, k, rng) * 2 - 1) * L
    aux = (lhs(a.r, k, rng) * 2 - 1) * L

    ee = np.full((a.r, k, len(names)), np.nan)
    t0 = time.time()
    n_fail = 0
    for i in range(a.r):
        f0 = evaluate(sys_, sg, groups, base[i], y0, ctrl, objectives, len(names))
        if np.isnan(f0).any():
            n_fail += 1
            continue
        for j in range(k):
            x = base[i].copy()
            x[j] = aux[i, j]
            delta = x[j] - base[i][j]
            if abs(delta) < 1e-12:
                continue
            fj = evaluate(sys_, sg, groups, x, y0, ctrl, objectives, len(names))
            if np.isnan(fj).any():
                n_fail += 1
                continue
            ee[i, j, :] = (fj - f0) / delta
        if (i + 1) % max(1, a.r // 10) == 0:
            el = time.time() - t0
            print(f"  trajectory {i+1}/{a.r}  ({el:.0f}s elapsed, "
                  f"{el/(i+1)*(a.r-i-1):.0f}s remaining, {n_fail} failed solves)", flush=True)

    results = {}
    for o, oname in enumerate(names):
        mu_star = np.nanmean(np.abs(ee[:, :, o]), axis=0)
        sigma = np.nanstd(ee[:, :, o], axis=0)
        n_ok = np.sum(~np.isnan(ee[:, :, o]), axis=0)
        order = np.argsort(-np.nan_to_num(mu_star))
        print(f"\n=== {oname} ===")
        print(f"  {'group':<7}{'mu_star':>14}{'sigma':>14}{'sigma/mu*':>12}{'n':>7}")
        print("  " + "-" * 54)
        rows = []
        for j in order:
            ratio = (sigma[j] / mu_star[j]) if mu_star[j] else float("nan")
            print(f"  {groups[j]:<7}{mu_star[j]:>14.4g}{sigma[j]:>14.4g}"
                  f"{ratio:>12.2f}{n_ok[j]:>7}")
            rows.append(dict(group=groups[j], mu_star=float(mu_star[j]),
                             sigma=float(sigma[j]), n=int(n_ok[j])))
        results[oname] = rows

    # Convergence check, following the PNAS approach of averaging over random subsets.
    print("\n=== convergence of mu_star (random trajectory subsets) ===")
    conv = {}
    for o, oname in enumerate(names):
        print(f"  {oname}:")
        prev = None
        for frac in (0.25, 0.5, 0.75, 1.0):
            m = max(2, int(a.r * frac))
            sel = rng.choice(a.r, m, replace=False)
            mu = np.nanmean(np.abs(ee[sel][:, :, o]), axis=0)
            drift = ("" if prev is None else
                     f"   max shift vs previous: {np.nanmax(np.abs(mu - prev)):.3g}")
            print(f"    r={m:<5} top group = {groups[int(np.nanargmax(mu))]}{drift}")
            prev = mu
        conv[oname] = True

    out = HERE / f"morris_{a.system.replace('+','_')}.json"
    out.write_text(json.dumps(dict(system=a.system, r=a.r, span=a.span, groups=groups,
                                   n_eval=n_eval, n_failed=n_fail,
                                   results=results), indent=2))
    print(f"\nwrote {out}\nDONE  ({time.time()-t0:.0f}s, {n_fail} failed solves)", flush=True)


if __name__ == "__main__":
    main()
