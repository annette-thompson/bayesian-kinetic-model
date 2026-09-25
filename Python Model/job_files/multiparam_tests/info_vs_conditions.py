"""How much information each endpoint condition buys under the Tier-1 data design, so
the condition count can be chosen on evidence.

Tier-1 design (Data/Tier1/Chain_<system>/): one time series of total fatty acid in C16
Equivalents (uM) at the base condition, plus per-species fatty-acid endpoints at the time
series' final time (720 s) over the sweep conditions. Cost is the number of UNIQUE initial conditions (~535 ms each on
C8). The time series runs at the base condition, which is also endpoint row 0, so the
base endpoint costs nothing extra: n endpoint conditions cost max(n, 1) solves.

Two measurements per system:

  PROFILE  per parameter, every other group at nominal: the delta=2.23 profile interval
           vs endpoint-condition count. Its validity was established by calibrating it
           against three measured single-parameter posteriors (reproduces the 95%
           interval to within 0.2% where the posterior is well identified).
  JOINT    every scaling group at once: Fisher information J^T diag(1/sigma^2) J from
           central-difference sensitivities in ln(multiplier), plus the [0.1, 10] prior
           every group carries (d1's converted window has the same log-space sd). A
           profile holds the other parameters fixed, so it cannot see two parameters
           trading off; the marginal width from the joint matrix can. Widths are put
           on the profile's scale (half-width sqrt(2*delta) sd) so the two compare.

Expected, not realised, information: the "observed" values are the model's own
noise-free prediction at nominal truth, and sigma comes from the Tier-1 files' sigma
columns (computed from the clean values before noise was added; one value per column
for relative_mean, one per point for pointwise). For a Gaussian likelihood with known
sigma the profile of noise-free data IS the noise-averaged profile, so nothing here
depends on the noise draw in the Tier-1 CSVs. Each point keeps its own sigma when a
design uses fewer conditions.

Condition orders compared (row 0, the base, is always first):
  row     file order
  cheap   fewest solver steps first -- what a cost-minimising design would pick
  greedy  each step adds the condition that most increases log det of the joint
          Fisher matrix over --params (D-optimal)

Likelihood replicated exactly: logp = sum Normal(observed | predicted, sigma), matching
the pm.Normal in inference_runner._build_pymc_model, with the model's (floorless) RHS.

SELF-CHECKS, printed not assumed:
  --legacy --system C8 --params a1 replays the old ladder design (per-species time
    series, relative_mean sigma recomputed from the data) and must reproduce C8's
    measured (no-floor) posterior [0.9796, 1.0204]: the rewrite still replicates the likelihood.
  Every profiled parameter's profile width must match its conditional Fisher width at
    the full design: the finite-difference sensitivities are right.

Writes info_tier1_<system>.json (tables) and .npz (sensitivities and per-condition
log-likelihood grids, so any other condition subset or parameter set can be scored in
the notebook without re-solving).

Usage: python info_vs_conditions.py --system C12 [--params a1,a2,c2,c3,d1]
       python info_vs_conditions.py --legacy --system C8 --params a1
"""
import argparse
import json
import sys
from importlib.util import module_from_spec, spec_from_file_location
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
CFG_ROOT = ROOT / "Results" / "Chain Scaling Tests"
HARD_CAP = 20_000
DELTA = 2.23          # calibrated against measured posteriors; see decision record D4
NOISE_FRAC = 0.10
SD_MULT = np.sqrt(2 * DELTA)        # a delta-profile half-width, in sds
PRIOR_SD_LN = np.log(10.0) / 1.96   # [0.1, 10] at 95%, in ln(multiplier)
FD_STEP = 0.05                      # log10 multiplier; central difference
DEFAULT_PARAMS = "a1,a2,c2,c3,d1"

# 95% intervals of the no-floor a1 [0.1, 10] ladder posteriors (the floor runs gave
# [0.4220, 1.2957], [0.9667, 1.0307], [0.9792, 1.0207]).
MEASURED_POSTERIOR = {"C4_NoFB": (0.3899, 1.2968), "C6": (0.9654, 1.0315),
                      "C8": (0.9796, 1.0204)}


def config_path(system):
    """The system's no-floor a1 [0.1, 10] config. A run keeps its old directory name
    until it finishes (e.g. C18+unsat's "a1 tightest nofloor-eqxnan"), so fall back to it."""
    for name in ("a1_0.1-10_no_floor", "a1 tightest nofloor-eqxnan"):
        p = CFG_ROOT / f"Chain {system} - {name}" / "solver_params.json"
        if p.exists():
            return p
    raise FileNotFoundError(f"no no-floor a1 config for {system}")


def group_value_for(group, mult):
    if group == "d1":
        return -np.log(mult) / 12.0
    if group == "d2":
        return -np.log(mult)
    return mult


def build(system):
    cfg = json.loads(config_path(system).read_text())
    sg = {k: float(v) for k, v in cfg["scaling_groups"].items()}
    srcs = [ROOT / p for p in cfg["output_paths"]["reactions_source"]]
    ctrl = cfg["ODE_stepsize_controller"]
    sys_ = gcd.ChainSystem(srcs, rtol=ctrl["rtol"], atol=ctrl["atol"], pcoeff=ctrl["pcoeff"],
                           icoeff=ctrl["icoeff"], dcoeff=ctrl["dcoeff"],
                           scaling_group_overrides=sg)
    targets = sys_.targets(gcd.UNSAT_PATTERN if "+unsat" in system else gcd.SAT_PATTERN)
    return sys_, targets, ctrl


def _endpoint_y0s(sys_, ep):
    sweep = [n for n in gcd.SWEEP_SPECIES if n in sys_.index_of]
    y0s = []
    for _, r in ep.iterrows():
        y0 = np.array(sys_.y0(), dtype=float)
        for n in sweep:
            c = f"{n} (uM)"
            if c in ep.columns:
                y0[sys_.index_of[n]] = float(r[c])
        y0s.append(y0)
    return y0s


def load_data(sys_, system, targets):
    """The old ladder design (per-species time series, relative_mean sigma recomputed
    from the data file). Used by --legacy and by joint_feasibility.py."""
    ep = pd.read_csv(ROOT / "Data" / f"Chain_{system}" / "init_vs_final_conc.csv")
    ts = pd.read_csv(ROOT / "Data" / f"Chain_{system}" / "time_vs_conc.csv")
    ocols = [f"{t} (uM)" for t in targets]
    y0s = _endpoint_y0s(sys_, ep)
    obs = [np.array([float(r[c]) for c in ocols]) for _, r in ep.iterrows()]
    tcol = [c for c in ts.columns if c.lower().startswith("time")][0]
    times = ts[tcol].to_numpy(dtype=float)
    ts_obs = ts[ocols].to_numpy(dtype=float)
    # relative_mean: one sigma per column, from the mean magnitude of that column
    sig_ep = NOISE_FRAC * np.abs(np.stack(obs)).mean(axis=0)
    sig_ts = NOISE_FRAC * np.abs(ts_obs).mean(axis=0)
    return y0s, np.stack(obs), times, ts_obs, sig_ep, sig_ts


def _load_fa_conc():
    """The calculation module the fit uses, so the C16-equivalent weights here are the
    same function the model's observable evaluates."""
    path = ROOT / "Calculation Files" / "Full_FAS" / "FA_conc.py"
    spec = spec_from_file_location("FA_conc", path)
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_tier1(sys_, system, targets, data_dir=None):
    """Returns (y0s, times, sig_ep [conditions x species], sig_ts [times], weights)."""
    data_dir = Path(data_dir) if data_dir else ROOT / "Data" / "Tier1" / f"Chain_{system}"
    ep = pd.read_csv(data_dir / "init_vs_final_conc.csv")
    ts = pd.read_csv(data_dir / "time_vs_conc.csv")
    fa = _load_fa_conc()
    y0s = _endpoint_y0s(sys_, ep)
    if not np.allclose(y0s[0], np.asarray(sys_.y0(), dtype=float)):
        raise RuntimeError("endpoint row 0 is not the base condition the time series runs at")

    ocols = [f"{t} (uM)" for t in targets]
    sig_ep = np.stack([ep[f"{c}_sigma"].to_numpy(dtype=float) for c in ocols], axis=1)
    times = ts["Time (s)"].to_numpy(dtype=float)
    sig_ts = ts[f"{fa.C16_EQUIV_NAME}_sigma"].to_numpy(dtype=float)
    weights = np.array([fa.c16_equiv_weight(t) for t in targets])
    return y0s, times, sig_ep, sig_ts, weights


def solve(sys_, theta, y0, ctrl, ts, idx):
    sol = dfrx.diffeqsolve(
        dfrx.ODETerm(sys_.network), dfrx.Kvaerno5(), t0=0.0, t1=float(ts[-1]), dt0=1e-6,
        y0=jnp.asarray(y0, dtype=jnp.float64), args=theta,
        saveat=dfrx.SaveAt(ts=jnp.asarray(ts, dtype=jnp.float64)),
        stepsize_controller=dfrx.PIDController(
            rtol=ctrl["rtol"], atol=ctrl["atol"], pcoeff=ctrl["pcoeff"],
            icoeff=ctrl["icoeff"], dcoeff=ctrl["dcoeff"]),
        max_steps=HARD_CAP, throw=False)
    if not bool(sol.result == dfrx.RESULTS.successful):
        return None
    return np.asarray(sol.ys)[:, idx]


def solver_steps(sys_, theta, y0, ctrl, t_end):
    return int(dfrx.diffeqsolve(
        dfrx.ODETerm(sys_.network), dfrx.Kvaerno5(), t0=0.0, t1=float(t_end), dt0=1e-6,
        y0=jnp.asarray(y0, dtype=jnp.float64), args=theta,
        saveat=dfrx.SaveAt(t1=True),
        stepsize_controller=dfrx.PIDController(
            rtol=ctrl["rtol"], atol=ctrl["atol"], pcoeff=ctrl["pcoeff"],
            icoeff=ctrl["icoeff"], dcoeff=ctrl["dcoeff"]),
        max_steps=HARD_CAP, throw=False).stats["num_steps"])


def predict_tier1(sys_, theta, y0s, times, ctrl, idx, weights):
    """(C16-equivalent time series, [per-species endpoint per condition]); None where a
    solve failed. Endpoint row 0 is read off the time-series solve's last point."""
    base = solve(sys_, theta, y0s[0], ctrl, times, idx)
    ts_pred = None if base is None else base @ weights
    ep = [None if base is None else base[-1]]
    for y0 in y0s[1:]:
        p = solve(sys_, theta, y0, ctrl, [times[-1]], idx)
        ep.append(None if p is None else p[0])
    return ts_pred, ep


def norm_logp(obs, pred, sigma):
    return float(np.sum(-0.5 * ((obs - pred) / sigma) ** 2
                        - np.log(sigma) - 0.5 * np.log(2 * np.pi)))


def bounds_at(mults, lp, delta):
    ok = np.isfinite(lp)
    m, v = np.asarray(mults)[ok], np.asarray(lp)[ok]
    if len(m) < 3:
        return None, None
    i = int(np.argmax(v)); target = v[i] - delta

    def cross(rng):
        pm_, pv = m[i], v[i]
        for j in rng:
            if v[j] <= target:
                if pv == v[j]:
                    return m[j]
                f = (pv - target) / (pv - v[j])
                return 10 ** (np.log10(pm_) + f * (np.log10(m[j]) - np.log10(pm_)))
            pm_, pv = m[j], v[j]
        return None
    return cross(range(i - 1, -1, -1)), cross(range(i + 1, len(m)))


def profile_grid():
    grid = np.unique(np.concatenate([np.linspace(-1.0, -0.06, 12),
                                     np.linspace(-0.06, 0.06, 25),
                                     np.linspace(0.06, 1.0, 12)]))
    return 10.0 ** grid


def width_from_fisher(F, free, groups, prior=True):
    """Full width, log10, of each free group's interval on the profile's scale."""
    ix = [groups.index(g) for g in free]
    Fs = F[np.ix_(ix, ix)].copy()
    if prior:
        Fs += np.eye(len(ix)) / PRIOR_SD_LN ** 2
    sd = np.sqrt(np.diag(np.linalg.pinv(Fs)))
    return 2 * SD_MULT * sd / np.log(10.0)


def legacy_check(system, param):
    sys_, targets, ctrl = build(system)
    sg = {k: float(v) for k, v in json.loads(config_path(system).read_text())["scaling_groups"].items()}
    idx = [sys_.index_of[t] for t in targets]
    y0s, ep_obs, times, ts_obs, sig_ep, sig_ts = load_data(sys_, system, targets)
    mults = profile_grid()
    lp = []
    for mm in mults:
        o = dict(sg); o[param] = group_value_for(param, mm)
        th = set_scaling_group_values(sys_.theta, sys_.params, o)
        tot, bad = 0.0, False
        for i, y0 in enumerate(y0s):
            p = solve(sys_, th, y0, ctrl, [150.0], idx)
            if p is None:
                bad = True; break
            tot += norm_logp(ep_obs[i], p[0], sig_ep)
        p = None if bad else solve(sys_, th, y0s[0], ctrl, times, idx)
        lp.append(np.nan if p is None else tot + norm_logp(ts_obs, p, sig_ts))
    lo, hi = bounds_at(mults, lp, DELTA)
    print(f"=== legacy design replay: {system}, {param}, all {len(y0s)} conditions + per-species time series ===")
    print(f"  profile interval [{lo:.4f}, {hi:.4f}]")
    if system in MEASURED_POSTERIOR and param == "a1":
        mlo, mhi = MEASURED_POSTERIOR[system]
        err = abs(np.log10(lo) - np.log10(mlo)) + abs(np.log10(hi) - np.log10(mhi))
        print(f"  measured posterior [{mlo:.4f}, {mhi:.4f}]  log10 discrepancy {err:.4f} -> "
              f"{'REPLICATION OK' if err < 0.05 else 'REPLICATION SUSPECT, do not trust Tier-1 tables'}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--system", default="C12")
    ap.add_argument("--params", default=DEFAULT_PARAMS,
                    help="groups to profile, and the free set for the joint/greedy analysis")
    ap.add_argument("--legacy", action="store_true", help="replay the old ladder design (self-check)")
    a = ap.parse_args()
    params = a.params.split(",")

    if a.legacy:
        for p in params:
            legacy_check(a.system, p)
        print("\nDONE")
        return

    sys_, targets, ctrl = build(a.system)
    sg = {k: float(v) for k, v in json.loads(config_path(a.system).read_text())["scaling_groups"].items()}
    groups = sorted(sg)
    missing = [p for p in params if p not in groups]
    if missing:
        raise SystemExit(f"{missing} are not scaling groups of {a.system}: {groups}")
    idx = [sys_.index_of[t] for t in targets]
    y0s, times, sig_ep, sig_ts, weights = load_tier1(sys_, a.system, targets)
    n_cond = len(y0s)

    def theta_with(group=None, mult=1.0):
        o = dict(sg)
        if group is not None:
            o[group] = group_value_for(group, mult)
        return set_scaling_group_values(sys_.theta, sys_.params, o)

    clean_ts, clean_ep = predict_tier1(sys_, theta_with(), y0s, times, ctrl, idx, weights)
    if clean_ts is None or any(p is None for p in clean_ep):
        raise RuntimeError("nominal solve failed")
    steps = [solver_steps(sys_, theta_with(), y, ctrl, times[-1]) for y in y0s]
    print(f"=== Tier-1 information vs endpoint conditions: {a.system} ===")
    print(f"    {n_cond} candidate conditions, time series {len(times)} points of C16 Equivalents; "
          f"solver steps per condition {steps}")

    # ---- joint: sensitivities of every output to every group, ln(multiplier) ----- #
    h_ln = FD_STEP * np.log(10.0)
    J_ts = np.full((len(times), len(groups)), np.nan)
    J_ep = np.full((n_cond, len(targets), len(groups)), np.nan)
    for g, group in enumerate(groups):
        tp, ep_p = predict_tier1(sys_, theta_with(group, 10 ** FD_STEP), y0s, times, ctrl, idx, weights)
        tm, ep_m = predict_tier1(sys_, theta_with(group, 10 ** -FD_STEP), y0s, times, ctrl, idx, weights)
        if tp is not None and tm is not None:
            J_ts[:, g] = (tp - tm) / (2 * h_ln)
        for i in range(n_cond):
            if ep_p[i] is not None and ep_m[i] is not None:
                J_ep[i, :, g] = (ep_p[i] - ep_m[i]) / (2 * h_ln)
    bad_groups = [groups[g] for g in range(len(groups))
                  if not (np.isfinite(J_ts[:, g]).all() and np.isfinite(J_ep[:, :, g]).all())]
    if bad_groups:
        print(f"    WARNING: a +/-{FD_STEP} log10 solve failed for {bad_groups}; their sensitivities are zeroed")
    J_ts = np.nan_to_num(J_ts); J_ep = np.nan_to_num(J_ep)
    F_ts = J_ts.T @ np.diag(1.0 / sig_ts ** 2) @ J_ts
    F_ep = np.stack([J_ep[i].T @ np.diag(1.0 / sig_ep[i] ** 2) @ J_ep[i] for i in range(n_cond)])

    def fisher(sel, use_ts=True):
        F = F_ts.copy() if use_ts else np.zeros_like(F_ts)
        for i in sel:
            F += F_ep[i]
        return F

    free_ix = [groups.index(p) for p in params]
    prior_P = np.eye(len(params)) / PRIOR_SD_LN ** 2

    def logdet_free(sel):
        F = fisher(sel)[np.ix_(free_ix, free_ix)] + prior_P
        return float(np.linalg.slogdet(F)[1])

    greedy = [0]
    while len(greedy) < n_cond:
        rest = [i for i in range(n_cond) if i not in greedy]
        greedy.append(max(rest, key=lambda j: logdet_free(greedy + [j])))
    orders = {"row": list(range(n_cond)),
              "cheap": [0] + sorted(range(1, n_cond), key=lambda i: steps[i]),
              "greedy": greedy}
    print(f"    orders: " + "  ".join(f"{k}={v}" for k, v in orders.items()))

    # ---- profile: per-condition log-likelihood on the grid, per parameter -------- #
    mults = profile_grid()
    profile_L, profile_T, profile_rows = {}, {}, []
    for p in params:
        L = np.full((len(mults), n_cond), np.nan); T = np.full(len(mults), np.nan)
        for k, mm in enumerate(mults):
            tsp, epp = predict_tier1(sys_, theta_with(p, mm), y0s, times, ctrl, idx, weights)
            if tsp is not None:
                T[k] = norm_logp(clean_ts, tsp, sig_ts)
            for i in range(n_cond):
                if epp[i] is not None:
                    L[k, i] = norm_logp(clean_ep[i], epp[i], sig_ep[i])
        profile_L[p], profile_T[p] = L, T
        for order_name, order in orders.items():
            for n in range(0, n_cond + 1):
                lp = T + (L[:, order[:n]].sum(axis=1) if n else 0.0)
                lo, hi = bounds_at(mults, lp, DELTA)
                width = None if lo is None or hi is None else float(np.log10(hi) - np.log10(lo))
                profile_rows.append(dict(param=p, order=order_name, n_endpoint=n, solves=max(n, 1),
                                         lo=lo, hi=hi, width=width))

    def fmt(w):
        return f"{w:7.4f}" if w is not None else "  unbnd"

    header = "".join(f"{n:>8}" for n in range(0, n_cond + 1))
    for order_name in ("greedy", "row", "cheap"):
        print(f"\n  PROFILE full width, log10, vs endpoint conditions [{order_name} order]  "
              f"(0 = time series only; solves = max(n, 1))")
        print(f"  {'param':<8}{header}")
        for p in params:
            ws = [r["width"] for r in profile_rows if r["param"] == p and r["order"] == order_name]
            print(f"  {p:<8}" + "".join(f" {fmt(w)}" for w in ws))

    print(f"\n  JOINT marginal full width, log10, with {params} all free, prior included "
          f"[greedy order]  (prior alone: {2 * SD_MULT * PRIOR_SD_LN / np.log(10):.2f})")
    print(f"  {'param':<8}{header}")
    joint_rows = []
    for n in range(0, n_cond + 1):
        sel = greedy[:n]
        joint_rows.append(dict(n_endpoint=n, widths=dict(zip(params, width_from_fisher(fisher(sel), params, groups).tolist()))))
    for p in params:
        print(f"  {p:<8}" + "".join(f" {fmt(r['widths'][p])}" for r in joint_rows))

    F_full = fisher(range(n_cond))
    print(f"\n  EVERY GROUP at the full design (all {n_cond} conditions + time series), full width log10")
    print(f"  {'group':<8}{'profile':>9}{'Fisher cond.':>14}{'marg, params free':>19}{'marg, all free':>16}")
    marg_params = dict(zip(params, width_from_fisher(F_full, params, groups)))
    marg_all = dict(zip(groups, width_from_fisher(F_full, groups, groups)))
    group_rows, check_ok = [], True
    for g in groups:
        w_cond = float(width_from_fisher(F_full, [g], groups, prior=False)[0])
        prof = next((r["width"] for r in profile_rows
                     if r["param"] == g and r["order"] == "row" and r["n_endpoint"] == n_cond), None)
        if prof is not None and w_cond < 1.0 and abs(w_cond / prof - 1.0) > 0.10:
            check_ok = False
        group_rows.append(dict(group=g, profile=prof, fisher_conditional=w_cond,
                               marginal_params_free=float(marg_params[g]) if g in marg_params else None,
                               marginal_all_free=float(marg_all[g])))
        print(f"  {g:<8}{fmt(prof) if prof is not None else '      -':>9}{w_cond:>14.4f}"
              f"{(f'{marg_params[g]:.4f}' if g in marg_params else '-'):>19}{marg_all[g]:>16.4f}")
    print(f"\n  SELF-CHECK: profile vs Fisher conditional width (identified params, within 10%) -> "
          f"{'SENSITIVITIES OK' if check_ok else 'MISMATCH, check FD step / solver noise before using JOINT'}")

    stem = HERE / f"info_tier1_{a.system}"
    np.savez(stem.with_suffix(".npz"), groups=np.array(groups), params=np.array(params), targets=np.array(targets),
             times=times, sig_ts=sig_ts, sig_ep=sig_ep, clean_ts=clean_ts, clean_ep=np.stack(clean_ep),
             J_ts=J_ts, J_ep=J_ep, steps=np.array(steps), mults=mults,
             **{f"profile_L_{p}": profile_L[p] for p in params},
             **{f"profile_T_{p}": profile_T[p] for p in params})
    stem.with_suffix(".json").write_text(json.dumps(dict(
        system=a.system, params=params, delta=DELTA, fd_step_log10=FD_STEP, steps=steps, orders=orders,
        profile=profile_rows, joint_greedy=joint_rows, groups=group_rows, self_check_ok=check_ok,
        bad_fd_groups=bad_groups), indent=2))
    print(f"\nWrote {stem}.json/.npz\nDONE")


if __name__ == "__main__":
    main()
