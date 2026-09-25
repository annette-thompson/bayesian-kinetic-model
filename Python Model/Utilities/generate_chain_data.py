"""Forward-simulate each chain-length system and export its training CSVs.

Replaces the nine hand-copied cells of ``Bayesian Inference/generate_scaling_test_data.ipynb``
for the chain-length ladder. The solve/export logic is carried over from that notebook
unchanged (``run_odes``, ``sweep_final_conc``, ``make_nonuniform_sweep``); what changes
is that the per-system part is a loop over one spec table instead of copy-paste, which
is how the 3-enzyme system ended up with a config declaring FabD/FabH/FabG = 0.05 while
its data had been generated at 0.01.

    python Utilities/generate_chain_data.py --systems C4
    python Utilities/generate_chain_data.py                # the whole ladder

Writes ``Data/Chain_C4/{time_vs_conc,init_vs_final_conc}.csv`` and so on.

INITIAL CONDITIONS ARE CONSTANT ACROSS RUNGS. On the old enzyme ladder the enzyme
concentrations were ramped up as enzymes were added (0.001 -> 1.0), so system size and
initial state varied together. Here only the chain cap varies, which is the entire point
of the axis -- so every rung uses the FullFAS initial conditions.
"""
from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

import numpy as np
import jax

jax.config.update("jax_enable_x64", True)   # must precede any jax array creation
import jax.numpy as jnp                     # noqa: E402
import diffrax as dfrx                      # noqa: E402
import pandas as pd                         # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from reaction_model_builder import (        # noqa: E402
    build_ode_system_from_reactions,
    discover_scaling_groups,
    make_namespace,
    set_scaling_group_values,
)
from generate_chain_systems import SAT_CAPS, UNSAT_CAPS, variant_dir, project_root  # noqa: E402

# From Test_FullFAS (notebook cell 14). Held fixed for every rung.
INITIAL_CONDITIONS = {
    "C3_MalCoA": 500.0, "C2_AcCoA": 500.0, "ACP": 10.0,
    "NADPH": 1000.0, "NADH": 1000.0,
    "FabD": 1.0, "FabH": 1.0, "FabG": 1.0, "FabZ": 1.0, "FabI": 1.0,
    "FabF": 1.0, "FabA": 1.0, "FabB": 1.0, "TesA": 10.0,
}
# Cofactors (NADPH/NADH) are deliberately NOT swept -- they are held fixed, matching
# every existing config's endpoint dataset.
SWEEP_SPECIES = ["FabD", "FabH", "FabG", "FabZ", "FabI", "TesA", "FabF", "FabA", "FabB",
                 "C3_MalCoA", "ACP", "C2_AcCoA"]

TIME_RANGE = (0, 720)
SAT_PATTERN = r"^C(\d+)_FA$"
UNSAT_PATTERN = r"^C(\d+)_FA(_unsat)?$"

_UP = np.geomspace(1.05, 100.0, 30)
_DOWN = np.geomspace(0.95, 0.01, 30)
SWEEP_FACTORS = tuple(float(f) for pair in zip(_UP, _DOWN) for f in pair)


def data_dir_name(cap: int, unsat: bool) -> str:
    return f"Chain_C{cap}_unsat" if unsat else f"Chain_C{cap}"


def round_sigfigs(value: float, sigfigs: int = 1) -> float:
    value = float(value)
    return 0.0 if value == 0.0 else float(f"{value:.{sigfigs}g}")


def log_distance(a, b) -> float:
    """Max per-species distance in log10 space.

    Concentrations span many orders of magnitude, so a plain relative difference is
    dominated by whichever species is largest; log space treats an order-of-magnitude
    shift in ANY tracked species as equally significant.
    """
    eps = 1e-300
    la = np.log10(np.abs(np.asarray(a, dtype=np.float64)) + eps)
    lb = np.log10(np.abs(np.asarray(b, dtype=np.float64)) + eps)
    return float(np.max(np.abs(la - lb)))


def relative_error(candidate, reference, floor: float = 1e-6) -> float:
    """Max relative error between two full-state vectors, masked to species where
    |reference| exceeds floor (avoids blowup from dividing by near-zero noise)."""
    candidate = np.asarray(candidate, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    mask = np.abs(reference) > floor
    if not mask.any():
        return float("nan")
    return float(np.max(np.abs(candidate[mask] - reference[mask]) / np.abs(reference[mask])))


def nominal_scaling_group_overrides(scaling_groups: list[str]) -> dict[str, float]:
    """The correct no-op value per scaling group: 1.0 for ordinary multiplicative
    groups, 0.0 for ``d``-prefixed groups (which enter ADDITIVELY inside an exp(), e.g.
    TesA's ``1/exp(12*d1+d2)`` -- nominal is d=0 giving exp(0)=1, NOT d=1). Verified
    against the hand-maintained notebook (ODE Runner/run_model.ipynb), which sets
    d1=0, d2=0 explicitly while every other group is 1. A previous blanket
    ``{g: 1 for g in scaling_groups}`` silently used d=1 everywhere, giving TesA's
    rate a ~440,000x error (1/exp(13) vs 1/exp(0)) -- caught only because C4_NoFB's
    baseline then failed to converge even within 200,000 steps at tight tolerance.
    """
    return {g: (0.0 if g.startswith("d") else 1.0) for g in scaling_groups}


class ChainSystem:
    """One built network plus the solve/export operations that need it.

    The notebook kept network/species/sp/theta as module globals that each cell
    overwrote, which is why its own docstring warns you not to skip a cell. Binding
    them to an instance removes that hazard entirely.
    """

    def __init__(self, reactions_dir: Path, rtol: float, atol: float,
                pcoeff: float = 0.2, icoeff: float = 0.4, dcoeff: float = 0.0,
                scaling_group_overrides: dict[str, float] | None = None):
        # Names are discovered first so the overrides can be validated BEFORE the
        # build; build_ode_system_from_reactions now refuses to invent a value for
        # any group, so it must be handed the complete explicit dict.
        discovered = discover_scaling_groups(reactions_dir)
        if scaling_group_overrides is None:
            raise ValueError(
                f"scaling_group_overrides must be provided explicitly (reactions_dir={reactions_dir} "
                f"has scaling_groups={sorted(discovered)}). A previous silent default of 1 for "
                "every group was wrong for 'd'-prefixed groups (additive inside exp(), nominal is 0) "
                "and produced a ~440,000x rate-constant error on TesA that went undetected for an "
                "entire session. Use nominal_scaling_group_overrides(scaling_groups) for the standard "
                "nominal parameterization, or pass an explicit dict for a deliberate non-nominal one.")
        missing = set(discovered) - set(scaling_group_overrides)
        extra = set(scaling_group_overrides) - set(discovered)
        if missing or extra:
            raise ValueError(
                f"scaling_group_overrides does not exactly match this network's scaling groups "
                f"(reactions_dir={reactions_dir}): missing={sorted(missing)}, extra={sorted(extra)}")

        out = build_ode_system_from_reactions(reactions_dir, scaling_group=scaling_group_overrides)
        self.network, self.species, self.params, param_values, scaling_groups = out
        self.sp = make_namespace(self.species)
        theta = jnp.array([param_values[p] for p in self.params], dtype=jnp.float64)
        self.theta = set_scaling_group_values(theta, self.params, scaling_group_overrides)
        self.rtol, self.atol = rtol, atol
        self.pcoeff, self.icoeff, self.dcoeff = pcoeff, icoeff, dcoeff
        self.index_of = {name: i for i, name in enumerate(self.species)}

    def y0(self) -> np.ndarray:
        y = np.zeros(len(self.species), dtype=np.float64)
        for name, conc in INITIAL_CONDITIONS.items():
            if name in self.index_of:          # every rung has all 9 enzymes, but stay safe
                y[self.index_of[name]] = conc
        return y

    def targets(self, pattern: str) -> list[str]:
        rx = re.compile(pattern)
        return [s for s in self.species if rx.fullmatch(s)]

    def solve(self, y0, max_steps: int, save_steps: bool = True, return_stats: bool = False):
        """Returns (times, concentrations), or (None, None) for a non-converged solve.
        With return_stats=True, returns a third element: the actual step count taken
        (even on a converged solve well under max_steps) -- lets a caller measure a
        reference condition's own cost, e.g. to size an adaptive per-candidate cap
        relative to it, rather than only checking against a fixed absolute ceiling.

        throw=False plus an explicit result check: a silent failure here would be
        written into the training data as if it were a real trajectory.
        """
        sol = dfrx.diffeqsolve(
            dfrx.ODETerm(self.network), dfrx.Kvaerno5(),
            t0=TIME_RANGE[0], t1=TIME_RANGE[1], dt0=1e-6,
            y0=jnp.asarray(y0, dtype=jnp.float64), args=self.theta,
            saveat=dfrx.SaveAt(steps=True) if save_steps else dfrx.SaveAt(t1=True),
            stepsize_controller=dfrx.PIDController(
                rtol=self.rtol, atol=self.atol, pcoeff=self.pcoeff, icoeff=self.icoeff, dcoeff=self.dcoeff),
            max_steps=max_steps, throw=False,
        )
        total = int(np.asarray(sol.stats["num_steps"]))
        if total >= max_steps or bool(sol.result != dfrx.RESULTS.successful):
            return (None, None, total) if return_stats else (None, None)
        accepted = int(np.asarray(sol.stats["num_accepted_steps"]))
        if save_steps:
            result = (sol.ts[:accepted], sol.ys[:accepted, :])
        else:
            result = (sol.ts, (sol.ys if sol.ys.ndim == 2 else sol.ys[None, :]))
        return (*result, total) if return_stats else result


def export_timeseries(sys_: ChainSystem, targets: list[str], out_dir: Path,
                      max_steps: int, n_points: int = 11, min_gap_seconds: float = 1.0,
                      min_observable: float = 0.0):
    T, C = sys_.solve(sys_.y0(), max_steps, save_steps=True)
    if T is None:
        raise RuntimeError(f"timeseries solve did not converge within {max_steps} steps")
    idx = [sys_.index_of[n] for n in targets]
    df = pd.DataFrame(np.column_stack([T, C[:, idx]]),
                      columns=["Time (s)"] + [f"{n} (uM)" for n in targets])

    # Thin to n_points, but never take two rows closer than min_gap_seconds: the
    # adaptive solver clusters steps near t=0, so a plain linspace over row INDEX
    # would spend most of the dataset on the first second.
    T_arr = np.asarray(T)
    keep = [0]
    for i in range(1, len(T_arr)):
        if T_arr[i] - T_arr[keep[-1]] >= min_gap_seconds:
            keep.append(i)
    keep = np.array(keep)
    picks = np.unique(np.round(np.linspace(0, len(keep) - 1, min(n_points, len(keep)))).astype(int))
    df = df.iloc[keep[picks]].reset_index(drop=True)

    if min_observable > 0.0:
        # Drop rows where every observable is below min_observable. Such a row asks the
        # fit to match a number the solver was never asked to resolve: at t=1e-6 the C4
        # observable is 1.76e-81, which is 73 ORDERS OF MAGNITUDE below atol=1e-8. The
        # step controller has no incentive to make it accurate, so its derivative with
        # respect to the parameters is unbounded noise -- and one such row stretches the
        # dataset's dynamic range from 4.3 decades to 76.6. Below float32's smallest
        # denormal (1.4e-45) it is not even representable. Set this at or near atol.
        cols = [f"{n} (uM)" for n in targets]
        before = len(df)
        df = df[(df[cols] >= min_observable).any(axis=1)].reset_index(drop=True)
        if len(df) < before:
            print(f"      dropped {before - len(df)} timeseries row(s) with all "
                  f"observables < {min_observable:g}")

    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "time_vs_conc.csv", index=False)
    return df


# One-at-a-time titration factors, in the order candidates are listed: nearest to
# baseline first, alternating down/up. Step-count ties in export_sweep_ranked's
# cheapest-first ranking break in this order, so milder perturbations win ties.
# 0.25-4x (0.75, 1.33, 0.5, 2, 0.25, 4) gave only 8 usable conditions on C8 and C12:
# most single-species changes that small move no fatty acid by the 0.2 decades the
# diversity rule needs.
TITRATION_FACTORS = (0.5, 2.0, 0.2, 5.0, 0.1, 10.0)


def make_titration_sweep(sys_: ChainSystem, names: list[str],
                         factors=TITRATION_FACTORS) -> tuple[dict[str, list[float]], list[str]]:
    """Candidate rows that change ONE species at a time, everything else at baseline.

    Unlike make_sweep, whose rows move every species by one shared factor (so its 60
    rows are effectively points on a single axis), each row here probes a different
    direction. Rows are ordered factor-first (all species at 0.75x, then all at 1.33x,
    ...). Values are kept to 3 significant figures -- make_sweep's 1 significant figure
    would round 1.33x of 1 uM back to exactly the baseline.

    Returns (sweep, labels): sweep maps species -> one value per row, as make_sweep;
    labels[i] names row i, e.g. "TesA x0.75".
    """
    names = [n for n in names if n in sys_.index_of]
    y0 = sys_.y0()
    base = {n: float(y0[sys_.index_of[n]]) for n in names}
    sweep = {n: [] for n in names}
    labels = []
    for factor in factors:
        for target in names:
            for n in names:
                sweep[n].append(round_sigfigs(base[n] * factor, 3) if n == target else base[n])
            labels.append(f"{target} x{factor:g}")
    return sweep, labels


def make_sweep(sys_: ChainSystem, names: list[str],
               offsets=(-0.08, -0.04, 0.0, 0.04, 0.08)) -> dict[str, list[float]]:
    """Candidate initial-condition rows: a shared base factor per row, plus a small
    deterministic per-species offset so the multipliers are not uniform across species.

    Names absent from this system are skipped rather than raising: not every rung
    carries all of SWEEP_SPECIES. C4_NoFB ("No FabF / No FabB") deliberately drops
    those two enzymes -- C4 *with* them fails via ACP sequestration, since at a
    4-carbon cap they bind ACP into complexes they can never productively resolve --
    so sweeping a species that is not in the network is meaningless, not an error.
    """
    missing = [n for n in names if n not in sys_.index_of]
    if missing:
        print(f"      make_sweep: skipping {len(missing)} species absent from this "
              f"system: {missing}")
    names = [n for n in names if n in sys_.index_of]
    y0 = sys_.y0()
    sweep = {n: [] for n in names}
    for row, factor in enumerate(SWEEP_FACTORS):
        for j, name in enumerate(names):
            base = float(y0[sys_.index_of[name]])
            sweep[name].append(round_sigfigs(base * factor * (1.0 + offsets[(row + 2 * j) % len(offsets)])))
    return sweep


def export_sweep(sys_: ChainSystem, targets: list[str], out_dir: Path, max_steps: int,
                 n_required: int = 9, min_log_diff: float = 0.2,
                 max_steps_relative_to_baseline: float | None = None,
                 strict_tolerance: tuple[float, float, float, float, float] | None = None,
                 strict_max_steps: int = 10_000,
                 strict_max_steps_relative_to_baseline: float | None = None,
                 prefer_fastest_at_strict: bool = False,
                 fastest_pool_multiplier: float = 2.0,
                 strict_max_steps_fallback_multiplier: float | None = None):
    """One solve per candidate row, keeping rows whose OUTPUT differs from every kept
    row by >= min_log_diff decades. Without that filter the kept rows cluster in output
    space and the 9 conditions all probe the same dynamical regime.

    ``max_steps`` is always the hard ceiling (also used to probe the baseline itself,
    so it must be generous enough for that to succeed). When
    ``max_steps_relative_to_baseline`` is set, the PER-CANDIDATE acceptance threshold
    instead becomes min(max_steps, ceil(factor * baseline_steps)) -- adaptive to how
    stiff this particular system's own unperturbed condition already is, rather than
    one fixed absolute cap applied identically across every rung of the chain-length
    ladder regardless of size. Leaving it None reproduces the original fixed-cap
    behavior exactly.

    ``strict_tolerance``, if given, is (pcoeff, icoeff, dcoeff, rtol, atol) for an
    EXTRA acceptance check: a row that already passed the checks above must ALSO
    converge within the strict-tolerance step cap at this (typically much tighter)
    setting. Found necessary empirically: rows accepted purely on loose-tolerance step
    count can be secretly near-pathological at tight tolerance (comparable
    candidate-tolerance step count and rejection count did NOT predict this), so cheap
    loose-tolerance checks alone cannot catch it -- only actually attempting the tight
    solve can. ``strict_max_steps`` is the absolute ceiling (also used to probe the
    baseline itself at strict tolerance). When ``strict_max_steps_relative_to_baseline``
    is set, the PER-CANDIDATE strict-tolerance cap instead becomes
    min(strict_max_steps, ceil(factor * baseline_strict_steps)) -- adaptive to how many
    steps THIS system's own baseline needs at strict tolerance, mirroring
    max_steps_relative_to_baseline's rationale exactly: a flat cap is either too loose
    for a small/cheap rung or too tight for a large/stiff one. Leaving it None uses
    strict_max_steps as a flat cap.

    ``prefer_fastest_at_strict``, when strict_tolerance is also set, changes selection
    from "first n_required candidates that pass, in sweep order" to "the n_required
    FASTEST-at-strict-tolerance candidates that pass" (diversity filter still applied,
    just in speed-sorted order instead of sweep order). Rationale: a candidate that
    solves fast even at tight tolerance has more margin from whatever stiff feature
    causes the pathology, so it should be more robust to the theta perturbations
    explored during actual MCMC sampling, not just at the nominal theta tested here.
    Costs more -- every surviving candidate gets the strict solve, not just enough to
    reach n_required. ``fastest_pool_multiplier`` bounds how far the search goes before
    sorting: stops once ceil(n_required * fastest_pool_multiplier) survivors are found
    (or the sweep factors run out), rather than exhaustively trying every factor.
    Necessary in practice, not just an optimization: SWEEP_FACTORS fans out to 100x and
    0.01x baseline, and empirically most of that extreme tail fails to converge at ANY
    tolerance -- exhaustively trying it can leave too few survivors to reach n_required
    at all, when the easy near-baseline factors (tried first, same order as always)
    would have been plenty.

    ``strict_max_steps_fallback_multiplier``, if set, must be >= strict_max_steps_relative_to_baseline
    and defines a second, looser threshold. Every candidate is solved at the LARGER
    (fallback) cap regardless -- no extra solves -- and classified into one of two
    tiers: PRIMARY if it converges within the tighter (relative_to_baseline) cap, or
    FALLBACK if it needs more than that but still converges within the looser cap.
    Selection fills n_required from PRIMARY survivors first (fastest first, as above);
    only if that pool is exhausted before reaching n_required does it fall back to
    FALLBACK survivors (also fastest first) to fill the remainder. Exists because the
    primary threshold can be too tight to find n_required diverse rows AT ALL for some
    candidate settings, even with the full sweep available -- found empirically on
    C20+unsat, where a 1.5x threshold left only 4-8 of the 9 required rows for some
    (PID, tolerance) candidates. Leaving this None means a stiff rejection is final.
    """
    sweep = make_sweep(sys_, SWEEP_SPECIES)
    idx = [sys_.index_of[n] for n in targets]
    y0_base = sys_.y0()

    effective_max_steps = max_steps
    if max_steps_relative_to_baseline is not None:
        _, _, baseline_steps = sys_.solve(y0_base, max_steps, save_steps=False, return_stats=True)
        if baseline_steps >= max_steps:
            raise RuntimeError(f"baseline condition itself did not converge within max_steps={max_steps}; "
                               "raise max_steps before using max_steps_relative_to_baseline")
        effective_max_steps = min(max_steps, max(1, int(np.ceil(max_steps_relative_to_baseline * baseline_steps))))
        print(f"      baseline solved in {baseline_steps} steps -> per-candidate cap set to "
              f"{effective_max_steps} ({max_steps_relative_to_baseline}x baseline, capped at {max_steps})")

    effective_strict_max_steps = strict_max_steps
    if strict_tolerance is not None and strict_max_steps_relative_to_baseline is not None:
        pcoeff, icoeff, dcoeff, s_rtol, s_atol = strict_tolerance
        baseline_sol = dfrx.diffeqsolve(
            dfrx.ODETerm(sys_.network), dfrx.Kvaerno5(),
            t0=TIME_RANGE[0], t1=TIME_RANGE[1], dt0=1e-6,
            y0=jnp.asarray(y0_base, dtype=jnp.float64), args=sys_.theta,
            saveat=dfrx.SaveAt(t1=True),
            stepsize_controller=dfrx.PIDController(
                rtol=s_rtol, atol=s_atol, pcoeff=pcoeff, icoeff=icoeff, dcoeff=dcoeff),
            max_steps=strict_max_steps, throw=False,
        )
        baseline_strict_steps = int(np.asarray(baseline_sol.stats["num_steps"]))
        if baseline_strict_steps >= strict_max_steps or bool(baseline_sol.result != dfrx.RESULTS.successful):
            raise RuntimeError(f"baseline condition itself did not converge within "
                               f"strict_max_steps={strict_max_steps} at strict tolerance; "
                               "raise strict_max_steps before using strict_max_steps_relative_to_baseline")
        effective_strict_max_steps = min(strict_max_steps,
                                         max(1, int(np.ceil(strict_max_steps_relative_to_baseline * baseline_strict_steps))))
        print(f"      baseline solved in {baseline_strict_steps} steps at strict tolerance -> per-candidate "
              f"strict cap set to {effective_strict_max_steps} "
              f"({strict_max_steps_relative_to_baseline}x baseline, capped at {strict_max_steps})")

        effective_strict_max_steps_fallback = effective_strict_max_steps
        if strict_max_steps_fallback_multiplier is not None:
            effective_strict_max_steps_fallback = min(
                strict_max_steps,
                max(effective_strict_max_steps,
                    int(np.ceil(strict_max_steps_fallback_multiplier * baseline_strict_steps))))
            print(f"      fallback strict cap set to {effective_strict_max_steps_fallback} "
                  f"({strict_max_steps_fallback_multiplier}x baseline, capped at {strict_max_steps}) -- used "
                  f"only to fill remaining rows if the primary cap alone doesn't reach n_required")
    else:
        effective_strict_max_steps_fallback = effective_strict_max_steps

    def _strict_check(y0):
        """Solves at the (possibly looser) fallback cap so a candidate needing more than
        the primary cap but less than the fallback one isn't truncated into a false
        rejection -- classified into a tier afterward from the step count alone."""
        pcoeff, icoeff, dcoeff, s_rtol, s_atol = strict_tolerance
        sol = dfrx.diffeqsolve(
            dfrx.ODETerm(sys_.network), dfrx.Kvaerno5(),
            t0=TIME_RANGE[0], t1=TIME_RANGE[1], dt0=1e-6,
            y0=jnp.asarray(y0, dtype=jnp.float64), args=sys_.theta,
            saveat=dfrx.SaveAt(t1=True),
            stepsize_controller=dfrx.PIDController(
                rtol=s_rtol, atol=s_atol, pcoeff=pcoeff, icoeff=icoeff, dcoeff=dcoeff),
            max_steps=effective_strict_max_steps_fallback, throw=False,
        )
        strict_steps = int(np.asarray(sol.stats["num_steps"]))
        ok = strict_steps < effective_strict_max_steps_fallback and bool(sol.result == dfrx.RESULTS.successful)
        passed_primary = ok and strict_steps < effective_strict_max_steps
        passed_fallback = ok
        return passed_primary, passed_fallback, strict_steps

    rows, kept, n_bad, n_similar, n_stiff = [], [], 0, 0, 0

    if strict_tolerance is not None and prefer_fastest_at_strict:
        pool_target = max(n_required, int(np.ceil(n_required * fastest_pool_multiplier)))
        primary_survivors = []   # (strict_steps, rounded_combo, out) within the primary cap
        fallback_survivors = []  # same, but only within the looser fallback cap
        for combo in zip(*sweep.values()):
            if len(primary_survivors) >= pool_target:
                break
            y0 = y0_base.copy()
            for name, conc in zip(sweep, combo):
                y0[sys_.index_of[name]] = round_sigfigs(conc)
            _, C = sys_.solve(y0, effective_max_steps, save_steps=False)
            if C is None:
                n_bad += 1
                continue
            out = [float(C[-1, i]) for i in idx]
            passed_primary, passed_fallback, strict_steps = _strict_check(y0)
            if not passed_fallback:
                n_stiff += 1
                continue
            entry = (strict_steps, [round_sigfigs(c) for c in combo], out)
            (primary_survivors if passed_primary else fallback_survivors).append(entry)

        primary_survivors.sort(key=lambda s: s[0])
        fallback_survivors.sort(key=lambda s: s[0])
        n_fallback_used = 0
        for pool, is_fallback in ((primary_survivors, False), (fallback_survivors, True)):
            for strict_steps, rounded_combo, out in pool:
                if len(rows) >= n_required:
                    break
                if kept and min(log_distance(out, p) for p in kept) < min_log_diff:
                    n_similar += 1
                    continue
                rows.append(rounded_combo + out)
                kept.append(out)
                if is_fallback:
                    n_fallback_used += 1
            if len(rows) >= n_required:
                break
        if n_fallback_used:
            print(f"      used {n_fallback_used} fallback-tier row(s) (passed the looser cap but not "
                  f"the primary one) to reach n_required")
    else:
        for combo in zip(*sweep.values()):
            y0 = y0_base.copy()
            for name, conc in zip(sweep, combo):
                y0[sys_.index_of[name]] = round_sigfigs(conc)
            _, C = sys_.solve(y0, effective_max_steps, save_steps=False)
            if C is None:
                n_bad += 1
                continue
            out = [float(C[-1, i]) for i in idx]
            if kept and min(log_distance(out, p) for p in kept) < min_log_diff:
                n_similar += 1
                continue
            if strict_tolerance is not None:
                _, passed_fallback, strict_steps = _strict_check(y0)
                if not passed_fallback:
                    n_stiff += 1
                    continue
            rows.append([round_sigfigs(c) for c in combo] + out)
            kept.append(out)
            if len(rows) >= n_required:
                break

    if len(rows) < n_required:
        raise RuntimeError(
            f"only {len(rows)}/{n_required} usable sweep rows "
            f"({n_bad} non-converged, {n_similar} too similar, {n_stiff} stiff at strict tolerance); "
            f"lower --min-log-diff or raise --sweep-max-steps")

    df = pd.DataFrame(rows, columns=[f"{n} (uM)" for n in sweep] + [f"{n} (uM)" for n in targets])
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "init_vs_final_conc.csv", index=False)
    return df, n_bad, n_similar, n_stiff


def export_sweep_ranked(sys_: ChainSystem, targets: list[str], out_dir: Path, max_steps: int,
                        min_log_diff: float = 0.2,
                        max_steps_relative_to_baseline: float = 1.5,
                        strict_rtol: float = 1e-10, strict_atol: float = 1e-12,
                        strict_probe_max_steps: int = 200_000,
                        strict_cap_override: int | None = None,
                        n_keep: int | None = None,
                        total_weights=None,
                        min_total_frac_of_baseline: float | None = None,
                        seed_diversity_with_baseline: bool = True,
                        candidates: tuple[dict[str, list[float]], list[str]] | None = None):
    """Three-phase sweep search, replacing export_sweep's interleaved loose+strict+
    diversity checking with a simpler, order-independent design: (1) exhaustively find
    every sweep candidate that converges at the candidate's OWN (loose) tolerance --
    no diversity filtering yet, (2) strict-tolerance-filter that whole survivor pool
    (self-referential reference, see below), (3) sort what's left by loose-tolerance
    steps ascending and THEN resolve diversity in that order, so when two candidates
    are too similar the cheaper one (by loose-tolerance steps) wins instead of
    whichever happened to come first in sweep-factor order. Finally keep the top
    ``n_keep`` (or all of them if None). No forced minimum count -- whatever survives
    is what gets reported, since forcing exactly N produced confusing all-or-nothing
    failures for some (PID, tolerance) candidates.

    The strict-tolerance reference is SELF-REFERENTIAL: the same PID as ``sys_`` itself,
    just at (strict_rtol, strict_atol). This answers "how much does this candidate's own
    practical tolerance distort the answer relative to its own PID at extreme precision,"
    rather than comparing every candidate against one externally-chosen reference PID.

    Both step caps are 1.5x-baseline (or whatever ``max_steps_relative_to_baseline`` is):
    ``loose_cap`` from the baseline's OWN step count at the candidate's tolerance,
    ``strict_cap`` from the baseline's OWN step count at the strict tolerance -- each
    computed once, then applied uniformly across all sweep candidates.

    ``strict_cap_override`` skips the strict baseline probe and uses the supplied cap
    directly. The probe is nondeterministic on heterogeneous GPU pools (see the note
    at step 2), so a cap measured for the same system in an earlier successful run is
    a valid substitute. Doing so leaves the baseline's own ``err_pct`` as None.

    ``seed_diversity_with_baseline`` puts the baseline's own output in the diversity
    filter before any candidate. Every caller prepends the baseline as row 0, and
    without the seed the filter never compared candidates against it: the first sweep
    factors (1.05x, 0.95x, small offsets) round back to exactly the baseline at one
    significant figure, so that candidate always passed and, being as expensive as the
    baseline, landed last -- a duplicate of row 0 in C14, C16+unsat, C20 and C20+unsat.

    ``min_total_frac_of_baseline`` with ``total_weights`` (one weight per target)
    rejects a candidate whose weighted total output, sum(w * out), is below that
    fraction of the baseline's -- e.g. C16 Equivalents >= 10% of baseline, so no
    condition asks the fit to match concentrations far below the measured range.

    ``candidates`` replaces the default make_sweep rows with a precomputed
    (sweep, labels) pair, e.g. make_titration_sweep's. Its values are used as given;
    make_sweep's are rounded to one significant figure as before. Kept rows carry
    their label.
    """
    if candidates is None:
        sweep = make_sweep(sys_, SWEEP_SPECIES)
        labels = [None] * len(next(iter(sweep.values())))
        _round = round_sigfigs
    else:
        sweep, labels = candidates
        _round = float
    idx = [sys_.index_of[n] for n in targets]
    y0_base = sys_.y0()
    if min_total_frac_of_baseline is not None:
        if total_weights is None or len(total_weights) != len(targets):
            raise ValueError("min_total_frac_of_baseline needs total_weights, one per target")
        total_weights = np.asarray(total_weights, dtype=np.float64)

    def _solve_full(y0, rtol, atol, cap):
        sol = dfrx.diffeqsolve(
            dfrx.ODETerm(sys_.network), dfrx.Kvaerno5(),
            t0=TIME_RANGE[0], t1=TIME_RANGE[1], dt0=1e-6,
            y0=jnp.asarray(y0, dtype=jnp.float64), args=sys_.theta,
            saveat=dfrx.SaveAt(t1=True),
            stepsize_controller=dfrx.PIDController(
                rtol=rtol, atol=atol, pcoeff=sys_.pcoeff, icoeff=sys_.icoeff, dcoeff=sys_.dcoeff),
            max_steps=cap, throw=False,
        )
        steps = int(np.asarray(sol.stats["num_steps"]))
        rejected = int(np.asarray(sol.stats["num_rejected_steps"]))
        ok = steps < cap and bool(sol.result == dfrx.RESULTS.successful)
        final = np.asarray(sol.ys[-1] if sol.ys.ndim == 2 else sol.ys)
        return final, steps, rejected, ok

    # Step 1: baseline at the candidate's own (loose) tolerance -> loose_cap.
    base_loose_final, base_loose_steps, base_loose_rejected, base_loose_ok = _solve_full(
        y0_base, sys_.rtol, sys_.atol, max_steps)
    if not base_loose_ok:
        raise RuntimeError(f"baseline did not converge at candidate tolerance within max_steps={max_steps}")
    loose_cap = max(1, int(np.ceil(max_steps_relative_to_baseline * base_loose_steps)))
    print(f"      baseline solved in {base_loose_steps} steps at candidate tolerance -> "
          f"per-candidate loose cap set to {loose_cap} ({max_steps_relative_to_baseline}x baseline)")

    # Step 2: baseline at the candidate's own PID + strict tolerance -> strict_cap,
    # and the baseline's own error (self-referential reference).
    #
    # strict_cap_override exists because this probe fails NONDETERMINISTICALLY on
    # Blanca: C6's baseline solved here in 2105 steps on 2026-09-03 and then blew
    # past a 200,000-step cap on 2026-09-04 at the same PID and tolerances, on
    # different GPU hardware. Since the only thing the probe contributes is a step
    # cap, a known-good cap measured earlier for the same system can be supplied
    # directly, skipping the solve. The cost is that the baseline's own err_pct is
    # then unknown (there is no strict baseline state to compare against), so it is
    # reported as None rather than guessed. Per-condition err_pct is unaffected --
    # those strict solves still run, using this cap.
    if strict_cap_override is not None:
        strict_cap = max(1, int(strict_cap_override))
        base_strict_final = None
        base_strict_steps = None
        print(f"      strict baseline probe SKIPPED -- strict cap set to {strict_cap} "
              f"from strict_cap_override (baseline err_pct will be unavailable)")
    else:
        base_strict_final, base_strict_steps, base_strict_rejected, base_strict_ok = _solve_full(
            y0_base, strict_rtol, strict_atol, strict_probe_max_steps)
        if not base_strict_ok:
            raise RuntimeError(f"baseline did not converge at strict tolerance within "
                               f"max_steps={strict_probe_max_steps}")
        strict_cap = max(1, int(np.ceil(max_steps_relative_to_baseline * base_strict_steps)))
        print(f"      baseline solved in {base_strict_steps} steps at strict tolerance (self-referential, "
              f"same PID) -> per-candidate strict cap set to {strict_cap} ({max_steps_relative_to_baseline}x baseline)")
    base_out = [float(base_loose_final[i]) for i in idx]
    base_total = None if min_total_frac_of_baseline is None else float(total_weights @ np.asarray(base_out))
    baseline_row = dict(
        condition="baseline", steps=base_loose_steps, rejected=base_loose_rejected, out=base_out,
        strict_steps=base_strict_steps, loose_cap=loose_cap, strict_cap=strict_cap,
        weighted_total=base_total,
        err_pct=(None if base_strict_final is None
                 else relative_error(base_loose_final, base_strict_final) * 100))

    # Step 3: exhaustive loose-tolerance search -- every sweep factor is tried, no
    # diversity filtering yet and no early stop at any N. Diversity is resolved LATER
    # (step 5), after strict-tolerance has already dropped some candidates, so that
    # when two candidates turn out to be too similar, the cheaper one (by steps) wins
    # instead of whichever happened to come first in sweep-factor order.
    survivors = []
    n_bad = 0
    n_low_total = 0
    # One entry per candidate: why it was dropped, or "kept"/"usable beyond n_keep". Candidates
    # that passed the step caps, weighted-total floor and strict check also carry their
    # endpoint outputs ("out") and strict-tolerance step count, so diversity can be re-tested
    # against any other set of conditions afterwards.
    candidate_log = []
    for label, combo in zip(labels, zip(*sweep.values())):
        y0 = y0_base.copy()
        for name, conc in zip(sweep, combo):
            y0[sys_.index_of[name]] = _round(conc)
        final, steps, rejected, ok = _solve_full(y0, sys_.rtol, sys_.atol, loose_cap)
        if not ok:
            n_bad += 1
            candidate_log.append(dict(label=label, result="over loose step cap / non-converged", steps=steps))
            continue
        out = [float(final[i]) for i in idx]
        total_frac = None if base_total is None else float(total_weights @ np.asarray(out)) / base_total
        if total_frac is not None and total_frac < min_total_frac_of_baseline:
            n_low_total += 1
            candidate_log.append(dict(label=label, result="below weighted-total floor", steps=steps,
                                      total_frac_of_baseline=total_frac))
            continue
        survivors.append(dict(y0=y0, combo=[_round(c) for c in combo], loose_final=final,
                              out=out, steps=steps, rejected=rejected, label=label, total_frac=total_frac))

    # Step 4: strict-tolerance filter over ALL loose-tolerance survivors (no diversity
    # filtering has happened yet, so every loose-converged candidate gets this check).
    passing = []
    n_stiff = 0
    for s in survivors:
        strict_final, strict_steps, strict_rejected, strict_ok = _solve_full(
            s["y0"], strict_rtol, strict_atol, strict_cap)
        if not strict_ok:
            n_stiff += 1
            candidate_log.append(dict(label=s["label"], result="stiff at strict tolerance", steps=s["steps"],
                                      strict_steps=strict_steps, total_frac_of_baseline=s["total_frac"]))
            continue
        err_pct = relative_error(s["loose_final"], strict_final) * 100
        passing.append(dict(combo=s["combo"], out=s["out"], steps=s["steps"], rejected=s["rejected"],
                            err_pct=err_pct, label=s["label"], total_frac=s["total_frac"],
                            strict_steps=strict_steps))

    # Step 5: rank by steps ascending, THEN resolve diversity in that order -- the
    # cheapest member of any too-similar cluster is processed first and claims that
    # output region, so any pricier candidate too close to it gets dropped instead.
    passing.sort(key=lambda r: r["steps"])
    diverse = []
    kept_outputs = [base_out] if seed_diversity_with_baseline else []
    n_similar = 0
    for r in passing:
        if kept_outputs and min(log_distance(r["out"], p) for p in kept_outputs) < min_log_diff:
            n_similar += 1
            candidate_log.append(dict(label=r["label"], result="too similar", steps=r["steps"],
                                      strict_steps=r["strict_steps"], out=r["out"],
                                      total_frac_of_baseline=r["total_frac"],
                                      nearest_log_distance=min(log_distance(r["out"], p) for p in kept_outputs)))
            continue
        kept_outputs.append(r["out"])
        diverse.append(r)

    n_found = len(diverse)
    kept = diverse if n_keep is None else diverse[:n_keep]
    for rank, r in enumerate(diverse):
        candidate_log.append(dict(label=r["label"], result="kept" if rank < len(kept) else "usable beyond n_keep",
                                  steps=r["steps"], strict_steps=r["strict_steps"], out=r["out"],
                                  total_frac_of_baseline=r["total_frac"]))

    rows = [r["combo"] + r["out"] for r in kept]
    df = pd.DataFrame(rows, columns=[f"{n} (uM)" for n in sweep] + [f"{n} (uM)" for n in targets])
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "init_vs_final_conc.csv", index=False)

    def _summary(rs):
        if not rs:
            return dict(n=0, max_steps=None, avg_steps=None, max_rejected=None, avg_rejected=None,
                       max_err_pct=None, avg_err_pct=None)
        steps_arr = np.array([r["steps"] for r in rs])
        rej_arr = np.array([r["rejected"] for r in rs])
        err_arr = np.array([r["err_pct"] for r in rs])
        return dict(n=len(rs), max_steps=int(steps_arr.max()), avg_steps=float(steps_arr.mean()),
                   max_rejected=int(rej_arr.max()), avg_rejected=float(rej_arr.mean()),
                   max_err_pct=float(err_arr.max()), avg_err_pct=float(err_arr.mean()))

    summary = _summary(kept)
    low_msg = "" if base_total is None else f", {n_low_total} below {min_total_frac_of_baseline:g}x baseline total"
    print(f"      found {n_found} usable condition(s) ({n_bad} non-converged{low_msg}, {n_similar} too similar, "
          f"{n_stiff} stiff at strict tolerance); kept {len(kept)}")
    print(f"      kept-set summary: max_steps={summary['max_steps']} avg_steps={summary['avg_steps']} "
          f"max_rejected={summary['max_rejected']} avg_rejected={summary['avg_rejected']} "
          f"max_err_pct={summary['max_err_pct']} avg_err_pct={summary['avg_err_pct']}")

    return dict(kept=kept, baseline=baseline_row, n_found=n_found, n_bad=n_bad, n_similar=n_similar,
               n_stiff=n_stiff, n_low_total=n_low_total, summary=summary, candidate_log=candidate_log)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--systems", default=None,
                    help="comma-separated rung names (e.g. C4,C12+unsat); default all")
    ap.add_argument("--rtol", type=float, default=1e-5)   # notebook's data-generation
    ap.add_argument("--atol", type=float, default=1e-8)   # tolerances, not the fit's
    ap.add_argument("--ts-max-steps", type=int, default=1500)
    ap.add_argument("--sweep-max-steps", type=int, default=300)
    ap.add_argument("--min-log-diff", type=float, default=0.2)
    ap.add_argument("--min-observable", type=float, default=0.0,
                    help="drop timeseries rows whose observables are all below this. "
                         "Default 0 reproduces the notebook exactly; set it at or near "
                         "atol to exclude points the solver cannot resolve.")
    a = ap.parse_args()

    rungs = [(c, False) for c in SAT_CAPS] + [(c, True) for c in UNSAT_CAPS]
    if a.systems:
        want = {s.strip() for s in a.systems.split(",")}
        rungs = [r for r in rungs if variant_dir(*r) in want]
        if not rungs:
            print(f"No rung matched {sorted(want)}", file=sys.stderr)
            return 1

    root = project_root()
    for cap, unsat in rungs:
        name = variant_dir(cap, unsat)
        rx_dir = root / "Reactions" / "EC_FAS_ME1" / name
        out_dir = root / "Data" / data_dir_name(cap, unsat)
        try:
            _scaling_groups = discover_scaling_groups(rx_dir)
            sys_ = ChainSystem(rx_dir, a.rtol, a.atol,
                              scaling_group_overrides=nominal_scaling_group_overrides(_scaling_groups))
            targets = sys_.targets(UNSAT_PATTERN if unsat else SAT_PATTERN)
            if not targets:
                raise RuntimeError("no C{n}_FA species in this network")
            ts = export_timeseries(sys_, targets, out_dir, a.ts_max_steps,
                                   min_observable=a.min_observable)
            sw, n_bad, n_sim, n_stiff = export_sweep(sys_, targets, out_dir, a.sweep_max_steps,
                                                      min_log_diff=a.min_log_diff)
            print(f"  {name:<12} {len(sys_.species):>4} species  targets={targets}")
            print(f"               timeseries {ts.shape[0]} rows, sweep {sw.shape[0]} rows "
                  f"({n_bad} non-converged, {n_sim} too similar)  -> Data/{data_dir_name(cap, unsat)}/")
        except Exception as exc:
            print(f"  {name:<12} FAILED: {type(exc).__name__}: {exc}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
