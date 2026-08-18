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


class ChainSystem:
    """One built network plus the solve/export operations that need it.

    The notebook kept network/species/sp/theta as module globals that each cell
    overwrote, which is why its own docstring warns you not to skip a cell. Binding
    them to an instance removes that hazard entirely.
    """

    def __init__(self, reactions_dir: Path, rtol: float, atol: float):
        out = build_ode_system_from_reactions(reactions_dir)
        self.network, self.species, self.params, param_values, scaling_groups = out
        self.sp = make_namespace(self.species)
        theta = jnp.array([param_values[p] for p in self.params], dtype=jnp.float64)
        self.theta = set_scaling_group_values(theta, self.params, {g: 1 for g in scaling_groups})
        self.rtol, self.atol = rtol, atol
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

    def solve(self, y0, max_steps: int, save_steps: bool = True):
        """Returns (times, concentrations), or (None, None) for a non-converged solve.

        throw=False plus an explicit result check: a silent failure here would be
        written into the training data as if it were a real trajectory.
        """
        sol = dfrx.diffeqsolve(
            dfrx.ODETerm(self.network), dfrx.Kvaerno5(),
            t0=TIME_RANGE[0], t1=TIME_RANGE[1], dt0=1e-6,
            y0=jnp.asarray(y0, dtype=jnp.float64), args=self.theta,
            saveat=dfrx.SaveAt(steps=True) if save_steps else dfrx.SaveAt(t1=True),
            stepsize_controller=dfrx.PIDController(
                rtol=self.rtol, atol=self.atol, pcoeff=0.2, icoeff=0.4, dcoeff=0),
            max_steps=max_steps, throw=False,
        )
        total = int(np.asarray(sol.stats["num_steps"]))
        if total >= max_steps or bool(sol.result != dfrx.RESULTS.successful):
            return None, None
        accepted = int(np.asarray(sol.stats["num_accepted_steps"]))
        if save_steps:
            return sol.ts[:accepted], sol.ys[:accepted, :]
        return sol.ts, (sol.ys if sol.ys.ndim == 2 else sol.ys[None, :])


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


def make_sweep(sys_: ChainSystem, names: list[str],
               offsets=(-0.08, -0.04, 0.0, 0.04, 0.08)) -> dict[str, list[float]]:
    """Candidate initial-condition rows: a shared base factor per row, plus a small
    deterministic per-species offset so the multipliers are not uniform across species."""
    y0 = sys_.y0()
    sweep = {n: [] for n in names}
    for row, factor in enumerate(SWEEP_FACTORS):
        for j, name in enumerate(names):
            base = float(y0[sys_.index_of[name]])
            sweep[name].append(round_sigfigs(base * factor * (1.0 + offsets[(row + 2 * j) % len(offsets)])))
    return sweep


def export_sweep(sys_: ChainSystem, targets: list[str], out_dir: Path, max_steps: int,
                 n_required: int = 9, min_log_diff: float = 0.2):
    """One solve per candidate row, keeping rows whose OUTPUT differs from every kept
    row by >= min_log_diff decades. Without that filter the kept rows cluster in output
    space and the 9 conditions all probe the same dynamical regime."""
    sweep = make_sweep(sys_, SWEEP_SPECIES)
    idx = [sys_.index_of[n] for n in targets]
    y0_base = sys_.y0()

    rows, kept, n_bad, n_similar = [], [], 0, 0
    for combo in zip(*sweep.values()):
        y0 = y0_base.copy()
        for name, conc in zip(sweep, combo):
            y0[sys_.index_of[name]] = round_sigfigs(conc)
        _, C = sys_.solve(y0, max_steps, save_steps=False)
        if C is None:
            n_bad += 1
            continue
        out = [float(C[-1, i]) for i in idx]
        if kept and min(log_distance(out, p) for p in kept) < min_log_diff:
            n_similar += 1
            continue
        rows.append([round_sigfigs(c) for c in combo] + out)
        kept.append(out)
        if len(rows) >= n_required:
            break

    if len(rows) < n_required:
        raise RuntimeError(
            f"only {len(rows)}/{n_required} usable sweep rows "
            f"({n_bad} non-converged, {n_similar} too similar); "
            f"lower --min-log-diff or raise --sweep-max-steps")

    df = pd.DataFrame(rows, columns=[f"{n} (uM)" for n in sweep] + [f"{n} (uM)" for n in targets])
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "init_vs_final_conc.csv", index=False)
    return df, n_bad, n_similar


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
            sys_ = ChainSystem(rx_dir, a.rtol, a.atol)
            targets = sys_.targets(UNSAT_PATTERN if unsat else SAT_PATTERN)
            if not targets:
                raise RuntimeError("no C{n}_FA species in this network")
            ts = export_timeseries(sys_, targets, out_dir, a.ts_max_steps,
                                   min_observable=a.min_observable)
            sw, n_bad, n_sim = export_sweep(sys_, targets, out_dir, a.sweep_max_steps,
                                            min_log_diff=a.min_log_diff)
            print(f"  {name:<12} {len(sys_.species):>4} species  targets={targets}")
            print(f"               timeseries {ts.shape[0]} rows, sweep {sw.shape[0]} rows "
                  f"({n_bad} non-converged, {n_sim} too similar)  -> Data/{data_dir_name(cap, unsat)}/")
        except Exception as exc:
            print(f"  {name:<12} FAILED: {type(exc).__name__}: {exc}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
