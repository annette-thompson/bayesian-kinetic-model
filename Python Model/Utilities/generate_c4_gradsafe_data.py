"""Generate a self-consistent, gradient-safe synthetic C4 dataset for speed testing.

The C4 reaction network is unchanged. The canonical `Data/Chain_C4/` data (low
enzyme/substrate concentrations, matching the FullFAS baseline used across the
whole chain-length ladder) triggers a singular linear solve in Kvaerno5's
implicit-step reverse-mode differentiation for any condition whose
characteristic concentration scale falls below ~30uM -- see the C4
speed-testing investigation for the diagnosis. The forward solve is always
fine; only the gradient breaks, so this is purely a numerical-conditioning
issue, not a modeling one.

This script reuses generate_chain_data.py's ChainSystem/export_timeseries/
sweep-candidate machinery unmodified, but (a) uses a rescaled initial-condition
baseline and (b) adds a direct gradient-finiteness check to sweep-row
selection (not just a scale proxy), so every row in the output is verified
gradient-safe at generation time rather than assumed so from a threshold.

Output is a NEW dataset directory (`Data/Chain_C4_gradsafe/`), not a
modification of `Data/Chain_C4/` or the shared `INITIAL_CONDITIONS` baseline
in generate_chain_data.py (which is used by every rung, not just C4) -- this
is a synthetic dataset for speed testing, not a replacement of real
experimental data.

Usage (from the "Python Model" directory):
    python Utilities/generate_c4_gradsafe_data.py
    python Utilities/generate_c4_gradsafe_data.py --baseline-scale 5.0 --n-rows 9
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import jax

jax.config.update("jax_enable_x64", True)  # must precede any jax array creation
import jax.numpy as jnp  # noqa: E402
import diffrax as dfrx  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import generate_chain_data as gcd  # noqa: E402
from generate_chain_systems import project_root  # noqa: E402

# Empirically observed: gradients are reliably finite once a condition's
# characteristic concentration scale is at/above ~30uM; below ~20uM they
# reliably fail. This is a proxy used only to pick a sensible starting
# baseline scale -- actual row selection below verifies gradient finiteness
# directly, not via this threshold.
SAFE_SCALE_HINT = 30.0
NATIVE_BASELINE_TESA = gcd.INITIAL_CONDITIONS["TesA"]  # 10.0


def rescaled_initial_conditions(scale: float) -> dict[str, float]:
    return {name: conc * scale for name, conc in gcd.INITIAL_CONDITIONS.items()}


def build_grad_check(sys_: gcd.ChainSystem, rtol: float, atol: float, max_steps: int,
                     free_param: str = "a2"):
    """Returns a function checking finiteness of d(solve)/d(free_param) at the
    nominal parameter value for a candidate y0. This mirrors what NUTS
    actually needs a gradient for -- the free kinetic parameter being
    inferred, not the (fixed, given) initial conditions -- using the same
    diagnostic that isolated this bug."""
    controller = dfrx.PIDController(rtol=rtol, atol=atol, pcoeff=0.2, icoeff=0.4, dcoeff=0.0)
    rhs = dfrx.ODETerm(sys_.network)
    solver = dfrx.Kvaerno5()  # the real, unmodified production solver
    saveat = dfrx.SaveAt(t1=True)
    param_idx = sys_.params.index(free_param)

    def solve(y0, free_val):
        theta = sys_.theta.at[param_idx].set(free_val)
        sol = dfrx.diffeqsolve(
            rhs, solver, t0=gcd.TIME_RANGE[0], t1=gcd.TIME_RANGE[1], dt0=1e-6,
            y0=y0, args=theta, saveat=saveat, stepsize_controller=controller,
            max_steps=max_steps, throw=False,
        )
        return jnp.sum(sol.ys)

    nominal_val = float(sys_.theta[param_idx])

    def grad_finite(y0_np: np.ndarray) -> bool:
        y0 = jnp.asarray(y0_np, dtype=jnp.float64)
        try:
            _, grad = jax.value_and_grad(lambda p: solve(y0, p))(nominal_val)
        except Exception:  # noqa: BLE001 - equinox raises a hard runtime error, not just NaN
            return False
        return bool(jnp.isfinite(grad))

    return grad_finite


def export_gradsafe_sweep(sys_: gcd.ChainSystem, targets: list[str], out_dir: Path,
                          max_steps: int, grad_finite, n_required: int = 9,
                          min_log_diff: float = 0.2) -> tuple[pd.DataFrame, int, int, int]:
    sweep = gcd.make_sweep(sys_, gcd.SWEEP_SPECIES)
    idx = [sys_.index_of[n] for n in targets]
    y0_base = sys_.y0()

    rows, kept, n_bad, n_similar, n_grad_bad = [], [], 0, 0, 0
    for combo in zip(*sweep.values()):
        y0 = y0_base.copy()
        for name, conc in zip(sweep, combo):
            y0[sys_.index_of[name]] = gcd.round_sigfigs(conc)
        _, C = sys_.solve(y0, max_steps, save_steps=False)
        if C is None:
            n_bad += 1
            continue
        out = [float(C[-1, i]) for i in idx]
        if kept and min(gcd.log_distance(out, p) for p in kept) < min_log_diff:
            n_similar += 1
            continue
        if not grad_finite(y0):
            n_grad_bad += 1
            continue
        rows.append([gcd.round_sigfigs(c) for c in combo] + out)
        kept.append(out)
        if len(rows) >= n_required:
            break

    if len(rows) < n_required:
        raise RuntimeError(
            f"only {len(rows)}/{n_required} usable gradient-safe sweep rows "
            f"({n_bad} non-converged, {n_similar} too similar, {n_grad_bad} non-finite gradient)"
        )

    df = pd.DataFrame(rows, columns=[f"{n} (uM)" for n in sweep] + [f"{n} (uM)" for n in targets])
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "init_vs_final_conc.csv", index=False)
    return df, n_bad, n_similar, n_grad_bad


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--baseline-scale", type=float, default=5.0,
                    help="multiplier on the FullFAS baseline (TesA 10uM -> 50uM at default), "
                         "picked comfortably above the ~30uM finite-gradient threshold")
    ap.add_argument("--rtol", type=float, default=1e-5)  # matches generate_chain_data.py's data-gen default
    ap.add_argument("--atol", type=float, default=1e-8)
    ap.add_argument("--grad-rtol", type=float, default=1e-3)  # matches the C4 inference config under test
    ap.add_argument("--grad-atol", type=float, default=1e-6)
    ap.add_argument("--grad-max-steps", type=int, default=20000)
    ap.add_argument("--ts-max-steps", type=int, default=1500)
    ap.add_argument("--sweep-max-steps", type=int, default=300)
    ap.add_argument("--min-log-diff", type=float, default=0.2)
    ap.add_argument("--n-rows", type=int, default=9)
    args = ap.parse_args()

    root = project_root()
    rx_dir = root / "Reactions" / "EC_FAS_ME1" / "C4"
    out_dir = root / "Data" / "Chain_C4_gradsafe"

    # Temporarily rescale the baseline for this run only -- does not touch the
    # shared INITIAL_CONDITIONS constant used by generate_chain_data.py for the
    # rest of the chain-length ladder.
    original_ic = gcd.INITIAL_CONDITIONS
    gcd.INITIAL_CONDITIONS = rescaled_initial_conditions(args.baseline_scale)
    try:
        sys_ = gcd.ChainSystem(rx_dir, args.rtol, args.atol)
        targets = sys_.targets(gcd.SAT_PATTERN)
        if not targets:
            raise RuntimeError("no C{n}_FA species in the C4 network")

        grad_finite = build_grad_check(sys_, args.grad_rtol, args.grad_atol, args.grad_max_steps)

        baseline_y0 = sys_.y0()
        if not grad_finite(baseline_y0):
            print(f"  baseline at {args.baseline_scale}x (TesA={gcd.INITIAL_CONDITIONS['TesA']:g}uM) "
                  "has a non-finite gradient -- raise --baseline-scale and retry.")
            return 1

        ts = gcd.export_timeseries(sys_, targets, out_dir, args.ts_max_steps)
        sw, n_bad, n_sim, n_grad_bad = export_gradsafe_sweep(
            sys_, targets, out_dir, args.sweep_max_steps, grad_finite,
            n_required=args.n_rows, min_log_diff=args.min_log_diff,
        )
        print(f"  C4 gradsafe  {len(sys_.species):>4} species  targets={targets}  "
              f"baseline_scale={args.baseline_scale}x")
        print(f"               timeseries {ts.shape[0]} rows (baseline gradient-checked, finite)")
        print(f"               sweep {sw.shape[0]} rows ({n_bad} non-converged, {n_sim} too similar, "
              f"{n_grad_bad} non-finite gradient, all kept rows gradient-verified) -> Data/Chain_C4_gradsafe/")
    finally:
        gcd.INITIAL_CONDITIONS = original_ic

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
