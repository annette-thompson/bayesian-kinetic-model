"""Generate synthetic "observed" data at a chosen non-default kinetic-parameter
truth, for Tier-1 simulation-based-calibration validation
(Outlines/Bayesian Framework/Outline_v2.md, Section 2.6).

Reuses generate_chain_data.ChainSystem/export_timeseries/export_sweep unmodified:
both export functions solve through ChainSystem.solve(), which reads sys_.theta
directly, so overwriting sys_.theta via set_scaling_group_values *before* calling
them produces a correct forward simulation at the chosen truth with zero changes
to either export function. TIME_RANGE/INITIAL_CONDITIONS are the same module
constants generate_chain_data.py itself uses, so the synthetic data lands on the
identical time grid / initial conditions as the real chain-ladder data -- no
"template" file is needed to match them.

Noise is injected here -- nothing in the framework does this. experiment_framework.py's
noise_model only sets the likelihood sigma; it never touches the data (confirmed by
reading its source: no RNG call anywhere in the noise-model code). The sigma actually
used to draw the noise is written into a dedicated "<observable> (uM)_sigma" column per
dataset, for use with "noise_model": "column" in solver_params.json, rather than left
for a noise model to recompute from the already-noisy data -- recomputing from noisy
data would decouple the assumed likelihood sigma from the true data-generating process
(the same class of problem already worked around for the real chain-ladder's LOO fix).

Usage:
    python Utilities/generate_synthetic_ground_truth.py \\
        --reactions_source "Reactions/EC_FAS_ME1/C8/FabD.yaml" \\
        --reactions_source "Reactions/EC_FAS_ME1/C8/FabH.yaml" \\
        --reactions_source "Reactions/EC_FAS_ME1/C8_variants/FabG_grouped.yaml" \\
        --reactions_source "Reactions/EC_FAS_ME1/C8_variants/FabI_grouped.yaml" \\
        --reactions_source "Reactions/EC_FAS_ME1/C8/FabZ.yaml" \\
        --reactions_source "Reactions/EC_FAS_ME1/C8/FabI.yaml" \\
        --reactions_source "Reactions/EC_FAS_ME1/C8/TesA.yaml" \\
        --reactions_source "Reactions/EC_FAS_ME1/C8/FabF.yaml" \\
        --reactions_source "Reactions/EC_FAS_ME1/C8/FabA.yaml" \\
        --reactions_source "Reactions/EC_FAS_ME1/C8/FabB.yaml" \\
        --scaling_group a2=3.0 --scaling_group g1=1.8 \\
        --scaling_group h1=1.0 --scaling_group h2=5.0 \\
        --out_dir "Data/Chain_C8_synthetic/base" \\
        --seed 0 --noise_frac 0.10
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import numpy as np
import pandas as pd
import jax

jax.config.update("jax_enable_x64", True)   # must precede any jax array creation

sys.path.insert(0, str(Path(__file__).resolve().parent))
import generate_chain_data as gcd  # noqa: E402
from reaction_model_builder import (  # noqa: E402
    build_ode_system_from_reactions, discover_scaling_groups, set_scaling_group_values,
)

# The real chain-ladder data was generated at (0, 150); the module default (0, 720)
# postdates it and would silently produce a different time grid / different values.
# See Utilities/generate_chain_data.py's own TIME_RANGE constant and this session's
# scratchpad regen_chain_data.py, which hit exactly this drift.
gcd.TIME_RANGE = (0, 150)


def _parse_scaling_group(spec: str) -> tuple[str, float]:
    name, _, value = spec.partition("=")
    if not _:
        raise argparse.ArgumentTypeError(f"expected NAME=VALUE, got {spec!r}")
    return name, float(value)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--reactions_source", action="append", default=[],
                        help="Reaction YAML file path; repeat once per file. Required unless "
                        "--from_clean_dir is given.")
    parser.add_argument("--from_clean_dir", default=None,
                        help="Skip the forward solve: take the clean values from this directory's "
                        "time_vs_conc.csv / init_vs_final_conc.csv (e.g. Data/Chain_C12, the "
                        "noise-free nominal-truth ladder data). Keeps the exact conditions, time "
                        "points and solver settings every ladder run was fitted to, so only the "
                        "observables and the noise differ. Nominal truth only: incompatible with "
                        "--scaling_group.")
    parser.add_argument("--scaling_group", action="append", default=[], type=_parse_scaling_group,
                        metavar="NAME=VALUE",
                        help="Ground-truth scaling-group override; repeat per group. "
                        "Groups not listed stay at their nominal (1.0) value.")
    parser.add_argument("--out_dir", required=True, help="Directory to write the synthetic CSVs + ground_truth.json into.")
    parser.add_argument("--seed", type=int, default=0, help="RNG seed for noise injection.")
    parser.add_argument("--noise_frac", type=float, default=0.10,
                        help="Relative noise fraction, matching the validated chain-ladder's noise level.")
    parser.add_argument("--noise_mode", choices=["relative_mean", "pointwise"], default="relative_mean",
                        help="relative_mean: one sigma per column = frac*mean(|clean|), matching the validated "
                        "chain-ladder data. pointwise: sigma_i = noise_floor + frac*|clean_i|, needed for "
                        "transient (non-monotone) observables like acyl-ACP intermediates, where a single "
                        "column-mean sigma destroys tail signal-to-noise on the decaying part of the trace.")
    parser.add_argument("--noise_floor", type=float, default=1e-8,
                        help="Floor added to pointwise sigma (ignored in relative_mean mode).")
    parser.add_argument("--min_observable", type=float, default=1e-8,
                        help="Drop timeseries rows entirely below this (same landmine the real ladder data hit).")
    parser.add_argument("--target_pattern", default=None,
                        help="Regex selecting which species become individual observable columns. "
                        "Defaults to generate_chain_data.SAT_PATTERN (fatty-acid products only, the "
                        "original behavior). Use this to target acyl-ACP intermediates instead of or "
                        "alongside the FA products, e.g. '^C(4|6|8|10|12)_(FA|AcACP)$'.")
    parser.add_argument("--aggregate_fa", action="store_true",
                        help="Replace the individual per-chain FA columns (matched by "
                        "generate_chain_data.SAT_PATTERN, regardless of --target_pattern) with a single "
                        "summed 'Total FA (uM)' column -- the cheapest possible measurement tier. Species "
                        "matched by --target_pattern that are NOT FA products (e.g. intermediates) are "
                        "kept as individual columns either way.")
    parser.add_argument("--c16_equiv_timeseries", action="store_true",
                        help="Replace the TIMESERIES FA columns with one carbon-weighted "
                        "'C16 Equivalents (uM)' column (sum of n/16 * [C{n}_FA], the FA_conc.py observable "
                        "of the same name). The endpoint dataset keeps its per-species columns.")
    parser.add_argument("--no_clip", action="store_true",
                        help="Leave noisy values below zero instead of clipping them to 0. The "
                        "likelihood is an unbounded Normal, so clipping biases every point whose "
                        "clean value is within a few sigma of zero -- data and likelihood then "
                        "disagree, which is exactly what a calibration test must not have.")
    parser.add_argument("--rtol", type=float, default=1e-5)
    parser.add_argument("--atol", type=float, default=1e-8)
    return parser.parse_args()


def _inject_noise(df, observable_cols: list[str], frac: float, rng: np.random.Generator,
                  mode: str = "relative_mean", floor: float = 1e-8, clip: bool = True):
    """Add noise in place; return the sigma (or pointwise-sigma params) used per column.

    relative_mean: one sigma = frac*mean(|clean|) for the whole column -- fine for
    monotone observables (FA rises through the whole trace) but wrong for a transient
    that decays several decades (a single column-mean sigma makes the decaying tail,
    where the information about degradation/downstream kinetics lives, contribute
    nothing). pointwise fixes this: sigma_i = floor + frac*|clean_i| tracks the signal
    itself, so the tail keeps a meaningful (if larger, relatively) noise floor instead
    of being swamped by noise sized for the peak.
    """
    sigmas: dict[str, object] = {}
    for col in observable_cols:
        clean = df[col].to_numpy(dtype=np.float64)
        if mode == "pointwise":
            sigma = floor + frac * np.abs(clean)
            sigmas[col] = {"mode": "pointwise", "floor": floor, "frac": frac}
        else:
            sigma = frac * float(np.mean(np.abs(clean)))
            sigmas[col] = sigma
        noisy = clean + rng.normal(0.0, sigma, size=clean.shape)
        df[col] = np.clip(noisy, 0.0, None) if clip else noisy
        df[f"{col}_sigma"] = sigma
    return sigmas


def _apply_aggregate_fa(df, targets: list[str]) -> list[str]:
    """Replace the individual FA columns (matched by UNSAT_PATTERN -- covers both plain
    C{n}_FA and the C{n}_FA_unsat branch -- whatever --target_pattern was) with one
    summed 'Total FA (uM)' column in place. Returns the observable-column list to
    noise-inject afterward (FA columns removed, "Total FA (uM)" appended if any FA
    species were present). Using SAT_PATTERN here would miss C{n}_FA_unsat entirely."""
    fa_rx = re.compile(gcd.UNSAT_PATTERN)
    fa_names = [n for n in targets if fa_rx.fullmatch(n)]
    non_fa_cols = [f"{n} (uM)" for n in targets if n not in fa_names]
    if not fa_names:
        return non_fa_cols
    fa_cols = [f"{n} (uM)" for n in fa_names]
    df["Total FA (uM)"] = df[fa_cols].sum(axis=1)
    df.drop(columns=fa_cols, inplace=True)
    return non_fa_cols + ["Total FA (uM)"]


def _load_fa_conc():
    """FA_conc.py, the calculation module the fit uses, so the C16-equivalent weights
    written into the data are the same function the model's observable evaluates."""
    path = gcd.project_root() / "Calculation Files" / "Full_FAS" / "FA_conc.py"
    spec = spec_from_file_location("FA_conc", path)
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _apply_c16_equiv(df, targets: list[str]) -> list[str]:
    """Replace the FA columns with one carbon-weighted C16-equivalent column in place;
    return the observable-column list to noise-inject afterward (as _apply_aggregate_fa)."""
    fa_conc = _load_fa_conc()
    fa_rx = re.compile(gcd.UNSAT_PATTERN)
    fa_names = [n for n in targets if fa_rx.fullmatch(n)]
    non_fa_cols = [f"{n} (uM)" for n in targets if n not in fa_names]
    if not fa_names:
        return non_fa_cols
    fa_cols = [f"{n} (uM)" for n in fa_names]
    weights = np.array([fa_conc.c16_equiv_weight(n) for n in fa_names])
    df[fa_conc.C16_EQUIV_NAME] = df[fa_cols].to_numpy(dtype=np.float64) @ weights
    df.drop(columns=fa_cols, inplace=True)
    return non_fa_cols + [fa_conc.C16_EQUIV_NAME]


def main() -> int:
    args = _parse_args()
    root = gcd.project_root()
    out_dir = root / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    overrides = dict(args.scaling_group)
    if args.from_clean_dir:
        if overrides:
            raise SystemExit("--from_clean_dir holds nominal-truth data; it cannot be combined with --scaling_group")
        return _main_from_clean(args, root, out_dir)
    if not args.reactions_source:
        raise SystemExit("--reactions_source is required unless --from_clean_dir is given")
    reaction_paths = [root / p for p in args.reactions_source]

    _scaling_groups = discover_scaling_groups(reaction_paths)
    sys_ = gcd.ChainSystem(reaction_paths, rtol=args.rtol, atol=args.atol,
                          scaling_group_overrides=gcd.nominal_scaling_group_overrides(_scaling_groups))
    if overrides:
        sys_.theta = set_scaling_group_values(sys_.theta, sys_.params, overrides)

    target_pattern = args.target_pattern or gcd.SAT_PATTERN
    targets = sys_.targets(target_pattern)
    if not targets:
        raise RuntimeError(f"--target_pattern {target_pattern!r} matched no species in this network")
    gcd.SWEEP_SPECIES = [n for n in gcd.SWEEP_SPECIES if n in sys_.index_of]

    # NOTE: min_observable drops a row when EVERY observable is below the floor. For an
    # FA-only dataset that only trims t=0. For an intermediates-ONLY dataset (no FA
    # column to keep a row alive), the same filter can trim the late, decaying part of
    # a transient's trace -- exactly where its downstream-kinetics information lives.
    # Pass --min_observable 0 when generating intermediates-only data.
    ts_df = gcd.export_timeseries(sys_, targets, out_dir, max_steps=1500, min_observable=args.min_observable)
    sw_df, n_bad, n_similar, n_stiff = gcd.export_sweep(sys_, targets, out_dir, max_steps=300, min_log_diff=0.2)

    sweep_dropped = {"non_converged": n_bad, "too_similar": n_similar, "stiff_at_strict_tolerance": n_stiff}
    ts_sigmas, sw_sigmas = _finish(args, out_dir, ts_df, sw_df, targets, target_pattern, overrides,
                                   extra={"reactions_source": args.reactions_source,
                                          "sweep_dropped": sweep_dropped})

    print(f"Wrote synthetic data to {out_dir}")
    print(f"Ground truth: {overrides}")
    print(f"Timeseries: {ts_df.shape[0]} rows, sigma={ts_sigmas}")
    print(f"Sweep: {sw_df.shape[0]} rows, sigma={sw_sigmas} ({n_bad} non-converged, {n_similar} too similar)")
    return 0


def _main_from_clean(args, root: Path, out_dir: Path) -> int:
    clean_dir = root / args.from_clean_dir
    ts_df = pd.read_csv(clean_dir / "time_vs_conc.csv")
    sw_df = pd.read_csv(clean_dir / "init_vs_final_conc.csv")
    # Every FA column the clean files carry, unsaturated included -- SAT_PATTERN, the
    # solve path's default, would silently drop the _unsat species of a +unsat system.
    target_pattern = args.target_pattern or gcd.UNSAT_PATTERN
    rx = re.compile(target_pattern)
    targets = [c[:-len(" (uM)")] for c in ts_df.columns
               if c.endswith(" (uM)") and rx.fullmatch(c[:-len(" (uM)")])]
    if not targets:
        raise RuntimeError(f"no column of {clean_dir / 'time_vs_conc.csv'} matches {target_pattern!r}")
    missing = [f"{n} (uM)" for n in targets if f"{n} (uM)" not in sw_df.columns]
    if missing:
        raise RuntimeError(f"endpoint file lacks timeseries observables {missing}")
    ts_sigmas, sw_sigmas = _finish(args, out_dir, ts_df, sw_df, targets, target_pattern, {},
                                   extra={"source_clean_dir": args.from_clean_dir})
    print(f"Wrote synthetic data to {out_dir} (clean values from {args.from_clean_dir})")
    print(f"Timeseries: {ts_df.shape[0]} rows, sigma={ts_sigmas}")
    print(f"Sweep: {sw_df.shape[0]} rows, sigma={sw_sigmas}")
    return 0


def _finish(args, out_dir: Path, ts_df, sw_df, targets, target_pattern, overrides, extra):
    """Shared by both data sources: reshape the observables, inject noise, write the
    CSVs and ground_truth.json."""
    rng = np.random.default_rng(args.seed)
    if args.aggregate_fa:
        ts_cols = _apply_aggregate_fa(ts_df, targets)
        sw_cols = _apply_aggregate_fa(sw_df, targets)
    elif args.c16_equiv_timeseries:
        ts_cols = _apply_c16_equiv(ts_df, targets)
        sw_cols = [f"{n} (uM)" for n in targets]
    else:
        ts_cols = sw_cols = [f"{n} (uM)" for n in targets]
    clip = not args.no_clip
    ts_sigmas = _inject_noise(ts_df, ts_cols, args.noise_frac, rng, mode=args.noise_mode,
                              floor=args.noise_floor, clip=clip)
    sw_sigmas = _inject_noise(sw_df, sw_cols, args.noise_frac, rng, mode=args.noise_mode,
                              floor=args.noise_floor, clip=clip)

    ts_df.to_csv(out_dir / "time_vs_conc.csv", index=False)
    sw_df.to_csv(out_dir / "init_vs_final_conc.csv", index=False)

    ground_truth = {
        **extra,
        "scaling_group_overrides": overrides,
        "seed": args.seed,
        "noise_frac": args.noise_frac,
        "noise_mode": args.noise_mode,
        "noise_floor": args.noise_floor if args.noise_mode == "pointwise" else None,
        "clip_negative": clip,
        "target_pattern": target_pattern,
        "aggregate_fa": args.aggregate_fa,
        "c16_equiv_timeseries": args.c16_equiv_timeseries,
        "timeseries_observables": ts_cols,
        "sweep_observables": sw_cols,
        "timeseries_sigma": ts_sigmas,
        "sweep_sigma": sw_sigmas,
    }
    with open(out_dir / "ground_truth.json", "w", encoding="utf-8") as fh:
        json.dump(ground_truth, fh, indent=2)
    return ts_sigmas, sw_sigmas


if __name__ == "__main__":
    raise SystemExit(main())
