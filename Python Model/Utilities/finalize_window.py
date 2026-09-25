"""Retroactively choose the tuning/posterior boundary for a resumable run.

The resumable sampler persists *every* draw -- the full warmup ++ sampling
timeline per chain -- so the burn-in boundary is a slicing decision made after
the fact, never a re-run. This regenerates the standard posterior netcdf from a
chosen window without resampling.

NUTS targets the posterior under any fixed step size / mass matrix, and window
adaptation changes them only between draws, so the whole warmup++sampling
timeline is a valid (inhomogeneous) chain with the posterior as its stationary
distribution. Discarding an initial burn-in from that timeline is therefore
legitimate -- which is exactly why keeping warmup lets you move the boundary.

Usage (from the "Python Model" directory):
    # default: discard all warmup (== what the run auto-finalized to)
    python Utilities/finalize_window.py --solver_params_file "Results/<run>/solver_params.json"

    # keep the last 100 warmup draws as posterior, and thin by 2
    python Utilities/finalize_window.py --solver_params_file ... --burn_in 200 --thin 2

    # cap the END of the window too -- e.g. a run that kept sampling well past
    # its own true convergence point (a stranded chain can delay the live
    # detector catching up to excluding it): full-timeline index, same
    # convention as --burn_in, so "stop at sampling draw 400" on a run with
    # n_tune=1000 is --end_draw 1400
    python Utilities/finalize_window.py --solver_params_file ... --end_draw 1400

    # just report what's on disk without rewriting anything
    python Utilities/finalize_window.py --solver_params_file ... --list
"""
from __future__ import annotations

import argparse
import json

import numpy as np

from inference_runner import (
    _build_model_bundle,
    _finalize_resumable_run,
    import_solver_params,
)
from resumable_sampler import ResumableSampler, load_draws, load_stats


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--solver_params_file", required=True, help="Path to the run's solver_params.json/yaml.")
    parser.add_argument(
        "--burn_in",
        type=int,
        default=None,
        help="Draws to discard from the START of the full (warmup++sampling) per-chain "
        "timeline. Default = n_tune + posterior_burn_in_draws (config), i.e. reproduces "
        "exactly what the live run auto-finalized to.",
    )
    parser.add_argument("--thin", type=int, default=1, help="Keep every THIN-th draw after burn-in (default 1).")
    parser.add_argument(
        "--end_draw",
        type=int,
        default=None,
        help="Discard draws from the END of the full (warmup++sampling) per-chain timeline, "
        "same index convention as --burn_in (i.e. an absolute full-timeline index, not a "
        "count of sampling draws). Default = keep through the last available draw.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="Report the available warmup/sampling draw counts and exit without rewriting.",
    )
    parser.add_argument(
        "--include_stranded",
        action="store_true",
        help="Do NOT exclude the chains status.json recorded as stranded (low-posterior-basin "
        "outliers the live r-hat/ESS check excluded). Default is to exclude them, matching "
        "what the live run actually converged on -- pass this only to deliberately inspect "
        "the unfiltered, all-chains artifact.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    imported = import_solver_params(args.solver_params_file)
    checkpoint_dir = imported.results_save_dir / "checkpoint"
    posterior_config = imported.solver_params.get("posterior_sampling", {})
    n_tune = int(posterior_config.get("tune", 0))
    posterior_burn_in_draws = int(posterior_config.get("posterior_burn_in_draws", 0))

    warm = load_draws(checkpoint_dir, "warmup")
    samp = load_draws(checkpoint_dir, "sampling")
    if not samp:
        raise SystemExit(
            f"No sampling draws found under {checkpoint_dir / ResumableSampler.DRAWS}. "
            "Has the run produced posterior draws yet?"
        )
    warm_stats = load_stats(checkpoint_dir, "warmup")
    samp_stats = load_stats(checkpoint_dir, "sampling")

    any_var = next(iter(samp))
    n_chains, n_warm = warm[any_var].shape[:2] if warm else (samp[any_var].shape[0], 0)
    n_samp = samp[any_var].shape[1]
    total = n_warm + n_samp

    print(f"Run: {imported.results_save_dir}")
    print(f"  chains={n_chains}  warmup draws={n_warm}  sampling draws={n_samp}  total per chain={total}")

    burn_in = (n_tune + posterior_burn_in_draws) if args.burn_in is None else args.burn_in
    end_draw = args.end_draw
    if args.list:
        stop = total if end_draw is None else min(end_draw, total)
        kept = len(range(burn_in, stop, args.thin))
        print(f"  with --burn_in {burn_in} --end_draw {end_draw} --thin {args.thin} -> "
              f"{kept} posterior draws/chain ({n_chains * kept} total). (not written; drop --list to write)")
        return

    if not 0 <= burn_in < total:
        raise SystemExit(f"--burn_in {burn_in} out of range for a timeline of {total} draws/chain.")
    if end_draw is not None and not burn_in < end_draw <= total:
        raise SystemExit(f"--end_draw {end_draw} must be > --burn_in ({burn_in}) and <= {total}.")

    # Concatenate the full timeline, then slice the chosen window.
    def _concat(w, s):
        return {name: np.concatenate([w[name], s[name]], axis=1) if w else s[name] for name in s}

    full = _concat(warm, samp)
    full_stats = _concat(warm_stats, samp_stats)
    posterior_window = (slice(None), slice(burn_in, end_draw, args.thin))
    sel = {name: arr[posterior_window] for name, arr in full.items()}
    sel_stats = {name: arr[posterior_window] for name, arr in full_stats.items()}
    # Draws before the cut become the warmup_posterior group (what you discarded).
    discarded = {name: arr[:, :burn_in] for name, arr in full.items()} if burn_in > 0 else None

    kept = sel[any_var].shape[1]
    print(f"  window: burn_in={burn_in} thin={args.thin} -> {kept} posterior draws/chain "
          f"({n_chains * kept} total)")
    if kept < 50:
        print("  WARNING: very few draws kept; summaries/LOO will be noisy.")

    stranded_chains = None
    status_file = checkpoint_dir / "status.json"
    if not args.include_stranded and status_file.exists():
        try:
            stranded_chains = json.loads(status_file.read_text()).get("stranded_chains") or None
        except (json.JSONDecodeError, OSError):
            stranded_chains = None
    if stranded_chains:
        print(f"  status.json records stranded chain(s) {stranded_chains} -- excluding "
              "(pass --include_stranded to keep them)")
    elif args.include_stranded:
        print("  --include_stranded: keeping every chain regardless of status.json")

    bundle = _build_model_bundle(imported)
    _finalize_resumable_run(
        imported=imported, bundle=bundle, draws=sel, stats=sel_stats,
        segment_seconds=0.0, warmup_draws=discarded, stranded_chains=stranded_chains,
    )
    print("\nRewrote posterior netcdf for the chosen window.")


if __name__ == "__main__":
    main()
