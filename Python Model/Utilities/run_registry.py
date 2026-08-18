"""Where the scaling configs live, and whether a run directory finished.

Stdlib only, and deliberately so: this is imported by benchmark drivers that must
NOT pull in jax (``jax_enable_x64`` is a global fixed at import time, so every
measurement has to own a fresh interpreter), and it is called from bash via the
``__main__`` block at the bottom.

Both rules below were previously written out three times each -- in
``Bayesian Inference/compare_sampler_benchmarks.py``, ``submit_scaling_matrix.sh``
and ``run_scaling_matrix_nate.sh`` -- and had already drifted apart. They live here
now so there is one definition to fix when the layout changes again.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

# The two scaling ladders. The chain-length one is current; the enzyme one is kept
# for provenance only (see scaling_configs).
CHAIN_TESTS_DIR = "Chain Scaling Tests"
SCALING_TESTS_DIR = "GPU Scaling Tests"

# A finished run leaves both of these next to its solver_params.json.
TIMING_FILE = "timing.json"
DEFAULT_POSTERIOR_FILE = "posterior_samples_pm.nc"
CHECKPOINT_DIR = "checkpoint"
STATUS_FILE = "status.json"

# Benchmark run directories are named with this prefix precisely so they do NOT
# match the ``Test*`` glob that compare_sampler_benchmarks.py uses to find real
# scaling results -- otherwise a 100-draw benchmark would be silently averaged
# into the production comparison.
BENCHMARK_PREFIX = "bench__"


def project_root() -> Path:
    """The "Python Model" directory, anchored on this file rather than cwd.

    Callers run from a job scratch dir, from ssh with no cwd guarantee, and from
    SLURM with --export=NONE, so cwd is never trustworthy here.
    """
    return Path(__file__).resolve().parent.parent


def size_key(label: str) -> tuple:
    """Sort key approximating increasing network size, for cheapest-first sweeps.

    The old enzyme ladder relied on ALPHABETICAL order happening to equal increasing
    size. That silently breaks on the chain-length ladder the moment single-digit
    rungs exist: 'C10' sorts before 'C4'. So parse the rung instead of trusting the
    string.

    Saturated rungs sort before unsaturated ones. Within each family the ordering is
    exact (sat 59..219 species, unsat 158..318); across families they overlap, so this
    is cheapest-first as a policy, not a strict size ordering. Sorting exactly would
    mean building every network, and this module must stay import-cheap -- it is
    imported by drivers that must never pull in jax.
    """
    m = re.search(r"\bC(\d+)", label)
    if not m:
        return (2, 0, 0, label)                     # unrecognised: sort last, stably
    return (1 if "unsat" in label else 0, int(m.group(1)), 0, label)


def scaling_configs(base: str = CHAIN_TESTS_DIR) -> list[Path]:
    """Config paths for one scaling ladder, cheapest-first.

    Mirrors the `find ... -mindepth 2 -maxdepth 2 -name solver_params.json` used by
    submit_scaling_matrix.sh and run_scaling_matrix_nate.sh, but ordered by size_key
    rather than by filename.

    Defaults to the CHAIN-LENGTH ladder. The enzyme ladder is retired as a scaling
    axis -- most of its state vector was structurally dead (78-81% in the small
    systems), so it varied deadness as much as size -- but its configs are kept for
    provenance and can still be listed with base=SCALING_TESTS_DIR.
    """
    root = project_root() / "Results" / base
    return sorted((p for p in root.glob("*/solver_params.json") if p.is_file()),
                  key=lambda p: size_key(config_label(p)))


def config_label(config_path: Path) -> str:
    """'Test FabD FabH FabG - a2' -- the label used in logs and result tables."""
    return Path(config_path).parent.name


def log_slug(label: str) -> str:
    """Label with spaces replaced, matching run_scaling_matrix_nate.sh's log naming."""
    return label.replace(" ", "_")


def all_test_configs(exclude_benchmarks: bool = True) -> list[Path]:
    """Every Test* config under Results/, optionally minus benchmark output.

    compare_sampler_benchmarks.py globs ``Results/**/Test*/solver_params.json``;
    the exclusion keeps generated benchmark runs out of the production comparison
    even if one is ever misnamed.
    """
    found = sorted((project_root() / "Results").glob("**/Test*/solver_params.json"))
    if not exclude_benchmarks:
        return found
    return [p for p in found if BENCHMARK_PREFIX not in str(p)]


def read_status(run_dir: Path) -> dict | None:
    """checkpoint/status.json, or None. Fields: phase, warmup_done, sampling_done,
    n_tune, n_draws, stopped_reason, n_invocations, is_done."""
    path = Path(run_dir) / CHECKPOINT_DIR / STATUS_FILE
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def read_timing(run_dir: Path) -> dict | None:
    """timing.json, or None. Only key currently written is posterior_sampling_sec,
    which is `sampler.run` time ONLY -- it excludes model build, JIT compile and
    finalize. Do not read it as total wall time."""
    try:
        return json.loads((Path(run_dir) / TIMING_FILE).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def run_state(run_dir: Path, posterior_file: str = DEFAULT_POSTERIOR_FILE) -> str:
    """One of: done | sampled_not_finalized | incomplete | no_output.

    The older three-state version collapsed the middle two, which hides the one
    failure mode worth catching: sampling completed but finalize crashed, so no
    netcdf was written. That is a real failure, whereas stopping early on a wall
    budget is not -- it just needs another segment.

    Never infer success from a process exit code: inference_runner.__main__ never
    calls sys.exit(), so it returns 0 even when the run did not finalize.
    """
    run_dir = Path(run_dir)
    finalized = (run_dir / TIMING_FILE).exists() and (run_dir / posterior_file).exists()
    if finalized:
        return "done"

    status = read_status(run_dir)
    if status is not None:
        # is_done is written by RunStatus.to_json_dict; fall back to phase for
        # checkpoints written by older versions.
        if status.get("is_done") or status.get("phase") == "done":
            return "sampled_not_finalized"
        return "incomplete"

    return "no_output"


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--state", metavar="RUN_DIR",
                    help="print run_state for a run directory and exit")
    ap.add_argument("--posterior-file", default=DEFAULT_POSTERIOR_FILE)
    ap.add_argument("--list-configs", action="store_true",
                    help="print the 7 scaling config paths, one per line")
    ap.add_argument("--list-labels", action="store_true",
                    help="print the 7 config labels, one per line")
    args = ap.parse_args()

    if args.state:
        print(run_state(args.state, args.posterior_file))
    elif args.list_configs:
        for c in scaling_configs():
            print(c)
    elif args.list_labels:
        for c in scaling_configs():
            print(config_label(c))
    else:
        ap.print_help()
