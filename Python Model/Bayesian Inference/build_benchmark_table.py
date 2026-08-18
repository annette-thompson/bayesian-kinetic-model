"""Merge the benchmark JSONLs into one table and project wall time for N draws.

Runs on the Mac after pulling Results/Benchmarks/ from both machines. Stdlib only.

    python "Bayesian Inference/build_benchmark_table.py" --draws 10000

The projection is deliberately explicit about what is measured and what is not.
Two terms caused trouble in practice and are handled accordingly:

* COMPILE is not a small correction. A 10-draw run on nate took 156 minutes of
  which only ~4.7 min was gradient work -- the rest was NUTS kernel compilation.
  So compile is a first-class measured term (Stage 2's warmup_compile_sec +
  sampling_compile_sec), never a multiplier on the probe's much smaller
  jit(vmap(value_and_grad)) compile. Cells with no Stage-2 measurement report
  their projection as None rather than guessing.

* LEAPFROGS PER DRAW must be the max over chains, not the mean: chains advance
  under vmap and NUTS's inner while_loop runs until the slowest finishes. Stage 2
  emits wall_leapfrog_per_draw for this.

Every projected row carries leapfrog_source so a borrowed or warmup-only figure
never reads as if it were measured for that cell.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "Utilities"))
from run_registry import project_root, run_state  # noqa: E402

BENCH = project_root() / "Results" / "Benchmarks"
# Finalize (posterior predictive + log-likelihood + LOO) re-solves the ODE per draw
# and sits OUTSIDE timing.json's posterior_sampling_sec. Measured at ~4 min for a
# 10-draw run; kept as a conservative fraction until enough Stage-3 cells exist to
# calibrate it directly (parent wall time minus posterior_sampling_sec).
FINALIZE_RESERVE = 0.25
STAGE3_MAX_HOURS = 20.0      # under Alpine's 24 h cap, with finalize outside it


def load_jsonl(pattern: str) -> list[dict]:
    rows: list[dict] = []
    for path in sorted(BENCH.glob(pattern)):
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue          # tolerate a torn last line from a killed job
    return rows


def cell_axes(r: dict) -> tuple:
    """The axes that define a distinct measurement, coarsest-last."""
    return (r.get("config") or r.get("config_label"), float(r.get("floor", 0.0)),
            int(r.get("precision", 64)), str(r.get("rtol", "1e-4")), str(r.get("atol", "1e-8")))


def leapfrog_index(rows: list[dict]) -> dict[tuple, dict]:
    """Leapfrog records keyed both finely and coarsely.

    L is a property of posterior geometry, so it is measured on the fastest device
    and borrowed across machines/devices -- recorded via leapfrog_source so that
    stays visible. Tolerance and precision DO change the geometry (a looser solve is
    a different log-posterior), so the fine key wins when present and the coarse
    (config, floor) key is only a fallback.
    """
    idx: dict[tuple, dict] = {}
    for r in rows:
        if not r.get("wall_leapfrog_per_draw"):
            continue
        idx[cell_axes(r)] = r
        idx.setdefault(cell_axes(r)[:2], r)
    return idx


def project_hours(s_sec: float, n_draws: int, n_tune: int, lf: dict | None) -> tuple[float | None, str]:
    """Wall hours for a run, or (None, reason) when a required term is unmeasured."""
    if s_sec is None:
        return None, "no timing"
    if lf is None:
        return None, "no leapfrog measurement"
    l_draw = lf.get("wall_leapfrog_per_draw")
    l_warm = lf.get("wall_leapfrog_per_warmup_step") or l_draw
    compile_sec = (lf.get("warmup_compile_sec") or 0.0) + (lf.get("sampling_compile_sec") or 0.0)
    if not l_draw:
        return None, "no leapfrog measurement"
    sampling = n_tune * l_warm * s_sec + n_draws * l_draw * s_sec
    total = (compile_sec + sampling) * (1.0 + FINALIZE_RESERVE)
    return total / 3600.0, lf.get("leapfrog_source", "measured")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--draws", type=int, default=10000, help="headline projection")
    ap.add_argument("--tune", type=int, default=1000)
    ap.add_argument("--stage3-draws", type=int, default=100)
    ap.add_argument("--stage3-tune", type=int, default=200)
    ap.add_argument("--max-hours", type=float, default=STAGE3_MAX_HOURS)
    a = ap.parse_args()

    timing = load_jsonl("timing__*.jsonl")
    lf_idx = leapfrog_index(load_jsonl("leapfrog*.jsonl"))
    if not timing:
        print(f"No timing rows under {BENCH}. Pull them from the machines first.")
        return 1

    out_rows = []
    for r in timing:
        # Rows written before the key-collision fix carry precision_probe's jax
        # device list in "device". cell_key was never clobbered, so recover from it.
        if isinstance(r.get("device"), list) and r.get("cell_key"):
            m, dev, fl, lab = r["cell_key"].split("|", 3)
            r["machine"], r["device"], r["floor"], r["config_label"] = m, dev, float(fl), lab
        s = (r.get("ms_per_grad_eval") or 0) / 1000.0 or None
        axes = cell_axes(r)
        lf = lf_idx.get(axes) or lf_idx.get(axes[:2])
        h_big, src = project_hours(s, a.draws, a.tune, lf)
        h_s3, _ = project_hours(s, a.stage3_draws, a.stage3_tune, lf)
        run_dir = r.get("stage3_run_dir")
        out_rows.append({
            "machine": r.get("machine"), "device": r.get("device"),
            "precision": r.get("precision", 64),
            "rtol": r.get("rtol", "1e-4"), "atol": r.get("atol", "1e-8"),
            "floor": r.get("floor"), "config": r.get("config_label"),
            "n_reactions": r.get("n_reactions"), "n_species": r.get("n_species"),
            "status": r.get("status"),
            "chains_finite": r.get("n_chains_finite"),
            "ms_per_eval": r.get("ms_per_grad_eval"),
            "probe_compile_sec": r.get("build_compile_sec"),
            f"hours_{a.draws}": None if h_big is None else round(h_big, 2),
            "hours_stage3": None if h_s3 is None else round(h_s3, 2),
            "stage3_eligible": (h_s3 is not None and h_s3 <= a.max_hours),
            "leapfrog_source": src,
            "stage3_state": run_state(run_dir) if run_dir else None,
        })

    out_rows.sort(key=lambda r: (r["n_species"] or 0, r["config"] or "", r["machine"] or "",
                                 r["device"] or "", int(r["precision"] or 64),
                                 str(r["rtol"]), str(r["atol"]), r["floor"] or 0))
    csv_path = BENCH / "benchmark_matrix.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out_rows[0].keys()))
        w.writeheader()
        w.writerows(out_rows)

    print(f"{len(out_rows)} cells -> {csv_path.relative_to(project_root())}\n")
    hdr = (f"{'system':<20}{'rx':>4}{'machine':>8}{'dev':>5}{'fp':>4}{'rtol':>8}{'atol':>8}"
           f"{'floor':>7}{'status':>10}{'fin':>5}{'ms/eval':>11}{'h_' + str(a.draws):>9}")
    print(hdr); print("-" * len(hdr))
    for r in out_rows:
        ms = f"{r['ms_per_eval']:,.0f}" if r["ms_per_eval"] else ("timeout" if r["status"] == "timeout" else "-")
        hb = f"{r['hours_' + str(a.draws)]:.1f}" if r[f"hours_{a.draws}"] is not None else "-"
        # chains_finite is a PREDICTOR, never a verdict: a vmapped-NaN cell has been
        # measured sampling cleanly with zero divergences.
        fin = r["chains_finite"] if r["chains_finite"] is not None else "-"
        print(f"{(r['config'] or '?')[:19]:<20}{r['n_reactions'] or 0:>4}{r['machine'] or '?':>8}"
              f"{r['device'] or '?':>5}{r['precision']:>4}{str(r['rtol']):>8}{str(r['atol']):>8}"
              f"{r['floor']:>7g}{(r['status'] or '?'):>10}{fin:>5}{ms:>11}{hb:>9}")

    if not lf_idx:
        print("\nNOTE: no Stage-2 leapfrog/compile measurements yet, so every projection is blank.")
        print("      ms/eval alone cannot give wall time -- a draw is a whole NUTS trajectory,")
        print("      and on this model compile has dominated short runs.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
