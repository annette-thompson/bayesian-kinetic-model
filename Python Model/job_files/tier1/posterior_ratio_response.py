"""Robustness of the ratiometric strategy across posterior draws, for Fig 8 (outline 3.5).

Mains et al. 2022 predicted, and confirmed in vivo, that raising FabF/FabB relative to TesA
lengthens the fatty acids and lowering it shortens them. This asks whether that holds at every
parameter set the data find plausible, or only at the point it was derived from.

Two readouts, each at n posterior draws and at the point estimate (every scaling group at its
no-op value), from the fatty acids at 720 s:

  sweep     the elongation-to-termination ratio R = (FabF and FabB) / TesA, relative to the
            Tier-1 baseline, stepped over [1/span, span]. FabF and FabB are scaled by sqrt(R) and
            TesA by 1/sqrt(R), so the three enzymes' geometric mean stays fixed. Per draw: the
            average-chain-length curve, its least-squares slope against log10 R, and whether it
            rises monotonically.
  optimum   a grid over the FabF, FabB and TesA multipliers (each in [1/span, span], --grid
            points per axis). Per draw: the setting with the longest and the shortest average
            chain length, and its ratio (FabF + FabB) / TesA relative to baseline. A grid rather
            than an optimiser, so the answer cannot depend on a starting point.

A truncated system spans a narrower chain-length range than the in vivo demonstration, so
expect at most an attenuated version of the shift at Tier 1 (outline 3.5): the question here
is its direction and its spread across draws.

Usage:
  python posterior_ratio_response.py --posterior "Results/Tier1/<run>/posterior_samples_pm.nc" --params a1,c3,a2
  python posterior_ratio_response.py --posterior job_files/multiparam_tests/mp_posterior.json \\
      --key a1c3_no_floor/C14 --system C14 --params a1,c3 --draws 3 --grid 3   # quick test
"""
import argparse
import itertools
import json
import re
import time
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import numpy as np

from forward_model import PROJECT, ForwardModel, posterior_draws
from make_tier1_rate_data import tolerances

HERE = Path(__file__).resolve().parent
_spec = spec_from_file_location("morris_screen", HERE.parent / "chain_system_sensitivity_analysis" / "morris_screen.py")
morris = module_from_spec(_spec)
_spec.loader.exec_module(morris)          # for the same three objectives the Morris screens use


def sweep_rows(fm, ratios):
    y0 = fm.y0()
    rows = []
    for R in ratios:
        y = y0.copy()
        for e in ("FabF", "FabB"):
            y[fm.index[e]] = y0[fm.index[e]] * np.sqrt(R)
        y[fm.index["TesA"]] = y0[fm.index["TesA"]] / np.sqrt(R)
        rows.append(y)
    return np.stack(rows)


def grid_rows(fm, mults):
    y0 = fm.y0()
    rows, settings = [], []
    for mf, mb, mt in itertools.product(mults, repeat=3):
        y = y0.copy()
        y[fm.index["FabF"]] *= mf
        y[fm.index["FabB"]] *= mb
        y[fm.index["TesA"]] *= mt
        rows.append(y)
        settings.append((mf, mb, mt))
    return np.stack(rows), settings


def evaluate(fm, theta, rows, objectives, n_obj):
    ys, ok = fm.run(rows, theta)
    return np.array([objectives(y[-1]) if good else np.full(n_obj, np.nan) for y, good in zip(ys, ok)])


def readout(fm, theta, sweep, grid, settings, ratios, objectives, n_obj, y_base):
    cl = evaluate(fm, theta, sweep, objectives, n_obj)[:, 1]             # average chain length
    x = np.log10(ratios)
    ok = np.isfinite(cl)
    slope = float(np.polyfit(x[ok], cl[ok], 1)[0]) if ok.sum() >= 2 else float("nan")
    g = evaluate(fm, theta, grid, objectives, n_obj)[:, 1]
    base_ratio = (y_base["FabF"] + y_base["FabB"]) / y_base["TesA"]

    def ratio_of(i):
        mf, mb, mt = settings[i]
        return float((mf * y_base["FabF"] + mb * y_base["FabB"]) / (mt * y_base["TesA"]) / base_ratio)

    i_max, i_min = int(np.nanargmax(g)), int(np.nanargmin(g))
    return {"avg_chain_length": [float(v) for v in cl], "slope_per_decade": slope,
            "monotone_increasing": bool(np.all(np.diff(cl[ok]) > 0)),
            "change_low_to_high": float(cl[ok][-1] - cl[ok][0]) if ok.any() else float("nan"),
            "longest": {"avg_chain_length": float(g[i_max]), "setting": settings[i_max], "ratio": ratio_of(i_max)},
            "shortest": {"avg_chain_length": float(g[i_min]), "setting": settings[i_min], "ratio": ratio_of(i_min)},
            "failed_solves": int(np.sum(~np.isfinite(cl)) + np.sum(~np.isfinite(g)))}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--posterior", required=True, help="posterior_samples_pm.nc, or an export_posterior_series.py JSON")
    ap.add_argument("--key", default=None, help="for a JSON export: '<set>/<system>'")
    ap.add_argument("--system", default="C14+unsat")
    ap.add_argument("--params", default="a1,c3,a2")
    ap.add_argument("--draws", type=int, default=50)
    ap.add_argument("--span", type=float, default=10.0)
    ap.add_argument("--n_ratios", type=int, default=9, help="ratio sweep points over [1/span, span]")
    ap.add_argument("--grid", type=int, default=5, help="optimum grid points per enzyme axis")
    ap.add_argument("--out", default=str(HERE / "posterior_ratio_response.json"))
    a = ap.parse_args()
    params = a.params.split(",")
    src = Path(a.posterior) if Path(a.posterior).is_absolute() else PROJECT / a.posterior

    t0 = time.perf_counter()
    rtol, atol = tolerances(a.system)
    fm = ForwardModel(a.system, times=[720.0], rtol=rtol, atol=atol)
    targets = [s for s in fm.species if re.fullmatch(r"C\d+_FA(_unsat)?", s)]
    objectives, names = morris.make_objectives(fm._sys, targets, 1e-9)
    ratios = np.logspace(-np.log10(a.span), np.log10(a.span), a.n_ratios)
    mults = np.logspace(-np.log10(a.span), np.log10(a.span), a.grid)
    sweep = sweep_rows(fm, ratios)
    grid, settings = grid_rows(fm, mults)
    y0 = fm.y0()
    y_base = {e: float(y0[fm.index[e]]) for e in ("FabF", "FabB", "TesA")}

    point = readout(fm, fm.theta(), sweep, grid, settings, ratios, objectives, len(names), y_base)
    draws = posterior_draws(src, params, a.draws, key=a.key)
    per_draw = []
    for i, d in enumerate(draws):
        per_draw.append({"params": d, **readout(fm, fm.theta(d), sweep, grid, settings, ratios, objectives, len(names), y_base)})
        print(f"  draw {i + 1}/{len(draws)} ({time.perf_counter() - t0:.0f} s)", flush=True)

    slopes = np.array([r["slope_per_decade"] for r in per_draw])
    summary = {"fraction_positive_slope": float(np.mean(slopes > 0)),
               "fraction_monotone_increasing": float(np.mean([r["monotone_increasing"] for r in per_draw])),
               "slope_per_decade_5_50_95": [float(v) for v in np.nanpercentile(slopes, [5, 50, 95])],
               "longest_ratio_5_50_95": [float(v) for v in np.nanpercentile([r["longest"]["ratio"] for r in per_draw], [5, 50, 95])],
               "shortest_ratio_5_50_95": [float(v) for v in np.nanpercentile([r["shortest"]["ratio"] for r in per_draw], [5, 50, 95])]}
    out = {"system": a.system, "posterior": str(a.posterior), "key": a.key, "params": params, "n_draws": len(draws),
           "tolerance": [rtol, atol], "ratios": [float(r) for r in ratios], "grid_multipliers": [float(m) for m in mults],
           "point_estimate": point, "summary": summary, "draws": per_draw,
           "seconds": round(time.perf_counter() - t0, 1)}
    Path(a.out).write_text(json.dumps(out, indent=1) + "\n")

    print(f"\npoint estimate: slope {point['slope_per_decade']:+.4f} chain-length units per decade of ratio, "
          f"monotone {point['monotone_increasing']}; longest at ratio {point['longest']['ratio']:.3g}, "
          f"shortest at {point['shortest']['ratio']:.3g}")
    print(f"posterior ({len(draws)} draws): positive slope in {summary['fraction_positive_slope']:.0%}, monotone in "
          f"{summary['fraction_monotone_increasing']:.0%}; slope 5/50/95% {np.round(summary['slope_per_decade_5_50_95'], 4).tolist()}")
    print(f"optimum ratio, longest 5/50/95% {np.round(summary['longest_ratio_5_50_95'], 3).tolist()}, "
          f"shortest {np.round(summary['shortest_ratio_5_50_95'], 3).tolist()}")
    print(f"wrote {a.out} ({out['seconds']} s)")


if __name__ == "__main__":
    main()
