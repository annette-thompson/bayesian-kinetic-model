"""One perturbation each of FabH, ACP, malonyl-CoA, FabF, FabB and TesA (6 conditions +
baseline) that passes every criterion on every system of a group.

A combination is valid for a group when, on every system in it:
  - each of the 6 conditions converged within 1.5x baseline solver steps, re-solved at the
    strict tolerance within 1.5x the baseline's strict steps, and kept total C16
    Equivalents >= 10% of baseline (all from the 0.1-10x sweep reports), and
  - the 6 conditions and the baseline are pairwise >= 0.2 decades apart (max over fatty
    acid species of |log10 ratio|), re-tested within THIS set -- not against whatever other
    conditions the sweep happened to keep first.

Each valid combination is scored on
  steps    mean and worst loose-tolerance solver steps, % of that system's baseline
  strict   the same at strict tolerance
  size     mean |log10 factor| (0.30 = 2-fold, 0.70 = 5-fold, 1.00 = 10-fold)
  margin   the smallest pairwise distance in decades over all systems (>= 0.2 by construction)
and reported as: uniform-fold sets (every perturbation the same fold change), the
cheapest set, the set with the lowest worst case, the smallest perturbations, and the
Pareto front of mean steps against size.

C4_NoFB has no FabF or FabB, so the widest group is C6 to C20+unsat.

Usage: python universal_conditions.py   (reads titration_0.1-10x/, writes universal_conditions.json)
"""
import itertools
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPORTS = HERE / "titration_0.1-10x"
ALL = ["C4_NoFB", "C6", "C8", "C10", "C12", "C12+unsat", "C14", "C14+unsat",
       "C16", "C16+unsat", "C18", "C18+unsat", "C20", "C20+unsat"]
GROUPS = {
    "C6_to_C20+unsat": ALL[1:],
    "C8_to_C20+unsat": ALL[2:],
    "C12_to_C20+unsat": ALL[4:],
    "tier1_ladder": ["C8", "C12", "C14", "C14+unsat"],
}
SPECIES = ["FabH", "ACP", "C3_MalCoA", "FabF", "FabB", "TesA"]
FACTORS = [0.1, 0.2, 0.5, 2.0, 5.0, 10.0]
MIN_LOG_DIFF = 0.2


def log_distance(a, b):
    la, lb = np.log10(np.abs(np.asarray(a)) + 1e-300), np.log10(np.abs(np.asarray(b)) + 1e-300)
    return float(np.max(np.abs(la - lb)))


def label(sp, f):
    return f"{sp} x{f:g}"


def main():
    reports = {s: json.loads((REPORTS / f"tier1_conditions_{s}.json").read_text()) for s in ALL
               if (REPORTS / f"tier1_conditions_{s}.json").exists()}
    missing = [s for s in ALL if s not in reports]
    cands = {s: {c["label"]: c for c in r["candidate_log"] if "out" in c} for s, r in reports.items()}
    result = {"missing_reports": missing, "groups": {}}
    if missing:
        print("MISSING reports:", missing)

    for g, members in GROUPS.items():
        if any(s not in reports for s in members):
            print(f"\n{g}: skipped (missing reports)")
            continue
        eligible = {sp: [f for f in FACTORS if all(label(sp, f) in cands[s] for s in members)] for sp in SPECIES}
        print(f"\n=== {g} ({len(members)} systems)")
        print("  factors passing steps / strict / C16 on every system:",
              "; ".join(f"{sp} {eligible[sp]}" for sp in SPECIES))
        if any(not v for v in eligible.values()):
            print("  no valid set: some species has no factor passing on every system")
            result["groups"][g] = dict(systems=members, eligible=eligible, n_valid=0)
            continue

        # per-system distances between every eligible condition and the baseline
        valid = []
        for combo in itertools.product(*(eligible[sp] for sp in SPECIES)):
            labs = [label(sp, f) for sp, f in zip(SPECIES, combo)]
            margin, ok = np.inf, True
            for s in members:
                outs = [reports[s]["baseline_out"]] + [cands[s][l]["out"] for l in labs]
                for i, j in itertools.combinations(range(7), 2):
                    d = log_distance(outs[i], outs[j])
                    margin = min(margin, d)
                    if d < MIN_LOG_DIFF:
                        ok = False
                        break
                if not ok:
                    break
            if not ok:
                continue
            steps = np.array([[cands[s][l]["steps"] / reports[s]["baseline_steps"] for l in labs] for s in members]) * 100
            strict = np.array([[cands[s][l]["strict_steps"] / reports[s]["baseline_strict_steps"] for l in labs]
                               for s in members]) * 100
            sizes = np.abs(np.log10(combo))
            valid.append(dict(conditions=labs, factors=list(combo),
                              mean_steps_pct=float(steps.mean()), worst_steps_pct=float(steps.max()),
                              mean_strict_pct=float(strict.mean()), worst_strict_pct=float(strict.max()),
                              mean_size_log10=float(sizes.mean()), max_size_log10=float(sizes.max()),
                              uniform_fold=bool(np.allclose(sizes, sizes[0])),
                              direction=("all down" if all(f < 1 for f in combo) else
                                         "all up" if all(f > 1 for f in combo) else "mixed"),
                              margin_decades=float(margin)))
        print(f"  valid 6-condition sets: {len(valid)} of {int(np.prod([len(v) for v in eligible.values()]))} combinations")
        if not valid:
            result["groups"][g] = dict(systems=members, eligible=eligible, n_valid=0)
            continue

        def show(title, rows):
            print(f"  {title}")
            for r in rows:
                print(f"    {', '.join(r['conditions'])}")
                print(f"       steps mean {r['mean_steps_pct']:.1f}% worst {r['worst_steps_pct']:.1f}% | strict mean "
                      f"{r['mean_strict_pct']:.1f}% worst {r['worst_strict_pct']:.1f}% | size {r['mean_size_log10']:.2f} "
                      f"| {r['direction']} | margin {r['margin_decades']:.2f} dec")

        by_steps = sorted(valid, key=lambda r: (r["mean_steps_pct"], r["worst_steps_pct"]))
        by_worst = sorted(valid, key=lambda r: (r["worst_steps_pct"], r["mean_steps_pct"]))
        by_size = sorted(valid, key=lambda r: (r["mean_size_log10"], r["mean_steps_pct"]))
        pareto, best = [], np.inf
        for r in sorted(valid, key=lambda r: (r["mean_size_log10"], r["mean_steps_pct"])):
            if r["mean_steps_pct"] < best - 1e-9:
                pareto.append(r); best = r["mean_steps_pct"]
        uniform = {}
        for fold, sz in (("2-fold", np.log10(2)), ("5-fold", np.log10(5)), ("10-fold", 1.0)):
            u = [r for r in valid if r["uniform_fold"] and abs(r["mean_size_log10"] - sz) < 1e-6]
            uniform[fold] = sorted(u, key=lambda r: (r["mean_steps_pct"], r["worst_steps_pct"]))
            print(f"  uniform {fold}: {len(u)} valid" + (f" (directions: {sorted(set(r['direction'] for r in u))})" if u else ""))
            if u:
                show(f"  best uniform {fold}:", uniform[fold][:2])
        show("cheapest (lowest mean steps):", by_steps[:3])
        show("lowest worst-case steps:", by_worst[:2])
        show("smallest perturbations:", by_size[:2])
        show("Pareto front, mean steps vs size:", pareto)
        result["groups"][g] = dict(systems=members, eligible=eligible, n_valid=len(valid),
                                   uniform={k: v[:5] for k, v in uniform.items()}, cheapest=by_steps[:10],
                                   lowest_worst=by_worst[:10], smallest=by_size[:10], pareto=pareto)
    (HERE / "universal_conditions.json").write_text(json.dumps(result, indent=2))
    print(f"\nwrote {HERE / 'universal_conditions.json'}")


if __name__ == "__main__":
    main()
