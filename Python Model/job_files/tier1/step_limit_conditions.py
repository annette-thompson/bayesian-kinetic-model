"""Which one-enzyme perturbations survive tighter solver-step limits, per system group?

For FabH, ACP, malonyl-CoA, FabF, FabB and TesA at 0.1/0.2/0.5/2/5/10x (from the 0.1-10x
sweep reports), at each step limit L in 1.5, 1.4, 1.3, 1.2, 1.1, 1.0:

  passes L     on EVERY system of the group: converged, loose-tolerance steps <= L x that
               system's baseline steps, strict-tolerance steps <= L x the baseline's
               strict steps, and total C16 Equivalents >= 10% of baseline
  usable at L  passes L AND belongs to at least one set of 6 (one condition per enzyme,
               all passing L) whose 6 conditions and the baseline are pairwise >= 0.2
               decades apart on every system of the group

A limit is feasible for a group when every enzyme keeps at least one usable condition.
The sweep itself capped both step counts at 1.5x, so 1.5 is the loosest level available.

C4_NoFB has no FabF or FabB, so for "all 14" the set search covers the enzymes every
system has; C6-C20+unsat is added as the widest group with all six.

FabB perturbations move no fatty acid by 0.2 decades from baseline on any saturated system
(0.04-0.12 at best; it matters only in the unsaturated branch), so every group containing a
saturated system has no 6-enzyme set. --without FabB repeats the analysis for the rest.

Usage: python step_limit_conditions.py [--without FabB]   (writes step_limit_conditions[_noFabB].json)
"""
import argparse
import itertools
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPORTS = HERE / "titration_0.1-10x"
ALL = ["C4_NoFB", "C6", "C8", "C10", "C12", "C12+unsat", "C14", "C14+unsat",
       "C16", "C16+unsat", "C18", "C18+unsat", "C20", "C20+unsat"]
GROUPS = {"all 14": ALL, "C6-C20+unsat": ALL[1:], "C8-C20+unsat": ALL[2:], "C12-C20+unsat": ALL[4:]}
SPECIES = ["FabH", "ACP", "C3_MalCoA", "FabF", "FabB", "TesA"]
SHORT = {"C3_MalCoA": "MalCoA"}
FACTORS = [0.1, 0.2, 0.5, 2.0, 5.0, 10.0]
LIMITS = [1.5, 1.4, 1.3, 1.2, 1.1, 1.0]
MIN_LOG_DIFF = 0.2


def lab(sp, f):
    return f"{sp} x{f:g}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--without", default="", help="comma-separated enzymes to leave out, e.g. FabB")
    a = ap.parse_args()
    dropped = [x for x in a.without.split(",") if x]
    species = [sp for sp in SPECIES if sp not in dropped]
    rep = {s: json.loads((REPORTS / f"tier1_conditions_{s}.json").read_text()) for s in ALL}
    cand = {s: {c["label"]: c for c in r["candidate_log"] if "out" in c} for s, r in rep.items()}
    labels = [lab(sp, f) for sp in species for f in FACTORS]

    # Pairwise diversity, once per system: row/col 0 is the baseline.
    far = {}
    for s in ALL:
        outs = [rep[s]["baseline_out"]] + [cand[s][l]["out"] if l in cand[s] else None for l in labels]
        n = len(outs)
        ok = np.zeros((n, n), dtype=bool)
        for i, j in itertools.combinations(range(n), 2):
            if outs[i] is None or outs[j] is None:
                continue
            la, lb = np.log10(np.abs(outs[i]) + 1e-300), np.log10(np.abs(outs[j]) + 1e-300)
            ok[i, j] = ok[j, i] = np.max(np.abs(la - lb)) >= MIN_LOG_DIFF
        far[s] = ok

    def step_ratio(s, l):
        c = cand[s].get(l)
        if c is None:
            return None
        return max(c["steps"] / rep[s]["baseline_steps"], c["strict_steps"] / rep[s]["baseline_strict_steps"])

    result = {}
    for g, members in GROUPS.items():
        present = [sp for sp in species if all(any(lab(sp, f) in cand[s] or
                   any(e["label"] == lab(sp, f) for e in rep[s]["candidate_log"]) for f in FACTORS) for s in members)]
        OK = np.logical_and.reduce([far[s] for s in members])
        print(f"\n=== {g} ({len(members)} systems)" + ("" if present == species else
              f"  -- {sorted(set(species) - set(present))} absent from C4_NoFB; sets use {present}"))
        result[g] = {}
        for L in LIMITS:
            passing = {sp: [f for f in FACTORS
                            if all((r := step_ratio(s, lab(sp, f))) is not None and r <= L + 1e-9 for s in members)]
                       for sp in present}
            usable = {sp: set() for sp in present}
            n_sets = 0
            if all(passing[sp] for sp in present):
                idx = {lab(sp, f): 1 + labels.index(lab(sp, f)) for sp in present for f in FACTORS}

                def extend(chosen, k):
                    nonlocal n_sets
                    if k == len(present):
                        n_sets += 1
                        for sp, f in zip(present, chosen):
                            usable[sp].add(f)
                        return
                    for f in passing[present[k]]:
                        i = idx[lab(present[k], f)]
                        if OK[0, i] and all(OK[idx[lab(present[m], chosen[m])], i] for m in range(k)):
                            extend(chosen + [f], k + 1)
                extend([], 0)
            feasible = all(usable[sp] for sp in present)
            fmt = lambda fs: ",".join(f"{f:g}" for f in sorted(fs)) or "-"
            print(f"  <= {L:.1f}x  passes: " + "  ".join(f"{SHORT.get(sp, sp)} [{fmt(passing[sp])}]" for sp in present))
            print(f"           usable: " + "  ".join(f"{SHORT.get(sp, sp)} [{fmt(usable[sp])}]" for sp in present)
                  + f"   -> {n_sets} valid sets" + ("" if feasible else "  NOT FEASIBLE"))
            result[g][f"{L:.1f}"] = dict(passes={sp: passing[sp] for sp in present},
                                         usable={sp: sorted(usable[sp]) for sp in present},
                                         n_valid_sets=n_sets, feasible=feasible)
    out = HERE / f"step_limit_conditions{'_no' + '_'.join(dropped) if dropped else ''}.json"
    out.write_text(json.dumps(result, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
