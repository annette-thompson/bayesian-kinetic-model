"""Select the benchmark test case by computed metrics, not by judgement.

Requirement: the test case must be as cheap as possible WITHOUT foreclosing any
downstream test. Cheapness alone would pick a parameter nothing can be asked about,
so every candidate is scored on capability first and cost second, and a candidate
failing any capability gate is excluded regardless of speed.

Every metric below is computed from the model files or from measurements already on
disk. None is hand-assigned, so re-running this on a different system reproduces the
selection logic rather than inheriting this system's answer.

CAPABILITY GATES (a candidate must pass all four)

  G1  ties >= 2 constants with DISTINCT nominal values
      A group tying one constant, or several identical ones, is mathematically the
      same as freeing a raw rate constant. Section 3.2 asks whether the data support
      the GROUPING; a group with no internal spread cannot answer that, so such a
      parameter forecloses 3.2 no matter how fast it runs.

  G2  present in every chain-length system
      Results have to transfer along the ladder. `f` exists only in +unsat systems,
      so a conclusion drawn from it cannot be compared across rungs.

  G3  not inside a compound scaling expression
      b1 appears as (b1/b2) and (b1/b3); its posterior is then partly a statement
      about b2 and b3, which makes 3.3's per-parameter shrinkage unattributable.

  G4  identifiable: measured influence above the floor
      A parameter the data cannot constrain yields posterior == prior. It would
      converge fast while teaching nothing, which is the failure mode cheapness
      alone would select for. Threshold is relative (a fraction of the strongest
      parameter on the same objective), so it transfers to systems with different
      absolute scales.

COST METRICS (used to rank only among candidates that pass all gates)

  Morris mu* on each objective, and sigma/mu* as an interaction proxy: a parameter
  with low sigma/mu* behaves consistently regardless of where the others sit, which
  makes single-parameter results more likely to survive into joint inference.

Usage: python select_testcase.py --system C8
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path("/projects/anth4580/Bayesian")
CFG_ROOT = ROOT / "Results" / "Chain Scaling Tests"
SCAL = ROOT / "job_files" / "chain_system_sensitivity_analysis"
ALL_SYSTEMS = ["C4_NoFB", "C6", "C8", "C10", "C12", "C12+unsat", "C14", "C14+unsat",
               "C16", "C16+unsat", "C18", "C18+unsat", "C20", "C20+unsat"]

# G4 threshold: a candidate must reach this fraction of the strongest parameter's
# mu* on at least one objective. Relative, so it transfers between systems.
IDENTIFIABILITY_FLOOR = 0.10


def max_chain(system):
    import re
    return int(re.match(r"C(\d+)", system).group(1))


def tied_constants(system):
    """Per group: the instantiated rate constants it scales, and their nominal values.

    Chain templates are expanded over the chain lengths the system actually contains,
    inheriting the file-level `defaults` block when a reaction omits its own list.
    """
    cfg = json.loads((CFG_ROOT / f"Chain {system} - a1 tightest"
                      / "solver_params.json").read_text())
    srcs = [ROOT / p for p in cfg["output_paths"]["reactions_source"]]
    mx, allow_unsat = max_chain(system), "+unsat" in system
    out, compound, enzymes = {}, {}, {}
    for src in srcs:
        if not src.exists():
            continue
        doc = yaml.safe_load(src.read_text())
        dflt = doc.get("defaults", {}) or {}
        for r in doc.get("reactions", []) or []:
            templ = bool(r.get("chain_template") or r.get("saturation_template"))
            ssrc = r if "sat_chain_lengths" in r else dflt
            usrc = r if "unsat_chain_lengths" in r else dflt
            sat = [int(c) for c in (ssrc.get("sat_chain_lengths") or [])]
            uns = [int(c) for c in (usrc.get("unsat_chain_lengths") or [])]
            chains = [c for c in sat if c <= mx]
            if allow_unsat:
                chains += [c for c in uns if c <= mx]
            n = len(chains) if templ else 1
            if n == 0:
                continue
            for field, vkeys in (("scaling_group", ("rate_const_value", "sat_rate_const_value")),
                                 ("rvs_scaling_group", ("rvs_rate_const_value",))):
                expr = r.get(field)
                if not expr:
                    continue
                names = {t for t in __import__("re").findall(r"[A-Za-z]\w*", str(expr))
                         if t in cfg.get("scaling_groups", {})}
                vals = []
                for vk in vkeys:
                    v = r.get(vk)
                    if isinstance(v, list) and sat:
                        vals += [float(v[sat.index(c)]) for c in chains
                                 if c in sat and sat.index(c) < len(v)]
                    elif isinstance(v, (int, float)):
                        vals += [float(v)] * n
                for g in names:
                    out.setdefault(g, []).extend(vals)
                    enzymes.setdefault(g, set()).add(src.stem)
                    if len(names) > 1:
                        compound[g] = compound.get(g, 0) + n
    return out, compound, enzymes


def morris(system):
    f = SCAL / f"morris_{system.replace('+','_')}.json"
    if not f.exists():
        return {}
    j = json.loads(f.read_text())
    out = {}
    for obj, rows in j["results"].items():
        top = max(r["mu_star"] for r in rows) or 1.0
        for r in rows:
            out.setdefault(r["group"], {})[obj] = dict(
                mu=r["mu_star"], frac=r["mu_star"] / top,
                ratio=(r["sigma"] / r["mu_star"]) if r["mu_star"] else float("nan"))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--system", default="C8")
    a = ap.parse_args()

    consts, compound, enzymes = tied_constants(a.system)
    mor = morris(a.system)
    present_all = {}
    for g in consts:
        present_all[g] = all(
            g in json.loads((CFG_ROOT / f"Chain {s} - a1 tightest"
                             / "solver_params.json").read_text()).get("scaling_groups", {})
            for s in ALL_SYSTEMS)

    print(f"=== candidate scoring: {a.system} ===")
    print(f"    G1 >=2 distinct nominal values | G2 in all 14 systems | "
          f"G3 not compound | G4 mu* >= {IDENTIFIABILITY_FLOOR:.0%} of top\n")
    hdr = ("group", "n_const", "n_distinct", "enz", "compound", "in_all",
           "best mu*%", "sig/mu*", "gates", "verdict")
    W = (7, 9, 11, 5, 10, 8, 10, 9, 8, 28)
    print("".join(h.ljust(w) for h, w in zip(hdr, W)))
    print("-" * sum(W))

    rows = []
    for g in sorted(consts):
        vals = consts[g]
        nd = len({round(v, 10) for v in vals})
        m = mor.get(g, {})
        best = max((v["frac"] for v in m.values()), default=0.0)
        ratio = np.median([v["ratio"] for v in m.values()]) if m else float("nan")
        g1, g2 = nd >= 2, present_all.get(g, False)
        g3, g4 = compound.get(g, 0) == 0, best >= IDENTIFIABILITY_FLOOR
        gates = "".join(("1" if x else ".") for x in (g1, g2, g3, g4))
        fails = [n for n, ok in zip(("G1", "G2", "G3", "G4"), (g1, g2, g3, g4)) if not ok]
        verdict = "CANDIDATE" if not fails else "excluded: " + ",".join(fails)
        rows.append(dict(group=g, n_const=len(vals), n_distinct=nd,
                         enzymes=len(enzymes.get(g, ())), compound=compound.get(g, 0),
                         in_all=g2, best_frac=best, sigma_ratio=float(ratio),
                         gates=gates, passes=not fails))
        print("".join(str(c).ljust(w) for c, w in zip(
            (g, len(vals), nd, len(enzymes.get(g, ())), compound.get(g, 0),
             "yes" if g2 else "NO", f"{best*100:.1f}", f"{ratio:.2f}", gates, verdict), W)))

    ok = [r for r in rows if r["passes"]]
    print(f"\n{len(ok)} of {len(rows)} pass all gates.")
    if ok:
        ok.sort(key=lambda r: (-r["best_frac"], r["sigma_ratio"]))
        print("\nRanked by influence, then by interaction stability (lower sigma/mu* better):")
        for i, r in enumerate(ok, 1):
            print(f"  {i}. {r['group']:<4} mu*={r['best_frac']*100:5.1f}% of top   "
                  f"sigma/mu*={r['sigma_ratio']:.2f}   ties {r['n_const']} constants, "
                  f"{r['n_distinct']} distinct, {r['enzymes']} enzymes")
    out = Path(__file__).resolve().parent / f"select_testcase_{a.system}.json"
    out.write_text(json.dumps(rows, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
