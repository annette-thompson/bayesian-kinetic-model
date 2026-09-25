"""Which titration conditions pass every criterion on every system of a group?

Reads the full 0.1-10x sweep reports (titration_0.1-10x/tier1_conditions_<system>.json)
and, for each group, intersects the conditions that passed ("kept" or "usable beyond
n_keep": converged within the step cap, C16 Equivalents >= 10% of baseline, passed the
strict-tolerance check, and >= 0.2 decades from every other usable condition and the
baseline). Any subset of one system's usable list is still mutually diverse, so a shared
set passes on each system as a set.

The shared 9 for a group are the shared passes with the fewest solver steps summed over
the group's systems (ties broken by the worst per-system ratio to that system's baseline).

Usage: python shared_conditions.py   (writes shared_conditions.json next to the reports)
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPORTS = HERE / "titration_0.1-10x"
ALL = ["C4_NoFB", "C6", "C8", "C10", "C12", "C12+unsat", "C14", "C14+unsat",
       "C16", "C16+unsat", "C18", "C18+unsat", "C20", "C20+unsat"]
GROUPS = {
    "all14": ALL,
    "C8_to_C20+unsat": ALL[2:],
    "C12_to_C20+unsat": ALL[4:],
    "tier1_ladder": ["C8", "C12", "C14", "C14+unsat"],
}
N_KEEP = 9
PASS = ("kept", "usable beyond n_keep")


def main():
    reports, missing = {}, []
    for s in ALL:
        f = REPORTS / f"tier1_conditions_{s}.json"
        if f.exists():
            reports[s] = json.loads(f.read_text())
        else:
            missing.append(s)
    for s, r in reports.items():
        assert r.get("titration_factors") == [0.5, 2.0, 0.2, 5.0, 0.1, 10.0], (s, r.get("titration_factors"))
    logs = {s: {c["label"]: c for c in r["candidate_log"]} for s, r in reports.items()}
    passes = {s: {l for l, c in logs[s].items() if c["result"] in PASS} for s in reports}

    out = {"missing_reports": missing, "per_system_pass_count": {s: len(p) for s, p in passes.items()}, "groups": {}}
    print("usable conditions per system:", {s: len(p) for s, p in passes.items()})
    if missing:
        print("MISSING reports:", missing)
    for g, members in GROUPS.items():
        have = [s for s in members if s in passes]
        if len(have) < len(members):
            print(f"\n{g}: skipped, reports missing for {[s for s in members if s not in passes]}")
            continue
        shared = set.intersection(*(passes[s] for s in have))
        rows = []
        for l in shared:
            steps = {s: logs[s][l]["steps"] for s in have}
            ratio = max(steps[s] / reports[s]["baseline_steps"] for s in have)
            rows.append((sum(steps.values()), ratio, l))
        rows.sort()
        chosen = [l for _, _, l in rows[:N_KEEP]]
        near = sorted((l, [s for s in have if l not in passes[s]]) for l in set().union(*(passes[s] for s in have))
                      if sum(l in passes[s] for s in have) == len(have) - 1)
        out["groups"][g] = dict(systems=have, n_shared=len(shared), enough=len(shared) >= N_KEEP,
                                shared_ranked=[dict(label=l, summed_steps=t, worst_ratio_to_baseline=round(r, 3))
                                               for t, r, l in rows],
                                chosen=chosen, missing_in_one_system=near)
        print(f"\n{g} ({len(have)} systems): {len(shared)} shared passes -> "
              f"{'enough' if len(shared) >= N_KEEP else 'NOT enough'} for {N_KEEP}")
        for t, r, l in rows:
            print(f"  {'*' if l in chosen else ' '} {l:<16} summed steps {t:5d}  worst x baseline {r:.3f}")
        if near:
            print("  pass on all but one system:", ", ".join(f"{l} (not {m[0]})" for l, m in near))
    (HERE / "shared_conditions.json").write_text(json.dumps(out, indent=2))
    print(f"\nwrote {HERE / 'shared_conditions.json'}")


if __name__ == "__main__":
    main()
