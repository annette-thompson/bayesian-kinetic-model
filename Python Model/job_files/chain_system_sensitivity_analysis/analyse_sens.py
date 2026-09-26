"""Consolidate the three-system scaling sensitivity sweep into a ranking."""
import json
from pathlib import Path

p = Path("/projects/anth4580/Bayesian/job_files/chain_system_sensitivity_analysis/scaling_sensitivity.json")
d = json.loads(p.read_text())
systems = list(d)

print("#" * 74)
print("d1 vs d2 AT FULL PRECISION  (identical => bug; tiny => near-degenerate)")
print("#" * 74)
for s in systems:
    a = d[s].get("d1", {}).get("curve", {})
    b = d[s].get("d2", {}).get("curve", {})
    if not a:
        continue
    print("=== %s ===" % s)
    for m in a:
        va, vb = a[m], b[m]
        if va is None or vb is None:
            continue
        rel = abs(va - vb) / max(abs(va), 1e-30) * 100
        flag = "*** EXACTLY IDENTICAL ***" if va == vb else "differ %.8f%%" % rel
        print("  m=%-8s d1=%.12g  d2=%.12g   %s" % (m, va, vb, flag))

print()
print("#" * 74)
print("IDENTIFIABILITY WINDOW BY SYSTEM (detect / saturate / span in decades)")
print("#" * 74)
groups = sorted({g for s in systems for g in d[s]})
print("%-6s" % "GROUP" + "".join("%28s" % s for s in systems))
for g in groups:
    row = "%-6s" % g
    for s in systems:
        e = d[s].get(g)
        if not e:
            row += "%28s" % "-"
            continue
        sp = e.get("span_decades")
        sp = "-" if sp is None else "%.1f" % sp
        row += "%28s" % ("det=%s sat=%s sp=%s" % (e.get("detect"), e.get("saturate"), sp))
    print(row)

print()
print("#" * 74)
print("PEAK RESPONSE (max median |rel change| over the swept range, %)")
print("#" * 74)
print("%-6s" % "GROUP" + "".join("%16s" % s for s in systems))
for g in groups:
    row = "%-6s" % g
    for s in systems:
        e = d[s].get(g)
        if not e:
            row += "%16s" % "-"
            continue
        vals = [v for v in e["curve"].values() if v is not None]
        row += "%16s" % ("%.0f%%" % (max(vals) * 100) if vals else "-")
    print(row)
