"""Local sensitivity near nominal -- the quantity that actually sets posterior width.

Peak response over a 6-decade sweep conflates two different things: how sharply the
data pins the parameter down near its true value (which is what the posterior width
depends on), and how violently the model breaks at extreme values (which is mostly a
statement about solver robustness). Posterior precision is governed by the LOCAL
slope at the true value relative to the noise, so this reports the m=0.5 and m=2
points -- a factor-of-two perturbation either side of nominal -- and the implied
signal-to-noise against the 10% relative noise model.
"""
import json
from pathlib import Path

p = Path("/projects/anth4580/Bayesian/job_files/chain_system_sensitivity_analysis/scaling_sensitivity.json")
d = json.loads(p.read_text())
systems = list(d)
NOISE = 0.10

groups = sorted({g for s in systems for g in d[s]})
print("LOCAL sensitivity: median |rel change| for a factor-of-2 perturbation")
print("SNR = that change / 10% noise.  SNR<1 means a 2x error is invisible.\n")
hdr = "%-6s" % "GROUP"
for s in systems:
    hdr += "%26s" % ("%s (0.5x / 2x | SNR)" % s)
print(hdr)
print("-" * len(hdr))
rank = {}
for g in groups:
    row = "%-6s" % g
    worst = []
    for s in systems:
        e = d[s].get(g)
        if not e:
            row += "%26s" % "-"
            continue
        c = e["curve"]
        lo, hi = c.get("0.5"), c.get("2.0")
        if lo is None or hi is None:
            row += "%26s" % "-"
            continue
        loc = min(lo, hi)          # the harder-to-detect side sets identifiability
        snr = loc / NOISE
        worst.append(snr)
        row += "%26s" % ("%.0f%% / %.0f%% | %.1f" % (lo * 100, hi * 100, snr))
    if worst:
        rank[g] = min(worst)
    print(row)

print("\nRanked by WORST-CASE local SNR across the three systems")
print("(the binding constraint: a group must be identifiable on every rung)\n")
print("%-8s%12s   %s" % ("GROUP", "min SNR", "reading"))
for g, v in sorted(rank.items(), key=lambda kv: -kv[1]):
    note = ("strong" if v >= 4 else "adequate" if v >= 2 else
            "marginal" if v >= 1 else "BELOW NOISE")
    print("%-8s%12.1f   %s" % (g, v, note))
