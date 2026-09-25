"""Figure S1-style view of the rate-design Tier-1 data: A initial rates, B profile, C time course.

Plots what the likelihood will actually see -- the noisy values with the sigma they were
drawn with -- not the clean simulation, so the error bars are the ones the fit has to work
against.

Usage: python -u plot_tier1_rate_data.py C20+unsat [--root Data/Tier1_rates]
"""
import argparse, re, sys
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT = Path(__file__).resolve().parent.parent.parent
LABELS = ["baseline", "FabH 0.1 uM", "FabB 0 uM", "TesA 0.5 uM", "FabZ 0 uM"]
INK, BAR, LINE = "#1f2933", "#2f6f9f", "#c4643a"

ap = argparse.ArgumentParser()
ap.add_argument("system")
ap.add_argument("--root", default="Data/Tier1_rates")
ap.add_argument("--out", default=None)
a = ap.parse_args()
d = PROJECT / a.root / f"Chain_{a.system}"
rates = pd.read_csv(d / "init_vs_rate.csv")
prof = pd.read_csv(d / "init_vs_final_conc.csv")
tc = pd.read_csv(d / "time_vs_conc.csv")

RCOL = "Initial Rate (uM C16 Equivalents/min)"
fa = [c for c in prof.columns if re.fullmatch(r"C\d+_FA(_unsat)? \(uM\)", c)]
order = sorted(fa, key=lambda c: (int(re.match(r"C(\d+)", c).group(1)), "_unsat" in c))
chain = [re.match(r"C(\d+)_FA(_unsat)?", c).group(1) + (":1" if "_unsat" in c else "") for c in order]

fig, axes = plt.subplots(1, 3, figsize=(17.5, 5.0), gridspec_kw={"width_ratios": [5, 6, 5]})

ax = axes[0]
x = np.arange(len(rates))
ax.bar(x, rates[RCOL], 0.62, yerr=rates.get(RCOL + "_sigma"), capsize=4, color=BAR)
ax.set_xticks(x); ax.set_xticklabels(LABELS[:len(rates)], rotation=30, ha="right", fontsize=9)
ax.set_ylabel("initial rate (µM C16 eq / min)")
ax.set_title("A. Initial rates, 150 s", fontsize=11)
ax.grid(axis="y", alpha=0.3)

ax = axes[1]
vals = prof[order].iloc[0].to_numpy()
errs = prof[[c + "_sigma" for c in order]].iloc[0].to_numpy() if (order[0] + "_sigma") in prof else None
xs = np.arange(len(order))
ax.bar(xs, vals, 0.68, yerr=errs, capsize=3, color=BAR)
ax.set_xticks(xs); ax.set_xticklabels(chain, rotation=90, fontsize=8)
ax.set_ylabel("fatty acid (µM)")
ax.set_title(f"B. Product profile, 720 s  (total {vals.sum():.1f} µM)", fontsize=11)
ax.grid(axis="y", alpha=0.3)

ax = axes[2]
scol = "C16 Equivalents (uM)"
ax.errorbar(tc["Time (s)"], tc[scol], yerr=tc.get(scol + "_sigma"), fmt="o-",
            color=LINE, ms=7, capsize=4, lw=1.6)
ax.set_xlabel("time (s)"); ax.set_ylabel("C16 equivalents (µM)")
ax.set_title("C. Time course, baseline", fontsize=11)
ax.grid(alpha=0.3)

fig.suptitle(f"Tier-1 rate design — {a.system}  (synthetic, noise = 10% + floor)", y=0.99)
fig.tight_layout(rect=(0, 0, 1, 0.95))
out = a.out or (PROJECT / a.root / f"tier1_rate_design_{a.system}.png")
fig.savefig(out, dpi=160)
print(f"wrote {out}")
print(f"\nA. rates (uM C16/min):")
for lab, v in zip(LABELS, rates[RCOL]): print(f"   {lab:<14}{v:>8.3f}")
print(f"\nB. profile total {vals.sum():.2f} uM across {len(order)} species")
print(f"C. time course {tc[scol].iloc[0]:.2f} -> {tc[scol].iloc[-1]:.2f} uM over {tc['Time (s)'].iloc[-1]:.0f} s")
