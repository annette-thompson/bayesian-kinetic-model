"""Initial rates (datasets A and D) with the model at two acetyl-CoA levels.

The model is normally solved at the in-vivo-representative basis -- 0.5 mM malonyl-CoA and
0.5 mM acetyl-CoA -- for every condition, which is what lets one parameter set answer to
both the in vitro rates and the in vivo profile. The in vitro experiments in A and D were
actually run at 0.1 mM acetyl-CoA. This plots both so the size of that choice is visible:

  500/500   the standard basis (malonyl-CoA 500, acetyl-CoA 500)
  500/100   malonyl-CoA held, acetyl-CoA dropped to what the experiment used

The three "no AcCoA" conditions in A are unaffected -- they set acetyl-CoA to 0 explicitly.

Usage: python -u plot_rates_accoa.py [--reactions C20+unsat+FBinit]
"""
import argparse, sys
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, ".")
import me1_config as cfg, me1_model as mm, me1_objective as mo

ap = argparse.ArgumentParser()
ap.add_argument("--reactions", default="C20+unsat+FBinit")
ap.add_argument("--out", default="rates_accoa_500_vs_100.png")
a = ap.parse_args()

rx = cfg.PROJECT / "Reactions" / "EC_FAS_ME1" / a.reactions
M = mm.ME1Model(reactions_dir=rx)
published = dict(cfg.PUBLISHED)
rates_A, _, _ = mo.load_data()
rates_D = pd.read_csv(cfg.DATA / "heldout_initial_rates.csv")
rates_25 = pd.read_csv(cfg.DATA / "heldout_2025_initial_rates.csv")

LEVELS = [(500.0, "500/500 (standard basis)", "#2f6f9f"),
          (100.0, "500/100 (as the experiment ran)", "#8a6bbf")]
pred = {}
for level, label, _ in LEVELS:
    cfg.BASE_SUBSTRATES["C2_AcCoA"] = level          # reaches every condition that
    pred[label] = {                                   # does not set acetyl-CoA itself
        "A": M.initial_rates_c16(published, cfg.RATE_CONDITIONS) / 2.5,
        "D": M.initial_rates_c16(published, cfg.HELDOUT_CONDITIONS) / 2.5,
        "2025": M.initial_rates_c16(published, cfg.HELDOUT_2025_CONDITIONS) / 2.5,
    }
cfg.BASE_SUBSTRATES["C2_AcCoA"] = 500.0              # restore

panels = [("A", [c["label"] for c in cfg.RATE_CONDITIONS],
           rates_A["measured_rate_uM_C16_per_min"].to_numpy(),
           rates_A["standard_error"].to_numpy(), "A. Initial rates (fitted)"),
          ("D", [c["label"] for c in cfg.HELDOUT_CONDITIONS],
           rates_D["measured_rate_uM_C16_per_min"].to_numpy(),
           rates_D["standard_error"].to_numpy(), "D. Initial rates (held out)"),
          ("2025", [c["label"] for c in cfg.HELDOUT_2025_CONDITIONS],
           rates_25["measured_rate_uM_C16_per_min"].to_numpy(),
           rates_25["standard_error"].to_numpy(),
           "2025 rates (held out; experiment at 100 uM acetyl-CoA)")]

fig, axes = plt.subplots(1, 3, figsize=(20, 5.4), gridspec_kw={"width_ratios": [7, 6, 4]})
w = 0.8 / 3
for ax, (key, names, meas, se, title) in zip(axes, panels):
    x = np.arange(len(names))
    ax.bar(x - w, meas, w, yerr=se, capsize=3, color="#c4643a", label="measured")
    for i, (_, label, colour) in enumerate(LEVELS):
        ax.bar(x + i * w, pred[label][key], w, color=colour, label=f"model, {label}")
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=35, ha="right", fontsize=8)
    ax.set_ylabel("initial rate (µM C16 eq/min)")
    bits = []
    for _, label, _c in LEVELS:
        p = pred[label][key]
        bits.append(f"{label.split()[0]}: r={np.corrcoef(meas, p)[0,1]:+.2f}, "
                    f"err={np.mean(np.abs(p-meas)):.2f}")
    ax.set_title(f"{title}\n" + "   ".join(bits), fontsize=9)
    ax.grid(axis="y", alpha=0.3); ax.legend(fontsize=8)
fig.suptitle(f"{a.reactions}: initial rates at two acetyl-CoA levels (published parameters)")
fig.tight_layout(rect=(0, 0, 1, 0.94)); fig.savefig(a.out, dpi=160)
print(f"wrote {a.out}\n")
for key, names, meas, se, _ in panels:
    print(f"--- dataset {key}")
    df = pd.DataFrame({"condition": names, "measured": meas})
    for _, label, _c in LEVELS:
        df[label.split()[0]] = pred[label][key]
    print(df.to_string(index=False, float_format=lambda v: f"{v:.3f}")); print()
