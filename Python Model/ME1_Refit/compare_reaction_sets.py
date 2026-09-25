"""C20+unsat vs C20+unsat+FBinit against all four Figure S1 datasets.

FBinit adds one reaction to each of FabB and FabF -- the decarboxylative condensation that
releases C4_BKeAcACP -- i.e. FabH-independent initiation. Without it the model still forms
the activated FabB*/FabF* species but has no way to release product from them, so that
route is a dead end. This quantifies what the two extra reactions buy, dataset by dataset.

Usage: python compare_reaction_sets.py [--out fourway_reaction_sets.png]
"""
import argparse, sys
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, ".")
import me1_config as cfg, me1_model as mm, me1_objective as mo

SETS = [("C20+unsat", cfg.PROJECT / "Reactions/EC_FAS_ME1/C20+unsat", "#8a6bbf"),
        ("C20+unsat+FBinit", cfg.REACTIONS, "#2f6f9f")]
MEAS = "#c4643a"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="fourway_reaction_sets.png")
    a = ap.parse_args()
    rates, tc, prof = mo.load_data()
    held = pd.read_csv(cfg.DATA / "heldout_initial_rates.csv")
    p = dict(cfg.PUBLISHED)
    t_data = tc["time_min"].to_numpy() * 60.0
    times = np.unique(np.concatenate([t_data, [cfg.ENDPOINT_S]]))

    results = {}
    for name, path, colour in SETS:
        M = mm.ME1Model(reactions_dir=path)
        fitted = M.initial_rates_c16(p) / 2.5
        heldout = M.initial_rates_c16(p, cfg.HELDOUT_CONDITIONS) / 2.5
        sol = M.solve(p, M.condition_y0(cfg.RATE_CONDITIONS[0]), times)
        at = {float(t): i for i, t in enumerate(times)}
        curve = np.array([sol.c16_equivalents[at[float(t)]] for t in t_data])
        profile = sol.profile[at[float(cfg.ENDPOINT_S)]]
        obj = mo.evaluate(M, p, data=(rates, tc, prof))
        results[name] = dict(colour=colour, fitted=fitted, heldout=heldout,
                             curve=curve, profile=profile, obj=obj)
        print(f"{name:<20} obj1={obj.obj1:9.4f}  obj2={obj.obj2:9.3f}  obj3={obj.obj3:9.3f}")

    fig, axes = plt.subplots(2, 2, figsize=(15, 9.5))
    n_sets = len(SETS)

    def bars(ax, names, measured, se, key, title):
        x = np.arange(len(names)); w = 0.8 / (n_sets + 1)
        ax.bar(x - w, measured, w, yerr=se, capsize=3, color=MEAS, label="measured")
        for i, (name, _, colour) in enumerate(SETS):
            ax.bar(x + i * w, results[name][key], w, color=colour, label=name)
        ax.set_xticks(x); ax.set_xticklabels(names, rotation=35, ha="right", fontsize=7)
        ax.set_ylabel("initial rate (µM C16 eq/min)"); ax.set_title(title, fontsize=10)
        ax.grid(axis="y", alpha=0.3); ax.legend(fontsize=8)

    bars(axes[0][0], [c["label"] for c in cfg.RATE_CONDITIONS],
         rates["measured_rate_uM_C16_per_min"].to_numpy(), rates["standard_error"].to_numpy(),
         "fitted", "A. Initial rates (fitted)")
    bars(axes[1][1], [c["label"] for c in cfg.HELDOUT_CONDITIONS],
         held["measured_rate_uM_C16_per_min"].to_numpy(), held["standard_error"].to_numpy(),
         "heldout", "D. Initial rates (held out)")

    ax = axes[1][0]
    ax.plot(tc["time_min"], tc["c16_equivalents_uM"], "o", color=MEAS, ms=8, label="measured")
    for name, _, colour in SETS:
        ax.plot(tc["time_min"], results[name]["curve"], "s-", color=colour, label=name)
    ax.set_xlabel("time (min)"); ax.set_ylabel("C16 equivalents (µM)")
    ax.set_title("C. Reference time course (fitted)", fontsize=10)
    ax.grid(alpha=0.3); ax.legend(fontsize=8)

    ax = axes[0][1]
    labels = [f"C{r.chain}{':1' if r.unsaturated else ''}" for r in prof.itertuples()]
    x = np.arange(len(labels)); w = 0.8 / (n_sets + 1)
    # Target shape scaled to each model's own total, which is what obj3 compares against.
    for i, (name, _, colour) in enumerate(SETS):
        ax.bar(x + i * w, results[name]["profile"], w, color=colour, label=name)
    ref_total = results[SETS[-1][0]]["profile"].sum()
    ax.bar(x - w, ref_total * prof["mole_fraction"].to_numpy(), w, color=MEAS,
           label="target shape (scaled to FBinit total)")
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=90, fontsize=7)
    ax.set_ylabel("fatty acid (µM)"); ax.set_title("B. Product profile at 720 s (fitted)", fontsize=10)
    ax.grid(axis="y", alpha=0.3); ax.legend(fontsize=8)

    fig.suptitle("Reaction sets against all four Figure S1 datasets — published parameters, no refitting")
    fig.tight_layout(rect=(0, 0, 1, 0.97)); fig.savefig(a.out, dpi=160)
    print(f"\nwrote {a.out}")

    rows = []
    for name, _, _ in SETS:
        r = results[name]
        rows.append({"reaction set": name,
                     "obj1 fitted rates": r["obj"].obj1, "obj2 time course": r["obj"].obj2,
                     "obj3 profile": r["obj"].obj3,
                     "held-out mean abs err":
                         float(np.mean(np.abs(r["heldout"] - held["measured_rate_uM_C16_per_min"])))})
    print(pd.DataFrame(rows).to_string(index=False, float_format=lambda x: f"{x:.3f}"))


if __name__ == "__main__":
    main()
