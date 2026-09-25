"""Run the model over every real experimental condition and plot it against the measurements.

No fitting: nominal parameters, so every disagreement is the current model's, not an
artefact of the inference pipeline. This is the "where do we actually stand" figure set.

Two raw datasets, each with its own buffer and its own observable:

  ME1_Dataset_S1_Kinetics.csv   average initial rate, uM C16 equivalents per MINUTE, with
                                a standard error and n per condition. Read off the same
                                150 s window the assay uses, so the model's number is
                                C16Equiv(150)/150 converted to per-minute.
  ME1_Dataset_S1_GCMS.csv       per-chain-length fatty acid in uM at the 720 s endpoint,
                                plus Total Prod., Palmitic Acid Equivalents (= C16
                                equivalents) and Avg Chain Length, each with a standard
                                error.

Two things about the GC/MS data shape the comparison and are handled explicitly rather than
silently:

  * C20 is not measured -- there is too little to read -- so measured C20 is taken as 0.
    The model's C20 is still plotted, since a model predicting appreciable C20 against an
    unreadable measurement is a real disagreement, not a missing value.
  * The measurement does not split saturated from unsaturated, so each measured bar is one
    number per chain length. The model's bar is drawn stacked, sat below unsat, so the
    total is comparable at a glance while the split the model claims stays visible.

Usage:
  python compare_model_to_experiment.py                      # C20+unsat
  python compare_model_to_experiment.py --system C16+unsat --out_dir "Results/..."
"""
import argparse
import json
import re
import sys
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent.parent
sys.path.insert(0, str(PROJECT / "Utilities"))

import diffrax as dfrx
import jax.numpy as jnp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import generate_chain_data as gcd

EXPERIMENTAL = PROJECT / "Data" / "Experimental"
RATE_TIME = 150.0          # the kinetics assay's own early window
END_TIME = 720.0           # the GC/MS endpoint
MAX_STEPS = 50_000
SEC_PER_MIN = 60.0

# Data column -> model species. The CoA and ACP names differ between the datasheet and the
# reaction files. NADPH is deliberately NOT mapped: the datasets run at 1300/2600 uM, but
# the cofactors are held at the model's usual 1000/1000 (per instruction), matching every
# existing config, which treats NADPH/NADH as a fixed, non-limiting pool rather than a
# swept input.
SPECIES_FOR = {
    "[FabA]": "FabA", "[FabB]": "FabB", "[FabD]": "FabD", "[FabF]": "FabF",
    "[FabG]": "FabG", "[FabH]": "FabH", "[FabI]": "FabI", "[FabZ]": "FabZ",
    "[TesA]": "TesA", "[Holo-ACP]": "ACP",
    "[Malonyl CoA]": "C3_MalCoA", "[Acetyl CoA]": "C2_AcCoA",
}
MEASURED_CHAINS = [4, 6, 8, 10, 12, 14, 16, 18]      # C20 is below the GC/MS read limit

# Two substrate treatments, solved side by side. The datasets were run at different CoA
# levels from each other (acetyl-CoA 100 uM in the kinetics assay, 300 in the GC/MS; malonyl
# -CoA 500 and 1500), while the model's standard initial condition is 500/500 for both. The
# cofactors are already held at the model's 1000/1000, so this asks the matching question of
# the substrates: how much of the model/measurement gap is the substrate concentration
# rather than the chemistry.
SUBSTRATES = [
    ("as measured", None),
    ("500/500", {"C3_MalCoA": 500.0, "C2_AcCoA": 500.0}),
]
INK, MEAS = "#1f2933", "#c4643a"
# One hue family per reaction set, dark/light within it per substrate treatment; the
# unsaturated half of each stacked profile bar is the lighter tint of its own colour.
PALETTE = {0: ("#2f6f9f", "#8fc0e0", "#5d97c0", "#b6d8ee"),
           1: ("#3d7d54", "#a8ceaf", "#6da583", "#c9e2d0")}


def load_fa_conc():
    spec = spec_from_file_location("FA_conc", PROJECT / "Calculation Files" / "Full_FAS" / "FA_conc.py")
    mod = module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def safe(text):
    """Filename-safe variant label."""
    return re.sub(r"[^A-Za-z0-9]+", "_", text).strip("_")


def read_raw(name):
    """Both sheets carry a units row under the header and a UTF-8 BOM."""
    df = pd.read_csv(EXPERIMENTAL / name, encoding="utf-8-sig", skiprows=[1])
    return df.dropna(how="all").reset_index(drop=True)


def solve_at(sys_, y0, ts):
    sol = dfrx.diffeqsolve(
        dfrx.ODETerm(sys_.network), dfrx.Kvaerno5(),
        t0=0.0, t1=float(max(ts)), dt0=1e-6,
        y0=jnp.asarray(y0, dtype=jnp.float64), args=sys_.theta,
        saveat=dfrx.SaveAt(ts=jnp.asarray(ts, dtype=jnp.float64)),
        stepsize_controller=dfrx.PIDController(
            rtol=sys_.rtol, atol=sys_.atol,
            pcoeff=sys_.pcoeff, icoeff=sys_.icoeff, dcoeff=sys_.dcoeff),
        max_steps=MAX_STEPS, throw=False)
    steps = int(np.asarray(sol.stats["num_steps"]))
    ok = steps < MAX_STEPS and bool(sol.result == dfrx.RESULTS.successful)
    return (np.asarray(sol.ys) if ok else None), steps


def y0_for(sys_, row, overrides=None):
    """Initial state from one data row, with an optional substrate treatment applied last
    so it wins over whatever the datasheet recorded."""
    y = sys_.y0()
    for col, species in SPECIES_FOR.items():
        if col in row.index and species in sys_.index_of and pd.notna(row[col]):
            y[sys_.index_of[species]] = float(row[col])
    for species, conc in (overrides or {}).items():
        if species in sys_.index_of:
            y[sys_.index_of[species]] = float(conc)
    return y


def chain_map(species_names):
    """{chain length: (saturated species or None, unsaturated species or None)}."""
    out = {}
    for s in species_names:
        m = re.fullmatch(r"^C(\d+)_FA(_unsat)?$", s)
        if m:
            n = int(m.group(1))
            sat, uns = out.get(n, (None, None))
            out[n] = (s, uns) if not m.group(2) else (sat, s)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--systems", default="C20+unsat,C20+unsat+FBinit",
                    help="comma-separated reaction sets; each is crossed with both substrate treatments")
    ap.add_argument("--out_dir", default="Results/Experimental Comparison")
    a = ap.parse_args()

    out_dir = PROJECT / a.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    system_names = [x.strip() for x in a.systems.split(",") if x.strip()]
    fa = load_fa_conc()

    built = []
    for name in system_names:
        rx = PROJECT / "Reactions" / "EC_FAS_ME1" / name
        if not rx.is_dir():
            raise SystemExit(f"no reactions directory at {rx}")
        groups = gcd.discover_scaling_groups(rx)
        s_ = gcd.ChainSystem(rx, 1e-8, 1e-10,
                             scaling_group_overrides=gcd.nominal_scaling_group_overrides(sorted(groups)))
        built.append((name, s_, chain_map(s_.species)))
        print(f"{name}: {len(s_.species)} species, chain lengths {sorted(chain_map(s_.species))}")

    # Every (reaction set x substrate treatment) pair, in one flat list so the plots can
    # simply iterate; colour comes from the reaction set, shade from the treatment.
    VARIANTS = [(f"{name} / {sname}", s_, ch, ov, PALETTE[si % len(PALETTE)][0 if ti == 0 else 2],
                 PALETTE[si % len(PALETTE)][1 if ti == 0 else 3])
                for si, (name, s_, ch) in enumerate(built)
                for ti, (sname, ov) in enumerate(SUBSTRATES)]
    vnames = [v[0] for v in VARIANTS]
    print(f"\n{len(VARIANTS)} model variants: {vnames}")

    def totals_for(chains, sys_, conc_row):
        """(per-chain sat, per-chain unsat, total uM, C16 equivalents, avg chain length)."""
        sat = {n: (float(conc_row[sys_.index_of[s]]) if s else 0.0) for n, (s, _) in chains.items()}
        uns = {n: (float(conc_row[sys_.index_of[u]]) if u else 0.0) for n, (_, u) in chains.items()}
        _ = sys_
        per = {n: sat[n] + uns[n] for n in chains}
        tot = sum(per.values())
        c16 = sum(n / 16.0 * v for n, v in per.items())
        acl = (sum(n * v for n, v in per.items()) / tot) if tot > 0 else float("nan")
        return sat, uns, tot, c16, acl

    # ---------------- kinetics: initial rate ----------------
    kin = read_raw("ME1_Dataset_S1_Kinetics.csv")
    krows = []
    for _, r in kin.iterrows():
        row = {"system": str(r["System"]), "measured": float(r["Average Initial Rate"]),
               "se": float(r["Standard error"])}
        for vname, s_, ch, ov, _c, _u in VARIANTS:
            ys, steps = solve_at(s_, y0_for(s_, r, ov), [RATE_TIME])
            row[f"model [{vname}]"] = (totals_for(ch, s_, ys[0])[3] / RATE_TIME * SEC_PER_MIN
                                       if ys is not None else np.nan)
            row[f"steps [{vname}]"] = steps
        krows.append(row)
    kdf = pd.DataFrame(krows)

    # One panel per variant, each against the same measured bars, so the four are read
    # side by side rather than as eight bars crowded onto one axis.
    nv = len(VARIANTS)
    ncol = 2; nrow = int(np.ceil(nv / ncol))
    ymax = float(np.nanmax([kdf["measured"].max()] + [kdf[f"model [{v}]"].max() for v in vnames])) * 1.12
    fig, axes = plt.subplots(nrow, ncol, figsize=(9.5 * ncol, 4.6 * nrow), squeeze=False)
    for ax, (vname, _s, _ch, _ov, col, _u) in zip(axes.ravel(), VARIANTS):
        x = np.arange(len(kdf)); w = 0.4
        ax.bar(x - w / 2, kdf["measured"], w, yerr=kdf["se"], capsize=3, color=MEAS, label="measured")
        ax.bar(x + w / 2, kdf[f"model [{vname}]"], w, color=col, label="model")
        r = kdf["measured"].corr(kdf[f"model [{vname}]"])
        err = float(np.nanmean(np.abs(kdf[f"model [{vname}]"] - kdf["measured"])))
        ax.set_xticks(x); ax.set_xticklabels(kdf["system"], rotation=32, ha="right", fontsize=7)
        ax.set_ylim(min(0, float(kdf["measured"].min()) * 1.2), ymax)
        ax.set_ylabel("initial rate (µM C16 eq / min)", fontsize=9)
        ax.set_title(f"{vname}   (r = {r:+.2f}, mean abs err {err:.2f})", fontsize=10)
        ax.axhline(0, color=INK, lw=0.8); ax.legend(fontsize=8); ax.grid(axis="y", alpha=0.3)
    for ax in axes.ravel()[nv:]:
        ax.axis("off")
    fig.suptitle("Initial rate vs measurement, per model variant (nominal parameters, no fitting)")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_dir / "kinetics_initial_rate.png", dpi=160); plt.close(fig)

    # ---------------- GC/MS ----------------
    gc = read_raw("ME1_Dataset_S1_GCMS.csv")
    grows, profiles = [], []
    for _, r in gc.iterrows():
        meas = {n: (float(r[f"C{n}"]) if f"C{n}" in r.index and pd.notna(r[f"C{n}"]) else 0.0)
                for n in MEASURED_CHAINS}
        meas_se = {n: (float(r[f"C{n} SE"]) if f"C{n} SE" in r.index and pd.notna(r[f"C{n} SE"]) else 0.0)
                   for n in MEASURED_CHAINS}
        prof = {"system": str(r["System"]), "meas": meas, "meas_se": meas_se, "model": {}}
        row = {"system": str(r["System"]),
               "meas_total": float(r["Total Prod."]), "meas_total_se": float(r["Total Prod. SE"]),
               "meas_c16eq": float(r["Palmitic Acid Equivalents"]),
               "meas_c16eq_se": float(r["Palmitic Acid Equivalents SE"]),
               "meas_acl": float(r["Avg Chain Length"]), "meas_acl_se": float(r["Avg Chain Length SE"])}
        for vname, s_, ch, ov, _c, _u in VARIANTS:
            ys, steps = solve_at(s_, y0_for(s_, r, ov), [END_TIME])
            row[f"steps [{vname}]"] = steps
            if ys is None:
                for k in ("model_total", "model_c16eq", "model_acl"):
                    row[f"{k} [{vname}]"] = np.nan
                continue
            sat, uns, tot, c16, acl = totals_for(ch, s_, ys[0])
            prof["model"][vname] = (sat, uns)
            row[f"model_total [{vname}]"] = tot
            row[f"model_c16eq [{vname}]"] = c16
            row[f"model_acl [{vname}]"] = acl
        profiles.append(prof if prof["model"] else None)
        grows.append(row)
    gdf = pd.DataFrame(grows)

    # One profile FIGURE per variant: 14 conditions each, model stacked sat + unsat.
    good = [p for p in profiles if p]
    all_chains = sorted(set(MEASURED_CHAINS) | {n for _v, _s, ch, *_ in VARIANTS for n in ch})
    for vi, (vname, _s, _ch, _ov, col, ucol) in enumerate(VARIANTS):
        ncolp = 4; nrowp = int(np.ceil(len(good) / ncolp))
        fig, axes = plt.subplots(nrowp, ncolp, figsize=(4.1 * ncolp, 2.9 * nrowp), squeeze=False)
        for ax, p in zip(axes.ravel(), good):
            if vname not in p["model"]:
                ax.axis("off"); continue
            xs = np.arange(len(all_chains)); w = 0.4
            sat, uns = p["model"][vname]
            m = [p["meas"].get(n, 0.0) for n in all_chains]
            e = [p["meas_se"].get(n, 0.0) for n in all_chains]
            sv = [sat.get(n, 0.0) for n in all_chains]
            uv = [uns.get(n, 0.0) for n in all_chains]
            ax.bar(xs - w / 2, m, w, yerr=e, capsize=2, color=MEAS, label="measured")
            ax.bar(xs + w / 2, sv, w, color=col, label="model sat")
            ax.bar(xs + w / 2, uv, w, bottom=sv, color=ucol, label="model unsat")
            ax.set_xticks(xs); ax.set_xticklabels([f"C{n}" for n in all_chains], fontsize=7)
            ax.set_title(p["system"][:46], fontsize=8)
            ax.tick_params(labelsize=7); ax.grid(axis="y", alpha=0.25)
        for ax in axes.ravel()[len(good):]:
            ax.axis("off")
        axes[0][0].set_ylabel("fatty acid (µM)")
        h, l = axes[0][0].get_legend_handles_labels()
        fig.legend(h, l, loc="lower center", ncol=3, frameon=False)
        fig.suptitle(f"Chain-length profiles at {END_TIME:g} s -- {vname}\n"
                     f"(model bars stacked sat + unsat; C20 not measured)", fontsize=11)
        fig.tight_layout(rect=(0, 0.035, 1, 0.95))
        fig.savefig(out_dir / f"gcms_profiles_{vi}_{safe(vname)}.png", dpi=160); plt.close(fig)

    # Aggregates: one row per quantity, one column per variant.
    quants = [("model_total", "meas_total", "meas_total_se", "total product (µM)"),
              ("model_c16eq", "meas_c16eq", "meas_c16eq_se", "C16 equivalents (µM)"),
              ("model_acl", "meas_acl", "meas_acl_se", "avg chain length")]
    fig, axes = plt.subplots(len(quants), nv, figsize=(4.3 * nv, 4.0 * len(quants)), squeeze=False)
    for qi, (mc, oc, ec, lab) in enumerate(quants):
        cols = [f"{mc} [{v}]" for v in vnames]
        lo = float(np.nanmin([gdf[oc].min()] + [gdf[c].min() for c in cols]))
        hi = float(np.nanmax([gdf[oc].max()] + [gdf[c].max() for c in cols]))
        pad = 0.08 * (hi - lo or 1.0)
        for vi, (vname, _s, _ch, _ov, col, _u) in enumerate(VARIANTS):
            ax = axes[qi][vi]
            d = gdf.dropna(subset=[f"{mc} [{vname}]", oc])
            ax.errorbar(d[oc], d[f"{mc} [{vname}]"], xerr=d[ec], fmt="o", color=col,
                        ecolor=INK, elinewidth=0.7, capsize=2, ms=6)
            ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], ls="--", lw=1, color=INK)
            ax.set_xlim(lo - pad, hi + pad); ax.set_ylim(lo - pad, hi + pad)
            r = d[f"{mc} [{vname}]"].corr(d[oc])
            ratio = float(np.median(d[f"{mc} [{vname}]"] / d[oc]))
            ax.set_title(f"{vname}\nr = {r:+.2f}, model/measured {ratio:.2f}", fontsize=9)
            ax.set_xlabel(f"measured {lab}", fontsize=8)
            if vi == 0:
                ax.set_ylabel(f"model {lab}", fontsize=9)
            ax.grid(alpha=0.3); ax.tick_params(labelsize=8)
    fig.suptitle("GC/MS aggregates per model variant (dashed = perfect agreement)")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_dir / "gcms_aggregates.png", dpi=160); plt.close(fig)

    # ---------------- reference time course ----------------
    ts_file = EXPERIMENTAL / "time_vs_conc.csv"
    if ts_file.exists():
        ts = pd.read_csv(ts_file)
        ref = kin.iloc[0]
        grid = np.linspace(1.0, END_TIME, 200)
        fig, ax = plt.subplots(figsize=(8.5, 5.2))
        for vname, s_, ch, ov, col, _u in VARIANTS:
            ys, _ = solve_at(s_, y0_for(s_, ref, ov), grid)
            if ys is not None:
                ax.plot(grid, [totals_for(ch, s_, y)[3] for y in ys], color=col, lw=2, label=vname)
        ax.plot(ts["Time (s)"], ts["C16 Equivalents (uM)"], "o", color=MEAS, ms=8,
                label="measured", zorder=5)
        ax.set_xlabel("time (s)"); ax.set_ylabel("C16 equivalents (µM)")
        ax.set_title("Reference time course, all model variants")
        ax.legend(fontsize=8); ax.grid(alpha=0.3)
        fig.tight_layout(); fig.savefig(out_dir / "reference_timecourse.png", dpi=160); plt.close(fig)

    kdf.to_csv(out_dir / "kinetics_comparison.csv", index=False)
    gdf.to_csv(out_dir / "gcms_comparison.csv", index=False)
    print("\n--- initial rate (µM C16/min) ---")
    print(kdf[["system", "measured", "se"] + [f"model [{v}]" for v in vnames]]
          .to_string(index=False, max_colwidth=30))
    print("\n--- agreement per variant ---")
    for v in vnames:
        r = kdf["measured"].corr(kdf[f"model [{v}]"])
        err = float(np.nanmean(np.abs(kdf[f"model [{v}]"] - kdf["measured"])))
        bits = [f"kinetics r={r:+.3f} err={err:.2f}"]
        for mc, oc, _e, lab in quants:
            d = gdf.dropna(subset=[f"{mc} [{v}]", oc])
            bits.append(f"{lab.split()[0]} r={d[f'{mc} [{v}]'].corr(d[oc]):+.2f} "
                        f"ratio={float(np.median(d[f'{mc} [{v}]'] / d[oc])):.2f}")
        print(f"  {v:<34} " + "  ".join(bits))
    print(f"\nwrote figures + csv to {out_dir}")
    (out_dir / "summary.json").write_text(json.dumps(
        {"variants": vnames, "kinetics": krows, "gcms": grows}, indent=1, default=float) + "\n")


if __name__ == "__main__":
    main()
