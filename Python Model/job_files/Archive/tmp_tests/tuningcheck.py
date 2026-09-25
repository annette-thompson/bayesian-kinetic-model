"""Did adaptation quality change with chain count? Each chain adapts independently
(vmapped window adaptation), so it should not -- this checks the runs rather than assuming."""
import glob, json, os, re
import numpy as np, zarr
BASE = "/projects/anth4580/Bayesian/Results/Chain Count Test"
print(f"{'chains':>7}{'step size: median':>19}{'geo-sd':>8}{'IQR ratio':>11}{'min':>9}{'max':>9}"
      f"{'accept last50: mean':>21}{'worst chain':>12}{'sampling accept min':>20}")
for d in sorted(glob.glob(f"{BASE}/*chains"), key=lambda s: int(re.search(r"_(\d+)chains", s).group(1))):
    z = zarr.open(os.path.join(d, "checkpoint", "draws.zarr"), mode="r")
    if "warmup_stats" not in z or "step_size" not in z["warmup_stats"]:
        continue
    n = int(re.search(r"_(\d+)chains", d).group(1))
    ss = np.asarray(z["warmup_stats"]["step_size"][:])[:, -1]      # final adapted value per chain
    acc = np.asarray(z["warmup_stats"]["acceptance_rate"][:])[:, -50:]
    per_chain_acc = acc.mean(axis=1)
    l = np.log(ss[ss > 0])
    geo_sd = float(np.exp(l.std()))
    q1, q3 = np.percentile(ss, [25, 75])
    samp = "-"
    if "sampling_stats" in z and "acceptance_rate" in z["sampling_stats"]:
        sa = np.asarray(z["sampling_stats"]["acceptance_rate"][:])
        if sa.size:
            samp = f"{sa.mean(axis=1).min():.3f}"
    print(f"{n:>7}{np.median(ss):>19.3f}{geo_sd:>8.2f}{q3/q1:>11.2f}{ss.min():>9.3f}{ss.max():>9.3f}"
          f"{per_chain_acc.mean():>21.3f}{per_chain_acc.min():>12.3f}{samp:>20}")
