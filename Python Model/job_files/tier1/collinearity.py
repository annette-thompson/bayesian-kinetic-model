"""Which pairs and trios of candidate parameters can the Tier-1 data tell apart?

Collinearity index (Brun, Reichert & Kunsch 2001, Water Resour. Res. 37:1015):
  s_p    = sensitivity of every Tier-1 data point to ln(multiplier of p), divided by that
           point's sigma (central difference, +/-0.05 log10, the model's own solver)
  s~_p   = s_p / ||s_p||
  gamma_K = 1 / sqrt(smallest eigenvalue of S~_K^T S~_K)   for a subset K
gamma = 1 means the parameters move the data in orthogonal directions; it grows as some
combination of them changes nothing the data can see. Brun et al. treat gamma > 10-15 as
poorly identifiable jointly. For a pair, gamma = 1/sqrt(1 - |cos|) with cos the angle
between the two sensitivity vectors. This is the data-design notion of interaction that
governs joint sampling (ridges such as c3/d1), not Morris's sigma/mu* model nonlinearity.

||s_p|| is reported too: a parameter the data barely see can look "independent" of
everything while still being unidentifiable on its own.

Data: Data/Tier1/Chain_<system>/ (C16 Equivalents time series + per-species endpoints at
720 s, sigma = 0.01 uM + 10% per point), clean model predictions at nominal truth.

Usage: python collinearity.py C12 [--params a1,a2,b3,c1,c2,c3,d1,d2,f] [--data_dir DIR --tag NAME]
"""
import argparse
import itertools
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "bench"))
sys.path.insert(0, str(HERE.parent / "multiparam_tests"))

import numpy as np
import info_vs_conditions as ivc
from reaction_model_builder import set_scaling_group_values

FD_STEP = 0.05  # log10


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("system")
    ap.add_argument("--params", default="a1,a2,b3,c1,c2,c3,d1,d2,f")
    ap.add_argument("--data_dir", default=None, help="Tier-1-format data dir (default Data/Tier1/Chain_<system>)")
    ap.add_argument("--tag", default="", help="suffix for the output json, e.g. shared14")
    ap.add_argument("--max_k", type=int, default=3, help="largest subset size to score (2=pairs, 3=trios, ...)")
    a = ap.parse_args()

    sys_, targets, ctrl = ivc.build(a.system)
    cfg = json.loads(ivc.config_path(a.system).read_text())
    sg = {k: float(v) for k, v in cfg["scaling_groups"].items()}
    params = [p for p in a.params.split(",") if p in sg]
    absent = [p for p in a.params.split(",") if p not in sg]
    idx = [sys_.index_of[t] for t in targets]
    y0s, times, sig_ep, sig_ts, weights = ivc.load_tier1(sys_, a.system, targets, a.data_dir)

    def predict(group=None, mult=1.0):
        o = dict(sg)
        if group is not None:
            o[group] = ivc.group_value_for(group, mult)
        th = set_scaling_group_values(sys_.theta, sys_.params, o)
        ts, ep = ivc.predict_tier1(sys_, th, y0s, times, ctrl, idx, weights)
        if ts is None or any(e is None for e in ep):
            raise RuntimeError(f"solve failed for {group} x{mult}")
        return np.concatenate([ts / sig_ts, (np.stack(ep) / sig_ep).ravel()])

    h = FD_STEP * np.log(10.0)
    S = np.stack([(predict(p, 10 ** FD_STEP) - predict(p, 10 ** -FD_STEP)) / (2 * h) for p in params], axis=1)
    norms = np.linalg.norm(S, axis=0)
    St = S / norms

    def gamma(K):
        ix = [params.index(k) for k in K]
        lam = np.linalg.eigvalsh(St[:, ix].T @ St[:, ix])[0]
        return float(1.0 / np.sqrt(lam)) if lam > 1e-12 else float("inf")

    subsets = {k: sorted(((gamma(K), K) for K in itertools.combinations(params, k)), key=lambda x: x[0])
               for k in range(2, min(a.max_k, len(params)) + 1)}
    pairs, trios = subsets.get(2, []), subsets.get(3, [])
    cos = St.T @ St

    print(f"=== {a.system}: {len(S)} weighted data points; params {params}" + (f" (absent: {absent})" if absent else ""))
    print("  sensitivity strength ||s|| (relative to strongest): " +
          ", ".join(f"{p} {n / norms.max():.3f}" for p, n in sorted(zip(params, norms), key=lambda x: -x[1])))
    print("  pairs, least interacting first (gamma, |cos|):")
    for g, K in pairs:
        print(f"    {'+'.join(K):<8} gamma {g:8.2f}  |cos| {abs(cos[params.index(K[0]), params.index(K[1])]):.3f}")
    for k in sorted(subsets):
        if k == 2:
            continue
        name = {3: "trios", 4: "quads", 5: "quints", 6: "sextets"}.get(k, f"{k}-sets")
        print(f"  {name}, least interacting first (gamma):")
        for g, K in subsets[k][:10]:
            print(f"    {'+'.join(K):<14} gamma {g:8.2f}")
        print(f"  ...most interacting {name}:", ", ".join(f"{'+'.join(K)} {g:.1f}" for g, K in subsets[k][-3:]))
    out = HERE / f"collinearity_{a.system}{'_' + a.tag if a.tag else ''}.json"
    out.write_text(json.dumps(dict(
        system=a.system, params=params, absent=absent, n_points=int(len(S)),
        sensitivity_norm=dict(zip(params, norms.tolist())),
        pairs=[dict(params=list(K), gamma=g, abs_cos=float(abs(cos[params.index(K[0]), params.index(K[1])]))) for g, K in pairs],
        trios=[dict(params=list(K), gamma=g) for g, K in trios],
        subsets={str(k): [dict(params=list(K), gamma=g) for g, K in v] for k, v in subsets.items()}), indent=2))
    print(f"wrote {out}\nDONE")


if __name__ == "__main__":
    main()
