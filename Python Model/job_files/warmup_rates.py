#!/usr/bin/env python3
"""Per-system draw rates and warmup ETAs for the live a1 [0.1, 10] inference set.

Companion to warmup_status.py, which answers "how far along is each system".
This one answers "how fast is each system going, and when will warmup finish",
which is the question that decides whether the large chain systems are viable
inside 24 h SLURM segments at all.

Rates come from checkpoint/progress_log.jsonl, written every
`checkpoint_every_steps` draws (currently 5). Two rates are reported because
they say different things:

  last5   seconds/draw over the most recent checkpoint interval. This is the
          rate that matters for the ETA, because warmup gets *faster* as dual
          averaging settles the step size and tree depth falls (C4_NoFB went
          32 -> 16 s/draw over its first 25 draws).
  avg     seconds/draw across every checkpoint so far, which stays pessimistic
          early on for exactly that reason.

Neither includes JIT compile time: the first checkpoint is only written after
compilation, so `elapsed - (draws * avg)` is roughly the compile cost.

Usage:
    python3 warmup_rates.py             # table
    python3 warmup_rates.py --parsable  # system|node|draws|last5|avg|eta_h|gpu
"""
import json
import os
import subprocess
import sys

BASE = "/projects/anth4580/Bayesian/Results/Chain Scaling Tests"
SUFFIX = "a1 tightest"
N_TUNE = 1000
ORDER = ["C4_NoFB", "C6", "C8", "C10", "C12", "C12+unsat", "C14", "C14+unsat",
         "C16", "C16+unsat", "C18", "C18+unsat", "C20", "C20+unsat"]


def nodes():
    """Map system name -> node, via squeue. Never fatal: rates work without it."""
    out = {}
    try:
        # module load is required -- a bare squeue targets Alpine, and `-M blanca`
        # needs the accounting DB, which has gone down before.
        r = subprocess.run(
            ["bash", "-lc",
             "module load slurm/blanca >/dev/null 2>&1; "
             "squeue -u anth4580 --format='%j|%R' --noheader"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        for line in r.stdout.decode().strip().splitlines():
            if "|" in line:
                name, reason = line.split("|", 1)
                out[name.strip().replace("a1t_", "")] = reason.strip()
    except Exception as exc:  # noqa: BLE001 - diagnostics only
        print(f"WARNING: squeue unavailable: {exc}", file=sys.stderr)
    return out


def read_rows(system):
    p = os.path.join(BASE, f"Chain {system} - {SUFFIX}", "checkpoint",
                     "progress_log.jsonl")
    if not os.path.exists(p):
        return []
    rows = []
    for line in open(p):
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except ValueError:
            continue  # a torn final line while the sampler is mid-write
    return rows


def total(row):
    return row["warmup_done"] + row["sampling_done"]


def stats(system, node):
    rows = read_rows(system)
    d = dict(system=system, node=node, draws=0, last5=None, avg=None,
             eta_h=None, gpu="-", phase="-")
    if not rows:
        return d
    d["draws"] = total(rows[-1])
    d["gpu"] = rows[-1].get("device", "?").replace("NVIDIA ", "")
    d["phase"] = rows[-1].get("phase", "-")
    if len(rows) < 2:
        return d
    dn = total(rows[-1]) - total(rows[-2])
    if dn > 0:
        d["last5"] = (rows[-1]["t"] - rows[-2]["t"]) / dn
    dn_all = total(rows[-1]) - total(rows[0])
    if dn_all > 0:
        d["avg"] = (rows[-1]["t"] - rows[0]["t"]) / dn_all
    rate = d["last5"] or d["avg"]
    if rate and rows[-1]["warmup_done"] < N_TUNE:
        d["eta_h"] = (N_TUNE - rows[-1]["warmup_done"]) * rate / 3600.0
    elif rate:
        d["eta_h"] = 0.0
    return d


def main():
    node_map = nodes()
    rows = [stats(s, node_map.get(s, "?")) for s in ORDER]

    if "--parsable" in sys.argv:
        for r in rows:
            def f(v, spec="{:.1f}"):
                return spec.format(v) if v is not None else "-"
            print(f"{r['system']}|{r['node']}|{r['draws']}|{f(r['last5'])}|"
                  f"{f(r['avg'])}|{f(r['eta_h'], '{:.2f}')}|{r['gpu']}")
        return

    W = (12, 13, 8, 11, 10, 10, 11)
    hdr = ("SYSTEM", "NODE", "DRAWS", "last5 s/dr", "avg s/dr", "WARM ETA", "GPU")
    print()
    print(f"a1 [0.1,10]  tune={N_TUNE}  draws=5000  chains=8  ckpt_every=5")
    print("=" * sum(W) + "=" * 6)
    print("  ".join(h.ljust(w) for h, w in zip(hdr, W)))
    print("-" * (sum(W) + 6))
    reported = 0
    for r in rows:
        if r["draws"]:
            reported += 1
        cells = [
            r["system"],
            r["node"][:13],
            str(r["draws"]) if r["draws"] else "-",
            f"{r['last5']:.1f}" if r["last5"] is not None else "-",
            f"{r['avg']:.1f}" if r["avg"] is not None else "-",
            f"{r['eta_h']:.1f}h" if r["eta_h"] is not None else "-",
            r["gpu"][:11],
        ]
        print("  ".join(c.ljust(w) for c, w in zip(cells, W)))
    print("-" * (sum(W) + 6))
    etas = [r["eta_h"] for r in rows if r["eta_h"] is not None]
    tail = f" | slowest warmup ETA {max(etas):.1f}h" if etas else ""
    print(f"{reported}/{len(rows)} checkpointed{tail}")
    print()


if __name__ == "__main__":
    main()
