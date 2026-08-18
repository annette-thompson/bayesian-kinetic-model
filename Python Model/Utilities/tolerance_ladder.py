"""Find the loosest tolerance that gives finite gradients, per system.

The current rtol/atol was validated on the SMALLEST network and does not survive
scaling up: at 5 enzymes most settings give non-finite gradients, and at 6-8 the
probe fails outright. Tolerance therefore has to be chosen per system rather than
globally, and "loosest that works" is what keeps the big systems affordable --
tightening costs roughly 2.7x per decade of rtol.

Walks a ladder from loose to tight and STOPS at the first setting with all chains
finite, so a system that works at a cheap tolerance never pays for the tight ones.

Records every rung (not just the winner) so the failure boundary is visible -- it
has been non-monotonic before, and a single pass/fail hides that.

    python Utilities/tolerance_ladder.py --config <solver_params.json> [--floor F]
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_registry import config_label, project_root  # noqa: E402

# Loose -> tight. Each rung is ~2-3x more expensive than the last, so stopping
# early matters. Pairs chosen from what has actually been observed to work:
# 1e-4/1e-8 works at 3-4 enzymes, 1e-6/1e-10 was needed on the old Alpine stack.
LADDER = [
    ("1e-3", "1e-6"), ("1e-4", "1e-8"), ("1e-4", "1e-10"),
    ("1e-5", "1e-8"), ("1e-5", "1e-10"), ("1e-6", "1e-10"), ("1e-7", "1e-12"),
]


def probe(config: Path, rtol: str, atol: str, floor: float, python: str,
          device: str, timeout_s: float, max_steps: int) -> dict:
    cmd = [python, "-u", "Utilities/precision_probe.py", "--config", str(config),
           "--rtol", rtol, "--atol", atol, "--max-steps", str(max_steps),
           "--evals", "1", "--max-eval-seconds", "900",
           "--label", f"ladder {rtol}/{atol}"]
    if floor > 0:
        cmd += ["--floor", f"{floor:g}"]
    env = dict(os.environ)
    if device == "cpu":
        env["JAX_PLATFORMS"] = "cpu"
    t0 = time.perf_counter()
    try:
        p = subprocess.run(cmd, cwd=project_root(), env=env, timeout=timeout_s,
                           capture_output=True, text=True)
    except subprocess.TimeoutExpired:
        return {"rtol": rtol, "atol": atol, "status": "timeout",
                "wall_sec": round(time.perf_counter() - t0, 1)}
    lines = [l for l in p.stdout.splitlines() if l.startswith("RESULT ")]
    if not lines:
        tail = (p.stderr.strip().splitlines() or ["no RESULT"])[-25:]
        return {"rtol": rtol, "atol": atol, "status": "error",
                "wall_sec": round(time.perf_counter() - t0, 1),
                "error_tail": [l[:200] for l in tail]}
    r = json.loads(lines[-1][len("RESULT "):])
    r.update(rtol=rtol, atol=atol, wall_sec=round(time.perf_counter() - t0, 1),
             status=("ok" if r.get("n_chains_finite") == r.get("n_chains") else
                     ("error" if not r.get("ok") else "nonfinite")))
    return r


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--floor", type=float, default=0.0)
    ap.add_argument("--device", default="gpu", choices=["gpu", "cpu"])
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--max-steps", type=int, default=20000)
    ap.add_argument("--timeout", type=float, default=3600)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    cfg = Path(a.config)
    label = config_label(cfg)
    out_path = Path(a.out) if a.out else (project_root() / "Results" / "Benchmarks" / "tolerance_ladder.jsonl")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"=== {label}  floor={a.floor:g}  device={a.device} ===", flush=True)
    winner = None
    for rtol, atol in LADDER:
        r = probe(cfg, rtol, atol, a.floor, a.python, a.device, a.timeout, a.max_steps)
        r.update(config_label=label, floor=a.floor, device=a.device)
        with open(out_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(r) + "\n"); fh.flush()
        print(f"  rtol={rtol:<6} atol={atol:<7} {r['status']:<9} "
              f"chains_finite={r.get('n_chains_finite')} "
              f"ms/eval={r.get('ms_per_grad_eval')} ({r.get('wall_sec')}s)", flush=True)
        if r["status"] == "ok":
            winner = (rtol, atol, r.get("ms_per_grad_eval"))
            break     # loosest that works -- everything tighter only costs more
    if winner:
        print(f"  -> LOOSEST WORKING: rtol={winner[0]} atol={winner[1]}  {winner[2]} ms/eval", flush=True)
    else:
        print("  -> NO tolerance on the ladder gave all-finite gradients", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
