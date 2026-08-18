"""Drive the benchmark matrix on one machine.

Axes: config x precision (32/64) x tolerance (rtol:atol) x floor x device. Every one
of them is part of the cell key, because resume-by-key treats a missing axis as
"already measured" and silently skips a cell it never ran.


Stdlib only, and it must stay that way. jax_enable_x64 is a global fixed at import
time, so every measurement needs its own interpreter; this parent shells out to
precision_probe.py per cell and must never import jax itself.

Results append to Results/Benchmarks/timing__<machine>__<device>.jsonl, one JSON
object per line, flushed immediately. Append-only plus resume-by-key is what makes
this survivable: Alpine jobs hit a 24 h cap mid-sweep, and nate's CPU cells on the
largest systems can each take over an hour.

A cell that produces NaN gradients or errors is NOT a failure here. Stage 1 only
measures cost and records gradient health as a PREDICTOR. The only thing entitled
to say FAIL is Stage 3, where a real Bayesian analysis either finishes or doesn't.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_registry import config_label, project_root, scaling_configs  # noqa: E402

# Phrases that identify the REAL cause when the visible exception is a mask.
# numba cannot serialise an exception whose context holds a jax.custom_vjp, so the
# surfaced error is "custom_vjp.__new__() missing 1 required positional argument:
# 'fun'" while the actual failure appears much earlier in the traceback -- outside
# any fixed-size tail. Scan the WHOLE stderr for these instead of guessing a depth.
ERROR_SIGNALS = (
    "Non-finite values returned by ODE forward solve",
    "Non-finite",
    "RESOURCE_EXHAUSTED",
    "out of memory",
    "maximum number of solver steps",
    "The maximum number of solver steps was reached",
    "singular",
    "NaN",
    "inf",
    "dtype",
    "float32",
    "float64",
)


def error_signals(stderr: str, limit: int = 12) -> list[str]:
    """Lines anywhere in stderr that name a plausible root cause, de-duplicated."""
    seen, out = set(), []
    for line in stderr.splitlines():
        t = line.strip()
        if not t or t.startswith(("File \"", "  File \"")):
            continue
        if any(sig in t for sig in ERROR_SIGNALS) and t not in seen:
            seen.add(t)
            out.append(t[:220])
            if len(out) >= limit:
                break
    return out


BENCH_DIR = project_root() / "Results" / "Benchmarks"
DEFAULT_FLOORS = (0.0, 0.0001, 0.001)


def timing_path(machine: str, device: str) -> Path:
    return BENCH_DIR / f"timing__{machine}__{device}.jsonl"


def cell_key(machine: str, device: str, floor: float, label: str,
             precision: int = 64, rtol: str = "1e-4", atol: str = "1e-8") -> str:
    """Every axis that changes the measurement must appear here.

    Resume-by-key is what survives Alpine's 24 h cap, so a key that omits an axis
    causes a silently WRONG skip: the fp32 cell would be treated as already measured
    because its fp64 twin was.
    """
    return f"{machine}|{device}|fp{precision}|{rtol}|{atol}|{floor:g}|{label}"


def parse_tolerances(spec: str) -> list[tuple[str, str]]:
    """'1e-4:1e-8,1e-6:1e-8' -> [('1e-4','1e-8'), ('1e-6','1e-8')]. Kept as strings so
    the recorded value is exactly what was passed to diffrax, not a reformatted float."""
    out = []
    for pair in spec.split(","):
        rtol, _, atol = pair.strip().partition(":")
        if not atol:
            raise ValueError(f"tolerance {pair!r} must be 'rtol:atol'")
        out.append((rtol.strip(), atol.strip()))
    return out


def load_done(path: Path) -> set[str]:
    """Keys already recorded, so a resumed sweep skips them."""
    done: set[str] = set()
    if not path.exists():
        return done
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue          # tolerate a torn last line from a killed job
        if r.get("cell_key"):
            done.add(r["cell_key"])
    return done


def append_row(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(row) + "\n")
        fh.flush()
        os.fsync(fh.fileno())     # a SLURM kill must not lose the row just measured


def run_timing_cell(config: Path, *, machine: str, device: str, floor: float,
                    python: str, rtol: str, atol: str, max_steps: int,
                    evals: int, timeout_s: float, max_eval_seconds: float | None,
                    precision: int = 64, utils: str | None = None) -> dict:
    """One cell. Returns the row to record; never raises."""
    label = config_label(config)
    # fp32 is not a flag but a whole parallel module tree: jax_enable_x64 is a global
    # fixed at import. Utilities32/ is rebuilt by Sync/make_utils32.sh, which the
    # driver invokes automatically whenever a 32-bit precision is requested.
    if utils is None:
        utils = "Utilities32" if precision == 32 else "Utilities"
    tag = f"{machine} {device} fp{precision} rtol={rtol} atol={atol} floor={floor:g} {label}"
    cmd = [python, "-u", "Utilities/precision_probe.py",
           "--config", str(config), "--utils", utils,
           "--rtol", rtol, "--atol", atol, "--max-steps", str(max_steps),
           "--evals", str(evals), "--label", tag]
    if precision == 32:
        cmd += ["--fp32"]
    if floor > 0:
        cmd += ["--floor", f"{floor:g}"]
    if max_eval_seconds:
        cmd += ["--max-eval-seconds", str(max_eval_seconds)]

    env = dict(os.environ)
    if device == "cpu":
        env["JAX_PLATFORMS"] = "cpu"        # same mechanism run_full_matrix.sh uses
    env.setdefault("PYTHONUNBUFFERED", "1")

    row = {
        "cell_key": cell_key(machine, device, floor, label, precision, rtol, atol),
        "machine": machine, "device": device, "floor": floor,
        "precision": precision, "rtol": rtol, "atol": atol, "utils": utils,
        "config_label": label, "config": str(config),
        "measured_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    t0 = time.perf_counter()
    try:
        proc = subprocess.run(cmd, cwd=project_root(), env=env, timeout=timeout_s,
                              capture_output=True, text=True)
    except subprocess.TimeoutExpired:
        row.update(status="timeout", wall_sec=round(time.perf_counter() - t0, 1),
                   note=f"exceeded {timeout_s}s; ms_per_grad_eval is a LOWER bound")
        return row
    row["wall_sec"] = round(time.perf_counter() - t0, 1)

    # Take the LAST RESULT line: jax prints plenty of unrelated chatter first, and
    # a GPU-less node emits a harmless cuInit error that must not be mistaken for one.
    result_lines = [l for l in proc.stdout.splitlines() if l.startswith("RESULT ")]
    if not result_lines:
        # Keep a TAIL of stderr, not just the last line. Errors here are often
        # masked: numba cannot serialise an exception whose context holds a
        # jax.custom_vjp, so the real failure (e.g. "Non-finite values returned by
        # ODE forward solve") appears EARLIER in the traceback and the final line
        # is a misleading "custom_vjp.__new__() missing 1 required positional
        # argument". Recording one line makes those cells undiagnosable without
        # re-running them, which on the big systems costs hours.
        err_lines = proc.stderr.strip().splitlines() or ["no RESULT line"]
        row.update(status="error", error=err_lines[-1][:300],
                   error_tail=[l[:200] for l in err_lines[-25:]],
                   error_signals=error_signals(proc.stderr))
        return row

    try:
        payload = json.loads(result_lines[-1][len("RESULT "):])
    except json.JSONDecodeError as exc:
        row.update(status="error", error=f"unparseable RESULT: {exc}")
        return row
    # precision_probe emits its own "device" (the jax device list, e.g.
    # ['cuda:0']) and "label", which would clobber this cell's coordinates.
    # Merge the payload UNDER the coordinates, not over them.
    coords = {k: row[k] for k in ("cell_key", "machine", "device", "floor", "precision",
                                  "rtol", "atol", "utils",
                                  "config_label", "config", "measured_at", "wall_sec")
              if k in row}
    row.update(payload)
    row["jax_device"] = row.pop("device", None) if isinstance(row.get("device"), list) else row.get("device")
    row.update(coords)

    if not payload.get("ok"):
        row["status"] = "error"
        # precision_probe truncates to 300 chars and the real cause may be masked;
        # keep the stderr tail alongside it for the same reason as above.
        err_lines = proc.stderr.strip().splitlines()
        if err_lines:
            row["error_tail"] = [l[:200] for l in err_lines[-25:]]
            row["error_signals"] = error_signals(proc.stderr)
    elif payload.get("n_chains_finite", 0) == payload.get("n_chains", 4):
        row["status"] = "ok"
    else:
        # Cost is still valid and the run may still sample -- a vmapped-NaN cell was
        # measured sampling 20/20 unique draws with zero divergences. Predictor only.
        row["status"] = "nonfinite"
    return row


def stage_timing(args: argparse.Namespace) -> int:
    full_ladder = scaling_configs(args.ladder) if args.ladder else scaling_configs()
    configs = full_ladder
    if not configs:
        print(f"FATAL: no configs found in the '{args.ladder or 'default'}' ladder. "
              f"Generate them with Utilities/generate_chain_configs.py.", file=sys.stderr)
        return 1
    if args.configs:
        wanted = {c.strip() for c in args.configs.split(",")}
        configs = [c for c in configs if config_label(c) in wanted]
    floors = [float(f) for f in args.floors.split(",")]
    devices = [d.strip() for d in args.devices.split(",")]
    precisions = [int(p) for p in str(args.precisions).split(",")]
    tolerances = parse_tolerances(args.tolerances)

    if 32 in precisions:
        # Rebuild rather than check. Utilities32/ is gitignored build output that
        # drifts silently: before this sweep its experiment_framework still predated
        # initial_condition_floor, so every fp32 floor cell would have reported "the
        # floor makes no difference" -- a false negative indistinguishable from a
        # result. make_utils32.sh is idempotent and takes under a second, and it does
        # far more than flip the x64 flag: it STRIPS every explicit float64, because
        # a hardcoded dtype in a 32-bit build yields a mixed pytensor/jax graph that
        # dies building the custom_vjp bridge.
        rc = subprocess.run(["bash", "Sync/make_utils32.sh"], cwd=project_root(),
                            capture_output=True, text=True)
        if rc.returncode != 0:
            print("FATAL: could not rebuild Utilities32/:\n" + rc.stdout + rc.stderr,
                  file=sys.stderr)
            return 1
        print("[fp32] rebuilt Utilities32/ from Utilities/", flush=True)

    deadline = time.time() + args.max_hours * 3600 if args.max_hours else None
    # Timeout scales with position in the FULL ladder, not the filtered subset, so
    # running one config alone still gets the budget its size warrants.
    order = full_ladder
    total = attempted = 0

    for device in devices:
        out_path = timing_path(args.machine, device)
        done = load_done(out_path)
        # configs are already sorted smallest-first, so a truncated sweep still
        # covers the systems most likely to be usable
        for config in configs:
            for precision in precisions:
                for rtol, atol in tolerances:
                    for floor in floors:
                        total += 1
                        key = cell_key(args.machine, device, floor, config_label(config),
                                       precision, rtol, atol)
                        if key in done and not args.force:
                            continue
                        if deadline and time.time() > deadline:
                            print(f"[deadline] stopping before {key}", flush=True)
                            return 0
                        # Bigger systems get proportionally longer to finish one eval.
                        idx = (order.index(config) + 1) if config in order else 1
                        timeout_s = args.timeout * idx
                        print(f"[cell] {key}  (timeout {timeout_s:.0f}s)", flush=True)
                        row = run_timing_cell(
                            config, machine=args.machine, device=device, floor=floor,
                            python=args.python, rtol=rtol, atol=atol,
                            max_steps=args.max_steps, evals=args.evals,
                            timeout_s=timeout_s, max_eval_seconds=args.max_eval_seconds,
                            precision=precision)
                        append_row(out_path, row)
                        attempted += 1
                        print(f"   -> {row.get('status')}  ms/eval={row.get('ms_per_grad_eval')}  "
                              f"chains_finite={row.get('n_chains_finite')}", flush=True)

    print(f"[done] {attempted} cells measured this pass, {total} in the matrix", flush=True)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stage", default="timing", choices=["timing"])
    ap.add_argument("--machine", default=socket.gethostname().split(".")[0])
    ap.add_argument("--devices", default="gpu,cpu")
    ap.add_argument("--floors", default=",".join(f"{f:g}" for f in DEFAULT_FLOORS))
    ap.add_argument("--configs", default=None, help="comma-separated labels; default all")
    ap.add_argument("--ladder", default=None,
                    help="scaling ladder directory under Results/; default is the "
                         "chain-length ladder (see run_registry.scaling_configs)")
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--precisions", default="64",
                    help="comma-separated: 64, 32, or 32,64. fp32 runs against "
                         "Utilities32/ and is refused if that tree is out of sync.")
    ap.add_argument("--tolerances", default="1e-4:1e-8",
                    help="comma-separated rtol:atol pairs, e.g. 1e-4:1e-8,1e-6:1e-8")
    ap.add_argument("--max-steps", type=int, default=20000)
    ap.add_argument("--evals", type=int, default=2)
    ap.add_argument("--max-eval-seconds", type=float, default=1800)
    ap.add_argument("--timeout", type=float, default=1800,
                    help="per-cell timeout for the SMALLEST system; scaled by config index")
    ap.add_argument("--max-hours", type=float, default=None,
                    help="stop starting new cells after this; lets a job checkpoint itself")
    ap.add_argument("--force", action="store_true", help="re-measure cells already recorded")
    args = ap.parse_args()

    if shutil.which(args.python) is None and not Path(args.python).exists():
        print(f"FATAL: interpreter not found: {args.python}", file=sys.stderr)
        return 1
    return stage_timing(args)


if __name__ == "__main__":
    raise SystemExit(main())
