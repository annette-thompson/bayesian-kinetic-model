"""Submit each system's inference run as soon as its rtol ladder job picks a tolerance.

Runs on the Blanca login node, once per monitor cycle. For every system whose
rtol_ladder job has written its results JSON and that has not already been
submitted, this:

  1. reads the chosen rtol out of rtol_ladder_results/<system>_plus0.json
  2. writes that rtol into the system's inference config so the solver settings
     used during sampling match the ones its training data was generated at.
     A mismatch here is exactly the failure we were avoiding by taking the
     inference set down before promoting data, so it is verified, not assumed.
  3. deletes the system's checkpoint tree. The data underneath it has changed,
     so any resumed chain would be sampling a different posterior than it
     started on.
  4. submits the inference job and records the SLURM id.

State lives in submitted.json so repeated calls are idempotent: a system already
submitted is skipped, and the script is safe to run on a timer.

Prints one line per action taken and nothing when there is nothing to do, so it
can drive a Monitor directly.
"""
import json
import shutil
import subprocess
import sys
from pathlib import Path

PROJECT = Path("/projects/anth4580/Bayesian")
LADDER = PROJECT / "job_files" / "rtol_ladder"
RESULTS = LADDER / "rtol_ladder_results"
STATE = LADDER / "submitted.json"
SBATCH = PROJECT / "Bayesian Inference" / "run_inference_segment_gpu_blanca.sh"
CONFIG_DIR = PROJECT / "Results" / "Chain Scaling Tests"
JOBIDS = PROJECT / "job_files" / "inference_a1_tightest_jobids.txt"

SYSTEMS = ["C4_NoFB", "C6", "C8", "C10", "C12", "C12+unsat", "C14", "C14+unsat",
           "C16", "C16+unsat", "C18", "C18+unsat", "C20", "C20+unsat"]


def load_state():
    if STATE.exists():
        try:
            return json.loads(STATE.read_text())
        except ValueError:
            pass
    return {}


def save_state(state):
    STATE.write_text(json.dumps(state, indent=2))


def config_path(system):
    return CONFIG_DIR / f"Chain {system} - a1 tightest" / "solver_params.json"


def set_rtol(system, rtol):
    """Point the inference config at the tolerance its data was generated with."""
    p = config_path(system)
    cfg = json.loads(p.read_text())
    old = cfg["ODE_stepsize_controller"]["rtol"]
    cfg["ODE_stepsize_controller"]["rtol"] = rtol
    p.write_text(json.dumps(cfg, indent=2))
    return old


def clear_checkpoint(system):
    d = CONFIG_DIR / f"Chain {system} - a1 tightest" / "checkpoint"
    if d.exists():
        shutil.rmtree(d)
        return True
    return False


def submit(system):
    cmd = ("module load slurm/blanca >/dev/null 2>&1; "
           f"sbatch --parsable --job-name='a1t_{system}' "
           f"'{SBATCH}' '{config_path(system)}'")
    r = subprocess.run(["bash", "-lc", cmd], stdout=subprocess.PIPE,
                       stderr=subprocess.PIPE)
    out = r.stdout.decode().strip()
    err = r.stderr.decode().strip()
    if r.returncode != 0 or not out.isdigit():
        return None, (err or out or "sbatch produced no job id")
    return out, None


def main():
    state = load_state()
    acted = False

    for system in SYSTEMS:
        if system in state:
            continue
        rj = RESULTS / f"{system}_plus0.json"
        if not rj.exists():
            continue
        try:
            res = json.loads(rj.read_text())
        except ValueError:
            continue  # still being written
        entry = res.get(system)
        if not isinstance(entry, dict) or "chosen_rtol" not in entry:
            print(f"{system}: results file has no chosen_rtol; skipping")
            continue

        rtol = float(entry["chosen_rtol"])
        promoted = entry.get("promoted", False)
        per_tol = {t["rtol"]: t for t in entry.get("per_tolerance", [])
                   if isinstance(t, dict)}
        won = per_tol.get(rtol, {})
        steps = won.get("total_steps")
        err = won.get("max_err_pct")

        old = set_rtol(system, rtol)
        cleared = clear_checkpoint(system)
        jid, failure = submit(system)
        acted = True

        if failure:
            print(f"{system}: SUBMIT FAILED after choosing rtol={rtol:g} -- {failure}")
            state[system] = dict(rtol=rtol, jobid=None, error=failure)
            save_state(state)
            continue

        detail = f"total_steps={steps}" if steps is not None else "kept existing 1e-5 data"
        errs = f", max_err={err:.2f}%" if isinstance(err, (int, float)) else ""
        note = "" if promoted else " (no eligible looser tolerance; data unchanged)"
        print(f"{system}: rtol {old:g} -> {rtol:g}{note} | {detail}{errs} | "
              f"checkpoint {'cleared' if cleared else 'absent'} | submitted job {jid}")
        state[system] = dict(rtol=rtol, jobid=jid, total_steps=steps, max_err_pct=err)
        save_state(state)

        with open(JOBIDS, "a") as f:
            f.write(f"{system} {jid}\n")

    if acted:
        done = len(state)
        print(f"({done}/{len(SYSTEMS)} systems submitted)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
