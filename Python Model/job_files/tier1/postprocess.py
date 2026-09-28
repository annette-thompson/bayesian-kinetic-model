"""Post-process every finished Tier-1 run in one command: pull results from the cluster, score
them, and redraw the drafts.

  python postprocess.py              # pull, then every step below
  python postprocess.py --no_pull    # use what is already local

Steps:
  1. pull            finished run folders from the cluster, without checkpoint/ (the draw
                     stores), newer files only; also the cost-estimate history and the log
                     of Slurm's estimated starts (cost_estimate.py)
  2. recovery        recovery_report.py over every finished run; the table is printed and the
                     full result written to Results/Tier1/figures/recovery.json
  3. identifiability identifiability_report.py for each finished R0-R3, R6 or R7 run that has
                     no identifiability.json yet, then the Fig 6/6b draft (plot_tier1_drafts.py
                     fig6) as Results/Tier1/figures/identifiability_<run>.png
  4. sbc             sbc.py ranks on the finished SBC replicates (sbc_ranks.json here, the
                     figure in Results/Tier1/figures)
  5. figures         tier1_result_figures.py all (Figs 2 and 4, the R7 and R8 checks)

Each step runs in its own process (the laptop's jaxlib aborts when a second model is compiled
in one process). A failing step is reported and the rest still run.
"""
import argparse
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
RESULTS = ROOT / "Results" / "Tier1"
CLUSTER = "curc:/projects/anth4580/Bayesian/Results/Tier1/"
CLUSTER_LOGS = ["curc:/projects/anth4580/Bayesian/job_files/tier1/" + f
                for f in ("cost_estimate_history.jsonl", "start_predictions.jsonl")]
sys.path.insert(0, str(HERE))
from tier1_status import group_of  # noqa: E402

IDENTIFIABILITY_GROUPS = ("R0", "R1", "R2", "R3", "R6", "R7")


def run(label, cmd, drop=None):
    """Run one step; `drop` is a substring of output lines to leave out (e.g. unfinished runs)."""
    print(f"\n=== {label}", flush=True)
    proc = subprocess.run([str(c) for c in cmd], cwd=HERE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                          universal_newlines=True)
    for line in proc.stdout.splitlines():
        if (drop and drop in line) or "Warning" in line or line.startswith(("[Cpu", "Total JAX", "  warnings.warn")):
            continue
        print(line, flush=True)
    if proc.returncode != 0:
        print(f"--- {label} FAILED (exit {proc.returncode})", flush=True)
    return proc.returncode == 0


def finished_runs():
    return sorted(d.name for d in RESULTS.iterdir()
                  if d.is_dir() and d.name.startswith("Tier1 ") and (d / "posterior_samples_pm.nc").exists())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--no_pull", action="store_true", help="skip pulling from the cluster")
    a = ap.parse_args()
    py = sys.executable
    ok = {}

    if not a.no_pull:
        ok["pull"] = run("pull", ["rsync", "-az", "--update", "--protect-args", "--exclude", "checkpoint/",
                                  "--exclude", "finalize_stage.nc", CLUSTER, str(RESULTS) + "/"])
        ok["pull"] &= run("pull logs", ["rsync", "-az"] + CLUSTER_LOGS + [str(HERE) + "/"])
    done = finished_runs()
    print(f"\n{len(done)} finished runs: " + ", ".join(r.replace("Tier1 ", "") for r in done), flush=True)

    (RESULTS / "figures").mkdir(exist_ok=True)
    ok["recovery"] = run("recovery", [py, "recovery_report.py", "--json", RESULTS / "figures" / "recovery.json"],
                         drop="(skipped: no posterior netcdf")

    ident = [r for r in done if group_of(r) in IDENTIFIABILITY_GROUPS]
    ok["identifiability"] = True
    for r in ident:
        out = RESULTS / r / "identifiability.json"
        if not out.exists():
            ok["identifiability"] &= run(f"identifiability {r}", [py, "identifiability_report.py", "--run", r])
        if out.exists():
            ok["identifiability"] &= run(f"fig6 draft {r}", [py, "plot_tier1_drafts.py", "fig6", out])

    ok["sbc"] = run("sbc", [py, "sbc.py", "ranks"])
    ok["figures"] = run("figures", [py, "tier1_result_figures.py", "all"])

    print("\n=== summary: " + ", ".join(f"{k} {'ok' if v else 'FAILED'}" for k, v in ok.items()))
    print(f"figures: {RESULTS / 'figures'}; per-run identifiability drafts in each run folder; "
          f"SBC ranks in {HERE / 'sbc_ranks.png'}")


if __name__ == "__main__":
    main()
