"""Dump per-chain free-parameter and log-posterior arrays for every tracked
run, as JSON.

Companion to export_progress_series.py (which carries timing, not content). This
carries what each chain actually sampled, so a notebook on a laptop can check
whether "converged" chains are all in the same place -- the stranded-chain
question -- and can draw ordinary inference plots (trace, posterior) without a
fresh SSH round trip per figure.

For a converged run (phase == "done"), pulls the SAMPLING phase: that's the
actual posterior. For a run that never reached sampling (a stalled floor run at
C10+ is the case this exists for), pulls whatever WARMUP is there instead, so a
still-unhealthy run's instability is visible rather than silently absent. Only
one of "sampling" / "warmup" is ever present per system, and which one is
recorded under "source".

2026-09-09: generalized from a1-only (single-parameter ladder) to however many
free parameters a run actually has -- the multi-parameter pilot (a1+c2, a1+c3)
needs both, not just a1. Parameter names are discovered from whichever
"<name>_log__" arrays exist in the zarr store, not hardcoded.

    python3 export_posterior_series.py > posterior.json
    python3 export_posterior_series.py tightest tightest_nofloor > posterior.json

Shape:
    {set: {system: {"source": "sampling" | "warmup",
                    "n_chains": 8,
                    "params": {"a1": [[chain0 draws...], ...], "c2": [[...], ...]},
                    "lp": [[chain0 draws...], [chain1 draws...], ...],
                    "accept": [[chain0...], ...],
                    "n_steps": [[chain0...], ...],
                    "step_size": [[chain0...], ...]}}}

"accept" and "n_steps" are the per-checkpoint acceptance_rate and n_steps that
justify calling a run dead rather than merely slow: a dead chain sits at
accept==0 and n_steps==1 every iteration (one leapfrog step, the ODE solve
fails, the proposal is rejected, the position never moves) -- a live chain does
not, however slow. "step_size" is NUTS's own per-chain adapted step size --
included for checking whether the (diagonal) mass matrix is coping with more
than one free parameter, not just for the single-parameter dead/alive check.

Each parameter in "params" is exp(<name>_log__) -- the constrained value, not
the sampler's internal unconstrained one. Rounded to 6 significant figures; lp
to 3 decimals, step_size to 8: this is for plotting, not for re-deriving a
posterior, and the JSON would otherwise run several MB larger for no visual
benefit.
"""
import json
import re
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import warmup_status as W


def _run_dir(system, suffix):
    return "%s/Chain %s - %s" % (W.BASE, system, suffix)


def _round(seq, ndig):
    return [round(float(v), ndig) for v in seq]


def _pull(system, suffix):
    ckpt = _run_dir(system, suffix) + "/checkpoint"
    meta_path = ckpt + "/checkpoint_meta.json"
    zpath = ckpt + "/draws.zarr"
    if not os.path.exists(meta_path) or not os.path.exists(zpath):
        return None
    try:
        meta = json.load(open(meta_path))
    except Exception:
        return None

    phase = meta.get("phase")
    source = "sampling" if phase == "done" else "warmup"

    try:
        import zarr
        import numpy as np
    except Exception as e:
        return {"error": "%s: %s" % (type(e).__name__, e)}

    try:
        g = zarr.open(zpath, mode="r")
        # A parameter is stored under "<name>_log__" only when its prior needed a
        # log transform (LogNormal, i.e. every multiplicative scaling group). A
        # prior on an unbounded parameter -- d1/d2, which enter TesA additively
        # inside exp() and are therefore Normal around a nominal of 0.0 -- has no
        # transform, so it is stored under its bare name and must NOT be
        # exponentiated. Keying only off "_log__" silently dropped such a
        # parameter from the export entirely rather than failing.
        # A third form, "<name>_x<factor>__", is a Normal prior sampled through
        # u = factor * x (inference_runner._ScaleTransform) so its coordinate
        # matches the log-scale groups; divide the factor back out.
        raw_keys = [k for k in g[source].array_keys()]
        params = {}
        for key in sorted(raw_keys):
            values = np.asarray(g["%s/%s" % (source, key)][:])
            scaled = re.match(r"^(.+)_x([0-9.eE+-]+)__$", key)
            if key.endswith("_log__"):
                params[key[:-len("_log__")]] = np.exp(values)
            elif scaled:
                params[scaled.group(1)] = values / float(scaled.group(2))
            else:
                params[key] = values
        param_names = sorted(params)
        if not param_names:
            return None
        stats_group = "%s_stats" % source
        lp = np.asarray(g["%s/lp" % stats_group][:])
        accept = np.asarray(g["%s/acceptance_rate" % stats_group][:])
        n_steps = np.asarray(g["%s/n_steps" % stats_group][:])
        step_size = (np.asarray(g["%s/step_size" % stats_group][:])
                    if "step_size" in g[stats_group].array_keys() else None)
    except Exception as e:
        return {"error": "%s: %s" % (type(e).__name__, e)}

    any_param = next(iter(params.values()))
    if any_param.size == 0:
        return None
    n_chains = any_param.shape[0] if any_param.ndim == 2 else 1

    out = {
        "source": source,
        "n_chains": int(n_chains),
        "params": {name: [_round(row, 6) for row in np.atleast_2d(arr)]
                  for name, arr in params.items()},
        "lp": [_round(row, 3) for row in np.atleast_2d(lp)],
        "accept": [_round(row, 4) for row in np.atleast_2d(accept)],
        "n_steps": [[int(v) for v in row] for row in np.atleast_2d(n_steps)],
    }
    if step_size is not None:
        out["step_size"] = [_round(row, 8) for row in np.atleast_2d(step_size)]
    return out


def main():
    which = [a for a in sys.argv[1:] if a in W.SETS] or list(W.SETS)
    out = {}
    for setname in which:
        _, jobids_file = W.SETS[setname]
        pairs = W._job_pairs(jobids_file)
        systems = sorted({s for s, _ in pairs}, key=W._chain_sort_key)
        out[setname] = {}
        for s in systems:
            # Per-system: a finished system may already be on the new
            # post-rename name while its still-running siblings are not.
            suffix = W.resolve_suffix(setname, s)
            rec = _pull(s, suffix)
            if rec is not None:
                out[setname][s] = rec
    json.dump(out, sys.stdout)


if __name__ == "__main__":
    main()
