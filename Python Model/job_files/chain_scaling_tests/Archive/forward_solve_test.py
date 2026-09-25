"""Does the negative-concentration floor in ReactionNetwork.__call__ change results?

Compares the production (floored) RHS against an unclamped variant with the
jnp.maximum(y, 0.0) removed, on C4_NoFB / C12 / C20+unsat, using each system's
real inference-time ODE settings (rtol=1e-5, atol=1e-7, pcoeff=0.4, icoeff=0.3,
dcoeff=0.0 -- confirmed identical across all three from their solver_params.json).

All stoichiometry in these three systems is integer (checked before writing this),
so removing the floor entirely is safe from the fractional-power NaN case the
floor also guards against -- any NaN/Inf here is genuinely the "large excursion
blows up the implicit solver" mechanism, not a confound.

Two sub-experiments:
  A. Baseline forward solve at nominal theta (d1=d2=0, every other group=1),
     t=[0, 720]s -- the future experimental horizon, not the current 150s data.
  B. The same y0, at N draws with every scaling group perturbed at once across
     wide ranges (multiplicative groups log-uniform over [0.001, 1000], matching
     the project's own a1 prior bounds; d-groups uniform over [-8, 8], since
     they're additive inside exp()) -- approximating what an under-adapted NUTS
     proposal can look like before the mass matrix/step size adapt. This is an
     approximation of the sampler's actual proposal distribution, not a
     reproduction of it -- see truncated_warmup_test.py for the direct version.

Reports, per run: steps taken/rejected, wall time, whether it crashed (NaN/Inf
in the accepted portion of the trajectory, or a non-successful diffrax result),
the most-negative concentration seen in the raw state trajectory (the floor
only affects the RHS's rate computation, not the state itself, so this is
visible regardless of which variant ran), and how far the final states of the
two variants actually differ.
"""
from __future__ import annotations

import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path("/projects/anth4580/Bayesian")
UTILITIES_DIR = PROJECT_ROOT / "Utilities"
sys.path.insert(0, str(UTILITIES_DIR))

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import diffrax as dfrx

from generate_chain_data import ChainSystem, nominal_scaling_group_overrides
from reaction_model_builder import (
    ReactionNetwork,
    discover_scaling_groups,
    set_scaling_group_values,
)

# Real inference-time ODE settings -- identical across all three systems, confirmed
# from Results/Chain Scaling Tests/Chain {C4_NoFB,C12,C20+unsat} - a1/solver_params.json
RTOL, ATOL = 1e-5, 1e-7
PCOEFF, ICOEFF, DCOEFF = 0.4, 0.3, 0.0
MAX_STEPS = 20_000
T1 = 720.0
N_EXTREME_DRAWS = 100
SEED = 0
SYSTEMS = ["C4_NoFB", "C12", "C20+unsat"]


def reactions_for(system: str) -> list[Path]:
    """Mirrors the reactions_source lists in each system's real solver_params.json."""
    root = PROJECT_ROOT / "Reactions" / "EC_FAS_ME1"
    if system == "C4_NoFB":
        # No FabF / No FabB -- C4 with them fails via ACP sequestration (see memory
        # nofb-means-no-fabf-fabb). Uses the plain C4 directory, 7 of 9 files.
        names = ["FabD", "FabH", "FabG", "FabZ", "FabI", "TesA", "FabA"]
        return [root / "C4" / f"{n}.yaml" for n in names]
    return sorted((root / system).glob("*.yaml"))


class UnclampedReactionNetwork(ReactionNetwork):
    """Same compiled network as ReactionNetwork, minus jnp.maximum(y, 0.0).

    All stoichiometry here is integer, so this cannot produce a fractional-power
    NaN on its own -- only the "large excursion blows up the implicit solver"
    mechanism the floor's comment describes.
    """

    def __call__(self, t, y, args):
        if self.param_idx_arr.shape[0] == 0:
            return jnp.zeros_like(y)
        theta = jnp.asarray(args)
        reactant_conc = y[self.reactant_idx_arr]  # no floor, vs. production's jnp.maximum(y, 0.0)
        reactant_powers = jnp.where(
            self.reactant_mask_arr,
            reactant_conc ** self.reactant_stoich_arr,
            1.0,
        )
        mass_action_terms = jnp.prod(reactant_powers, axis=1)
        scale_factors = jnp.stack([
            fn(theta) if fn is not None else jnp.ones(())
            for fn in self.scale_fns
        ])
        rates = theta[self.param_idx_arr] * scale_factors * mass_action_terms
        return self.stoich_matrix @ rates


def unclamp(network: ReactionNetwork) -> UnclampedReactionNetwork:
    return UnclampedReactionNetwork(
        network.param_idx_arr, network.scale_fns, network.reactant_idx_arr,
        network.reactant_stoich_arr, network.reactant_mask_arr, network.stoich_matrix,
    )


# Named diffrax result codes, checked explicitly rather than inferred from step
# count -- max_steps_reached (ran out of step budget, not a numerical failure) and
# singular (the exact "linear solver returned non-finite output" crash reproduced
# in truncated_warmup_test.py) are mechanistically different outcomes and must not
# be conflated into one "non-converged" bucket.
RESULT_NAMES = ["successful", "max_steps_reached", "singular", "nonlinear_divergence", "event_occurred"]


def classify_result(result) -> str:
    for name in RESULT_NAMES:
        if result == getattr(dfrx.RESULTS, name):
            return name
    return f"other:{result}"


def run_once(network, theta, y0, label: str) -> dict:
    t_wall0 = time.perf_counter()
    sol = dfrx.diffeqsolve(
        dfrx.ODETerm(network), dfrx.Kvaerno5(),
        t0=0.0, t1=T1, dt0=1e-6,
        y0=jnp.asarray(y0, dtype=jnp.float64), args=theta,
        saveat=dfrx.SaveAt(steps=True),
        stepsize_controller=dfrx.PIDController(
            rtol=RTOL, atol=ATOL, pcoeff=PCOEFF, icoeff=ICOEFF, dcoeff=DCOEFF),
        max_steps=MAX_STEPS, throw=False,
    )
    jax.block_until_ready(sol)
    wall_s = time.perf_counter() - t_wall0

    total = int(np.asarray(sol.stats["num_steps"]))
    accepted = int(np.asarray(sol.stats["num_accepted_steps"]))
    # Only the accepted prefix is meaningful -- SaveAt(steps=True) pads the rest
    # of the fixed-size buffer, and that padding must not be mistaken for a crash.
    ys_valid = np.asarray(sol.ys[:accepted]) if accepted > 0 else np.zeros((0,))
    finite = np.isfinite(ys_valid)
    crashed = (not bool(finite.all())) if ys_valid.size else True
    min_conc = float(np.min(ys_valid[finite])) if finite.any() else float("nan")
    result_name = classify_result(sol.result)
    success = (result_name == "successful") and not crashed and total < MAX_STEPS
    final_state = ys_valid[-1] if (accepted > 0 and not crashed) else None
    return dict(label=label, success=success, crashed=crashed, result_name=result_name,
               num_steps=total, num_rejected=total - accepted, wall_seconds=wall_s,
               min_concentration=min_conc, final_state=final_state)


def sample_extreme_overrides(rng: np.random.Generator, scaling_groups) -> dict[str, float]:
    out = {}
    for g in scaling_groups:
        if g.startswith("d"):
            out[g] = float(rng.uniform(-8.0, 8.0))
        else:
            out[g] = float(np.exp(rng.uniform(np.log(0.001), np.log(1000.0))))
    return out


def main() -> None:
    rng = np.random.default_rng(SEED)
    for system in SYSTEMS:
        print(f"\n=== {system} ===", flush=True)
        rx = reactions_for(system)
        scaling_groups = discover_scaling_groups(rx)
        sys_ = ChainSystem(rx, RTOL, ATOL, PCOEFF, ICOEFF, DCOEFF,
                          scaling_group_overrides=nominal_scaling_group_overrides(scaling_groups))
        y0 = sys_.y0()
        network_floor = sys_.network
        network_nofloor = unclamp(sys_.network)

        print("  -- Part A: baseline (nominal theta), 720s --", flush=True)
        res_floor = run_once(network_floor, sys_.theta, y0, "baseline/floor")
        res_nofloor = run_once(network_nofloor, sys_.theta, y0, "baseline/no-floor")
        for r in (res_floor, res_nofloor):
            print(f"    {r['label']:<20} success={r['success']!s:<5} crashed={r['crashed']!s:<5} "
                 f"steps={r['num_steps']:>6} rejected={r['num_rejected']:>6} "
                 f"wall_s={r['wall_seconds']:>8.3f} min_conc={r['min_concentration']:.6g}", flush=True)
        if res_floor["final_state"] is not None and res_nofloor["final_state"] is not None:
            diff = float(np.max(np.abs(res_floor["final_state"] - res_nofloor["final_state"])))
            print(f"    max |final_state difference| floor vs no-floor: {diff:.6g}", flush=True)
        else:
            print("    final_state comparison skipped (one or both variants crashed)", flush=True)

        print(f"  -- Part B: {N_EXTREME_DRAWS} extreme-parameter draws, same y0, 720s --", flush=True)
        counts_floor = Counter()
        counts_nofloor = Counter()
        max_steps_floor = max_steps_nofloor = 0
        wall_floor = wall_nofloor = 0.0
        worst_min_conc_floor = worst_min_conc_nofloor = 0.0
        jsonl_path = PROJECT_ROOT / "job_files" / "chain_scaling_tests" / f"extreme_draws_{system.replace('+', 'plus')}.jsonl"
        with open(jsonl_path, "w") as jf:
            for i in range(N_EXTREME_DRAWS):
                overrides = sample_extreme_overrides(rng, scaling_groups)
                theta_i = set_scaling_group_values(sys_.theta, sys_.params, overrides)
                r_f = run_once(network_floor, theta_i, y0, f"extreme[{i}]/floor")
                r_nf = run_once(network_nofloor, theta_i, y0, f"extreme[{i}]/no-floor")
                counts_floor[r_f["result_name"]] += 1
                counts_nofloor[r_nf["result_name"]] += 1
                max_steps_floor = max(max_steps_floor, r_f["num_steps"])
                max_steps_nofloor = max(max_steps_nofloor, r_nf["num_steps"])
                wall_floor += r_f["wall_seconds"]
                wall_nofloor += r_nf["wall_seconds"]
                if np.isfinite(r_f["min_concentration"]):
                    worst_min_conc_floor = min(worst_min_conc_floor, r_f["min_concentration"])
                if np.isfinite(r_nf["min_concentration"]):
                    worst_min_conc_nofloor = min(worst_min_conc_nofloor, r_nf["min_concentration"])
                jf.write(json.dumps(dict(
                    draw=i, overrides=overrides,
                    floor_result=r_f["result_name"], floor_steps=r_f["num_steps"],
                    floor_wall_s=r_f["wall_seconds"], floor_min_conc=r_f["min_concentration"],
                    nofloor_result=r_nf["result_name"], nofloor_steps=r_nf["num_steps"],
                    nofloor_wall_s=r_nf["wall_seconds"], nofloor_min_conc=r_nf["min_concentration"],
                )) + "\n")
                jf.flush()
        print(f"    floor:    {dict(counts_floor)}, max_steps={max_steps_floor}, "
             f"total_wall_s={wall_floor:.2f}, worst_min_conc={worst_min_conc_floor:.6g}", flush=True)
        print(f"    no-floor: {dict(counts_nofloor)}, max_steps={max_steps_nofloor}, "
             f"total_wall_s={wall_nofloor:.2f}, worst_min_conc={worst_min_conc_nofloor:.6g}", flush=True)
        print(f"    per-draw detail written to {jsonl_path}", flush=True)


if __name__ == "__main__":
    main()
