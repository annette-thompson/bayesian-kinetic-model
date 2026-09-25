"""Direct reproduction test: does removing the negative-concentration floor
actually crash a real NUTS warmup, or was some other bug responsible for the
crash the floor's comment attributes to it? And does pointing the implicit
solver's linear solve at AutoLinearSolver(well_posed=False) -- the exact fix
named in that crash's own error message -- let it survive without the floor?

Runs the REAL, unmodified inference_runner.py end to end (same import, same
CLI parsing, same sampler) on C4_NoFB with a truncated config (tune=200,
draws=10 -- see the "masking-check-{floor,nofloor,robust}" solver_params.json
next to this file), so the sampler proposes real, adapting parameter values
instead of the hand-picked "extreme" draws in forward_solve_test.py's Part B.

Three modes, set via environment variable MASK_TEST_MODE (default "nofloor"):
  nofloor -- production. reaction_model_builder has had no floor since
             2026-09-14, so no patch is applied. (Before then this mode
             monkeypatched the jnp.maximum(y, 0.0) floor away; that
             configuration crashed in job 28159531.)
  floor  -- LEGACY, only to reproduce archived floor runs: the old clamp is
             patched back into ReactionNetwork.__call__.
  robust -- same floor removal, PLUS diffrax.Kvaerno5 is monkeypatched so
            that calling it (with no arguments, exactly how
            inference_runner.py's _build_solver does) returns the same
            Kvaerno5 with one field changed via equinox.tree_at:
            root_finder.linear_solver.well_posed = False instead of the
            default None. This is not a custom solver -- it is the
            standard, built-in Kvaerno5 with one existing configuration
            switch flipped, the same switch diffrax's own crash message
            named as the fix for this exact situation.

Neither inference_runner.py nor reaction_model_builder.py is modified on
disk -- this file is the only new code. Both patches target module-level
objects (reaction_model_builder.ReactionNetwork, diffrax.Kvaerno5) rather
than anything defined inside inference_runner.py itself, specifically so
they survive being picked up via sys.modules regardless of how
inference_runner.py is executed (runpy.run_path below creates a fresh
__main__ execution of that file's own code, but its `from
reaction_model_builder import ...` and `import diffrax as dfrx` statements
still resolve through the shared module cache, where these patches live).
"""
from __future__ import annotations

import os
import runpy
import sys
from pathlib import Path

PROJECT_ROOT = Path("/projects/anth4580/Bayesian")
UTILITIES_DIR = PROJECT_ROOT / "Utilities"
sys.path.insert(0, str(UTILITIES_DIR))

MODE = os.environ.get("MASK_TEST_MODE", "nofloor")
assert MODE in ("floor", "nofloor", "robust"), f"unknown MASK_TEST_MODE={MODE!r}"

if MODE == "floor":
    import jax.numpy as jnp
    import reaction_model_builder as rmb  # bare import -- must match inference_runner.py's own

    def _clamped_call(self, t, y, args):
        if self.param_idx_arr.shape[0] == 0:
            return jnp.zeros_like(y)
        theta = jnp.asarray(args)
        reactant_conc = jnp.maximum(y[self.reactant_idx_arr], 0.0)  # the legacy floor
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

    rmb.ReactionNetwork.__call__ = _clamped_call
    print("==> MASK_TEST_MODE=floor: LEGACY negative-concentration floor restored for this run", flush=True)
else:
    print(f"==> MASK_TEST_MODE={MODE}: no floor (production behavior)", flush=True)

if MODE == "robust":
    import diffrax as dfrx
    import equinox as eqx

    _robust_solver = eqx.tree_at(
        lambda s: s.root_finder.linear_solver.well_posed,
        dfrx.Kvaerno5(), False, is_leaf=lambda x: x is None,
    )
    dfrx.Kvaerno5 = lambda: _robust_solver  # _build_solver calls getattr(dfrx, "Kvaerno5")()
    print("==> MASK_TEST_MODE=robust: Kvaerno5's root_finder.linear_solver.well_posed set to "
         "False (was None) -- the standard solver, one field changed, no custom solver code", flush=True)

SOLVER_PARAMS_OVERRIDE = os.environ.get("MASK_TEST_SOLVER_PARAMS")
if SOLVER_PARAMS_OVERRIDE:
    solver_params = Path(SOLVER_PARAMS_OVERRIDE)
else:
    CONFIG_SUFFIX = os.environ.get("MASK_TEST_CONFIG_SUFFIX", MODE)
    solver_params = PROJECT_ROOT / "Results" / "Chain Scaling Tests" / f"Chain C4_NoFB - masking-check-{CONFIG_SUFFIX}" / "solver_params.json"
MAX_HOURS = os.environ.get("MASK_TEST_MAX_HOURS", "0.75")
print(f"==> Config: {solver_params}", flush=True)
print(f"==> Max hours: {MAX_HOURS}", flush=True)

sys.argv = [
    "inference_runner.py",
    "--solver_params_file", str(solver_params),
    "--max_hours", MAX_HOURS,
]

runpy.run_path(str(UTILITIES_DIR / "inference_runner.py"), run_name="__main__")
