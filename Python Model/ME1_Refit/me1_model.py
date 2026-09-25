"""Solve the ME1 model at a given set of scaling parameters.

The network is built once and reused. A p_vec only changes 18 numbers inside the parameter
vector, so refitting never rebuilds or recompiles the ODE -- which is what makes a few
hundred objective evaluations practical.

Two conversions matter and are easy to get wrong, so they live in one place:

  * the YAML is written at the PUBLISHED p_vec with every scaling group at its no-op value,
    so a scaling group is a RELATIVE change. Multiplicative parameters take published x
    multiplier; d1 and d2 sit inside an exponential and take published + offset.
  * "C16 equivalents" weights each fatty acid by carbon number / 16, so a C8 counts half of
    a C16. The plain sum is a different quantity ("total product") and both appear in the
    objective.
"""
from __future__ import annotations

import re
import sys
from dataclasses import dataclass

import diffrax as dfrx
import jax
import jax.numpy as jnp
import numpy as np

import me1_config as cfg

sys.path.insert(0, str(cfg.PROJECT / "Utilities"))
from reaction_model_builder import (  # noqa: E402
    build_ode_system_from_reactions,
    discover_scaling_groups,
    set_scaling_group_values,
)

FA_RE = re.compile(r"^C(\d+)_FA(_unsat)?$")


def to_relative(p_abs: dict) -> dict:
    """Absolute p_vec -> the scaling-group values the reaction files expect."""
    out = {}
    for name, value in p_abs.items():
        base = cfg.PUBLISHED[name]
        out[name] = (value - base) if name in cfg.ADDITIVE else (value / base)
    return out


def to_absolute(p_rel: dict) -> dict:
    """Inverse of to_relative, for reporting a refit in the published parameterisation."""
    out = {}
    for name, value in p_rel.items():
        base = cfg.PUBLISHED[name]
        out[name] = (base + value) if name in cfg.ADDITIVE else (base * value)
    return out


@dataclass
class Solution:
    times: np.ndarray
    c16_equivalents: np.ndarray        # uM, carbon-weighted
    total_fa: np.ndarray               # uM, plain sum
    profile: np.ndarray                # (time, 14) per chain length, PROFILE_ORDER
    steps: int
    ok: bool


class ME1Model:
    """The ME1 network, solvable at any p_vec."""

    def __init__(self, reactions_dir=None, rtol=cfg.RTOL, atol=cfg.ATOL):
        self.dir = reactions_dir or cfg.REACTIONS
        groups = sorted(discover_scaling_groups(self.dir))
        # No-op values: 1 for multiplicative groups, 0 for the additive d-type ones. A
        # silent default of 1 everywhere would rescale TesA by e^d2 and has caused a real
        # error in this project before.
        nominal = {g: (0.0 if g in cfg.ADDITIVE else 1.0) for g in groups}
        net, species, params, values, _ = build_ode_system_from_reactions(
            self.dir, scaling_group=nominal)
        self.network, self.species, self.params = net, species, params
        self.theta0 = set_scaling_group_values(
            jnp.array([values[p] for p in self.params], dtype=jnp.float64),
            self.params, nominal)
        self.index_of = {s: i for i, s in enumerate(species)}
        self.rtol, self.atol = rtol, atol

        self.fa_species = [s for s in species if FA_RE.fullmatch(s)]
        self.fa_idx = jnp.asarray([self.index_of[s] for s in self.fa_species])
        self.fa_weights = jnp.asarray(
            [int(FA_RE.fullmatch(s).group(1)) / 16.0 for s in self.fa_species])
        # Column order for the profile, matching the MATLAB F_raw vector.
        self.profile_idx = []
        for n, unsat in cfg.PROFILE_ORDER:
            name = f"C{n}_FA_unsat" if unsat else f"C{n}_FA"
            self.profile_idx.append(self.index_of[name] if name in self.index_of else None)

    # ---------------------------------------------------------------- parameters
    def theta_for(self, p_abs: dict):
        """Parameter vector at an absolute p_vec, leaving everything else untouched."""
        return set_scaling_group_values(self.theta0, self.params, to_relative(p_abs))

    def rate_constants(self, p_abs: dict) -> dict:
        """Effective rate constant per reaction at this p_vec: scaling expression x the
        published constant. Used by the guard rails, and useful on its own for seeing what
        a proposed p_vec does to the underlying chemistry."""
        from reaction_model_builder import _eval_scale_expr, load_elementary_reactions
        theta = np.asarray(self.theta_for(p_abs))
        out = {}
        for r in load_elementary_reactions(self.dir):
            for key, val, grp in ((r.rate_const_key, r.rate_const_value, r.scaling_group),
                                  (r.rvs_rate_const_key, r.rvs_rate_const_value, r.rvs_scaling_group)):
                if key is None or val is None:
                    continue
                vals = val if isinstance(val, (list, tuple)) else [val]
                scale = 1.0 if grp is None else float(_eval_scale_expr(str(grp), self.params, theta))
                out.setdefault(key, []).extend(float(v) * scale for v in vals)
        return out

    # ---------------------------------------------------------------- initial state
    def y0(self, substrates: dict, enzymes: dict) -> np.ndarray:
        y = np.zeros(len(self.species), dtype=np.float64)
        for name, conc in {**substrates, **enzymes}.items():
            if name in self.index_of:
                y[self.index_of[name]] = float(conc)
        return y

    def condition_y0(self, condition: dict) -> np.ndarray:
        """Initial state for a condition.

        Accepts either form: `drop` (a list of enzymes set to zero, as the fitted
        conditions use) or `enzymes` (explicit overrides, which the held-out set needs
        because it raises concentrations as well as zeroing them).
        """
        subs = dict(cfg.BASE_SUBSTRATES)
        if "acetyl_coa" in condition:
            subs["C2_AcCoA"] = condition["acetyl_coa"]
        enz = dict(cfg.BASE_ENZYMES)
        for name in condition.get("drop", []):
            enz[name] = 0.0
        enz.update(condition.get("enzymes", {}))
        return self.y0(subs, enz)

    # ---------------------------------------------------------------- solving
    def _batch_solver(self):
        """One jitted, vmapped solve over a stack of initial states.

        The MATLAB called the solver once per condition in a loop, paying full setup each
        time. Here the seven conditions differ only in y0, so they are vectorised into a
        single call: one compilation, then every refit iteration solves them together.
        """
        if getattr(self, "_batch", None) is None:
            def one(theta, y0, ts):
                sol = dfrx.diffeqsolve(
                    dfrx.ODETerm(self.network), dfrx.Kvaerno5(),
                    t0=0.0, t1=ts[-1], dt0=1e-6, y0=y0, args=theta,
                    saveat=dfrx.SaveAt(ts=ts),
                    stepsize_controller=dfrx.PIDController(rtol=self.rtol, atol=self.atol),
                    max_steps=cfg.MAX_STEPS, throw=False)
                return sol.ys, sol.stats["num_steps"], sol.result == dfrx.RESULTS.successful
            self._batch = jax.jit(jax.vmap(one, in_axes=(None, 0, None)))
        return self._batch

    def solve_batch(self, p_abs: dict, y0_list, times):
        """Solutions for several initial states at once. Returns a list of Solution."""
        times = np.atleast_1d(np.asarray(times, dtype=float))
        ys, steps, ok = self._batch_solver()(
            self.theta_for(p_abs),
            jnp.asarray(np.stack(y0_list), dtype=jnp.float64),
            jnp.asarray(times))
        ys, steps, ok = np.asarray(ys), np.asarray(steps), np.asarray(ok)
        out = []
        for i in range(len(y0_list)):
            good = bool(ok[i]) and int(steps[i]) < cfg.MAX_STEPS
            if not good:
                nan = np.full(len(times), np.nan)
                out.append(Solution(times, nan, nan,
                                    np.full((len(times), len(self.profile_idx)), np.nan),
                                    int(steps[i]), False))
                continue
            block = ys[i][:, np.asarray(self.fa_idx)]
            prof = np.column_stack([ys[i][:, j] if j is not None else np.zeros(len(times))
                                    for j in self.profile_idx])
            out.append(Solution(times, block @ np.asarray(self.fa_weights),
                                block.sum(axis=1), prof, int(steps[i]), True))
        return out

    def initial_rates_c16(self, p_abs: dict, conditions=None):
        """C16 equivalents at the assay window for every condition, in one batched solve."""
        conditions = conditions or cfg.RATE_CONDITIONS
        sols = self.solve_batch(p_abs, [self.condition_y0(c) for c in conditions],
                                [cfg.RATE_WINDOW_S])
        return np.array([s.c16_equivalents[0] if s.ok else np.nan for s in sols])

    def solve(self, p_abs: dict, y0, times) -> Solution:
        times = np.atleast_1d(np.asarray(times, dtype=float))
        sol = dfrx.diffeqsolve(
            dfrx.ODETerm(self.network), dfrx.Kvaerno5(),
            t0=0.0, t1=float(times.max()), dt0=1e-6,
            y0=jnp.asarray(y0, dtype=jnp.float64), args=self.theta_for(p_abs),
            saveat=dfrx.SaveAt(ts=jnp.asarray(times)),
            stepsize_controller=dfrx.PIDController(rtol=self.rtol, atol=self.atol),
            max_steps=cfg.MAX_STEPS, throw=False)
        steps = int(np.asarray(sol.stats["num_steps"]))
        ok = steps < cfg.MAX_STEPS and bool(sol.result == dfrx.RESULTS.successful)
        if not ok:
            n = len(times)
            nan = np.full(n, np.nan)
            return Solution(times, nan, nan, np.full((n, len(self.profile_idx)), np.nan), steps, False)
        ys = np.asarray(sol.ys)
        block = ys[:, np.asarray(self.fa_idx)]
        prof = np.column_stack([ys[:, i] if i is not None else np.zeros(len(times))
                                for i in self.profile_idx])
        return Solution(times, block @ np.asarray(self.fa_weights), block.sum(axis=1),
                        prof, steps, True)

    def initial_rate_c16(self, p_abs: dict, condition: dict) -> float:
        """C16 equivalents accumulated by the assay window, in uM.

        A concentration, not a rate: dividing by the window to reach the uM/min the
        measurements use is the objective's job.
        """
        s = self.solve(p_abs, self.condition_y0(condition), [cfg.RATE_WINDOW_S])
        return float(s.c16_equivalents[0])
