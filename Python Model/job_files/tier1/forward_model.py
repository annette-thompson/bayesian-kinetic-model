"""Batched forward solves of one truncated FAS system, compiled once.

Shared by the Tier-1 post-processing that needs many solves at different parameter values
or enzyme concentrations: the expected-information grid (Fig 9), posterior-integrated
sensitivity (Fig 7) and the ratio response across posterior draws (Fig 8).

Scaling groups are entries of the parameter vector the network reads at run time, so one
compiled solve serves every parameter set and every set of initial conditions, with no rebuild
per draw. That also keeps each process to one compile: on the laptop, jaxlib 0.7.0's XLA:CPU
JIT can abort when a second model is compiled in the same process.

    fm = ForwardModel("C14+unsat", times=[150.0, 720.0])
    y0 = np.stack([fm.y0(changes) for _, changes in CONDITIONS])
    ys, ok = fm.run(y0, fm.theta({"a1": 1.2}))      # (conditions, times, species), (conditions,)
    obs = fm.observe(ys, ["C16 Equivalents (uM)"])  # name -> (conditions, times)
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent.parent
sys.path.insert(0, str(PROJECT / "Utilities"))
sys.path.insert(0, str(HERE))

import diffrax as dfrx
import jax
import jax.numpy as jnp
import numpy as np

import generate_chain_data as gcd
from experiment_framework import load_observable_definitions
from reaction_model_builder import discover_scaling_groups
from make_tier1_rate_data import CONDITIONS, FLOOR_CONC, NOISE_FRAC  # noqa: F401 (re-exported)

jax.config.update("jax_enable_x64", True)

CALC_MODULE = PROJECT / "Calculation Files/Full_FAS/FA_acylACP_conc.py"
PRODUCTION_PID = (0.4, 0.3, 0.0)
ENZYMES = ("FabD", "FabH", "FabG", "FabZ", "FabI", "FabF", "FabA", "FabB", "TesA")


def posterior_draws(source, params, n, key=None):
    """n parameter sets ({param: value}) thinned evenly from a pooled posterior.

    source is a run's posterior_samples_pm.nc (stranded chains already excluded at finalize),
    or a JSON written by job_files/export_posterior_series.py, with key "<set>/<system>"
    (e.g. "a1c3_no_floor/C14"). Both hold parameters on their natural scale.
    """
    source = Path(source)
    if source.suffix == ".nc":
        import arviz as az
        post = az.from_netcdf(source).posterior
        pooled = {p: np.asarray(post[p].values).ravel() for p in params}
    else:
        import json
        entry = json.loads(source.read_text())
        for part in key.split("/"):
            entry = entry[part]
        pooled = {p: np.asarray(entry["params"][p]).ravel() for p in params}
    size = len(next(iter(pooled.values())))
    idx = np.linspace(0, size - 1, n).round().astype(int)
    return [{p: float(pooled[p][i]) for p in params} for i in idx]


class ForwardModel:
    def __init__(self, system, times, rtol=1e-8, atol=1e-10, pid=PRODUCTION_PID,
                 max_steps=200_000, reactions=None, calc_module=CALC_MODULE):
        rx = PROJECT / "Reactions" / "EC_FAS_ME1" / (reactions or system)
        self.system = system
        self.groups = sorted(discover_scaling_groups(rx))
        self.nominal = gcd.nominal_scaling_group_overrides(self.groups)
        self._sys = gcd.ChainSystem(rx, rtol, atol, pcoeff=pid[0], icoeff=pid[1], dcoeff=pid[2],
                                    scaling_group_overrides=self.nominal)
        self.species = list(self._sys.species)
        self.index = dict(self._sys.index_of)
        self._param_index = {name: i for i, name in enumerate(self._sys.params)}
        self.times = jnp.asarray(times, dtype=jnp.float64)
        self.observables = load_observable_definitions(calc_module, species_names=self.species)
        network = self._sys.network
        controller = dfrx.PIDController(rtol=rtol, atol=atol, pcoeff=pid[0], icoeff=pid[1], dcoeff=pid[2])
        # Plain values, not the jnp array: inside jit a captured array becomes a tracer.
        t_end, save_ts = float(times[-1]), np.asarray(times, dtype=np.float64)

        def solve(y0, theta):
            sol = dfrx.diffeqsolve(
                dfrx.ODETerm(network), dfrx.Kvaerno5(), t0=0.0, t1=t_end, dt0=1e-6,
                y0=y0, args=theta, saveat=dfrx.SaveAt(ts=save_ts),
                stepsize_controller=controller, max_steps=max_steps, throw=False)
            return sol.ys, sol.result == dfrx.RESULTS.successful

        self._solve = jax.jit(jax.vmap(solve, in_axes=(0, None)))

    def theta(self, values=None):
        """Parameter vector at nominal scaling values, with `values` (group -> value) applied."""
        theta = self._sys.theta
        for group, value in (values or {}).items():
            if group not in self.nominal:
                raise KeyError(f"{group!r} is not a scaling group of {self.system}")
            theta = theta.at[self._param_index[group]].set(float(value))
        return theta

    def y0(self, changes=None):
        """Tier-1 baseline initial concentrations with `changes` (species -> uM) applied."""
        y = self._sys.y0()
        for name, conc in (changes or {}).items():
            y[self.index[name]] = float(conc)
        return y

    def run(self, y0_batch, theta):
        """Solve every row of y0_batch at theta: (rows, times, species) and a success flag per row."""
        ys, ok = self._solve(jnp.asarray(y0_batch, dtype=jnp.float64), theta)
        return np.asarray(ys), np.asarray(ok)

    def observe(self, ys, names):
        """Named observables for solved trajectories ys (rows, times, species): name -> (rows, times)."""
        out = {}
        for name in names:
            obs = self.observables[name]
            rows = [np.asarray(obs.compute(times=self.times, concentrations=jnp.asarray(y),
                                           species_index=self.index)[name]) for y in ys]
            out[name] = np.stack(rows)
        return out

    def names(self, pattern):
        """Observable names matching a regular expression, in the module's order."""
        import re
        rx = re.compile(pattern)
        return [name for name in self.observables if rx.fullmatch(name)]
