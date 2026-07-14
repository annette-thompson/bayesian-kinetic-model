"""Checkpointed, resumable BlackJAX NUTS sampler.

Alpine caps jobs at 24h. This module runs NUTS in resumable chunks so a chain of
dependent SLURM jobs can pick up exactly where the previous one left off, run
longer on demand, and let you choose the tuning-vs-posterior boundary
retroactively (every warmup and sampling draw is persisted).

Design
------
BlackJAX's ``window_adaptation`` is internally a single ``scan`` over a
``(nuts_state, adaptation_state)`` pair driven by a per-step Stan ``schedule``
(see blackjax.adaptation.window_adaptation.window_adaptation). We reproduce that
loop but run it in fixed-size chunks, persisting the carried state after each
chunk. Because the per-chain RNG key is carried and split one step at a time,
resuming from a checkpoint continues the *same* Markov chain bit-for-bit --
running N steps as one chunk or as several interrupted chunks yields identical
draws (verified by the ``__main__`` self-test).

The pure-jax/blackjax core (:class:`ResumableSampler`) is model-agnostic: it
takes a ``logdensity_fn`` over a list-of-arrays position and a stack of initial
positions. The PyMC glue that produces those from a reaction model lives in
:func:`prepare_from_pymc` so the core stays testable with a toy log-density.

Crash consistency: each chunk writes draws to the zarr store *first*, then the
checkpoint. So the checkpoint's draw count is always <= what's in zarr; on
resume we truncate any un-checkpointed zarr tail. The checkpoint is the source
of truth.
"""
from __future__ import annotations

import dataclasses
import json
import os
import pickle
import time
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Any, Callable

import numpy as np

import jax
import jax.numpy as jnp
import blackjax
import blackjax.adaptation.window_adaptation as _wa
import zarr


# Per-draw sample statistics recorded for both warmup and sampling phases.
# name -> (extractor(info, new_state, step_size), numpy dtype)
_STAT_SPECS: dict[str, tuple[Callable[[Any, Any, Any], Any], str]] = {
    "lp": (lambda info, state, ss: state.logdensity, "f8"),
    "acceptance_rate": (lambda info, state, ss: info.acceptance_rate, "f8"),
    "diverging": (lambda info, state, ss: info.is_divergent.astype(jnp.int8), "i1"),
    "energy": (lambda info, state, ss: info.energy, "f8"),
    "tree_depth": (lambda info, state, ss: info.num_trajectory_expansions.astype(jnp.int32), "i4"),
    "n_steps": (lambda info, state, ss: info.num_integration_steps.astype(jnp.int32), "i4"),
    "step_size": (lambda info, state, ss: ss, "f8"),
}
_STAT_NAMES = list(_STAT_SPECS)


@dataclass(frozen=True)
class SamplerSpec:
    """Immutable configuration for a resumable run."""

    n_tune: int
    n_draws: int
    n_chains: int
    target_accept: float = 0.8
    is_mass_matrix_diagonal: bool = True
    initial_step_size: float = 1.0
    checkpoint_every: int = 50
    random_seed: int = 0

    def to_json_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


@dataclass
class RunStatus:
    """Outcome of a :meth:`ResumableSampler.run` call."""

    phase: str          # "warmup" | "sampling" | "done"
    warmup_done: int
    sampling_done: int
    n_tune: int
    n_draws: int
    stopped_reason: str  # "completed" | "time_budget" | "nothing_to_do"
    n_invocations: int = 0  # how many run() calls (i.e. job segments) so far

    @property
    def is_done(self) -> bool:
        return self.phase == "done"

    def to_json_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self) | {"is_done": self.is_done}


# --------------------------------------------------------------------------- #
# Pytree <-> numpy helpers for pickling blackjax NamedTuple state.
# --------------------------------------------------------------------------- #
def _to_numpy(tree: Any) -> Any:
    return jax.tree_util.tree_map(np.asarray, tree)


def _to_jax(tree: Any) -> Any:
    return jax.tree_util.tree_map(jnp.asarray, tree)


# --------------------------------------------------------------------------- #
# Chunked step factories (jitted). Each closes over the log-density and the
# blackjax kernel/adaptation functions (all static), taking the carried state
# plus a runtime schedule slice so chunks of equal length reuse one compile.
# --------------------------------------------------------------------------- #
def _make_warmup_chunk(
    logdensity_fn: Callable,
    target_accept: float,
    is_mass_matrix_diagonal: bool,
) -> tuple[Callable, Callable, Callable]:
    kernel = blackjax.nuts.build_kernel()
    adapt_init, adapt_step, adapt_final = _wa.base(
        is_mass_matrix_diagonal, target_acceptance_rate=target_accept
    )

    def _one_step(carry, stage):
        nuts_state, adapt_state, key = carry
        key, subkey = jax.random.split(key)
        new_state, info = kernel(
            subkey,
            nuts_state,
            logdensity_fn,
            adapt_state.step_size,
            adapt_state.inverse_mass_matrix,
        )
        new_adapt = adapt_step(adapt_state, stage, new_state.position, info.acceptance_rate)
        stats = tuple(spec[0](info, new_state, new_adapt.step_size) for spec in _STAT_SPECS.values())
        return (new_state, new_adapt, key), (new_state.position, stats)

    @jax.jit
    def run_chunk(nuts_state, adapt_state, keys, schedule_chunk):
        def per_chain(ns, ast, key):
            (fns, fast, fkey), (positions, stats) = jax.lax.scan(
                _one_step, (ns, ast, key), schedule_chunk
            )
            return fns, fast, fkey, positions, stats

        return jax.vmap(per_chain)(nuts_state, adapt_state, keys)

    return run_chunk, adapt_init, adapt_final


def _make_sampling_chunk(logdensity_fn: Callable) -> Callable:
    kernel = blackjax.nuts.build_kernel()

    def _one_step(carry, _):
        nuts_state, key, step_size, imm = carry
        key, subkey = jax.random.split(key)
        new_state, info = kernel(subkey, nuts_state, logdensity_fn, step_size, imm)
        stats = tuple(spec[0](info, new_state, step_size) for spec in _STAT_SPECS.values())
        return (new_state, key, step_size, imm), (new_state.position, stats)

    @partial(jax.jit, static_argnums=(4,))
    def run_chunk(nuts_state, keys, step_sizes, imms, n_steps):
        def per_chain(ns, key, step_size, imm):
            (fns, fkey, _, _), (positions, stats) = jax.lax.scan(
                _one_step, (ns, key, step_size, imm), None, length=n_steps
            )
            return fns, fkey, positions, stats

        return jax.vmap(per_chain)(nuts_state, keys, step_sizes, imms)

    return run_chunk


# --------------------------------------------------------------------------- #
# Durable draw store (zarr): resizable (chains, draws, *shape) arrays.
# --------------------------------------------------------------------------- #
class _DrawStore:
    def __init__(self, path: Path, var_names: list[str], n_chains: int, chunk_len: int):
        self._root = zarr.open_group(store=zarr.storage.LocalStore(str(path)), mode="a")
        self._var_names = var_names
        self._n_chains = n_chains
        self._chunk_len = chunk_len

    def _ensure(self, group: str, name: str, tail_shape: tuple[int, ...], dtype: str):
        key = f"{group}/{name}"
        if key not in self._root:
            self._root.create_array(
                key,
                shape=(self._n_chains, 0, *tail_shape),
                chunks=(self._n_chains, self._chunk_len, *tail_shape),
                dtype=dtype,
            )
        return self._root[key]

    def append(self, phase: str, positions: list[np.ndarray], stats: tuple[np.ndarray, ...]) -> None:
        """positions[v]: (chains, L, *shape); stats[s]: (chains, L)."""
        for name, arr in zip(self._var_names, positions):
            z = self._ensure(phase, name, arr.shape[2:], "f8")
            n0 = z.shape[1]
            z.resize((self._n_chains, n0 + arr.shape[1], *arr.shape[2:]))
            z[:, n0:] = np.asarray(arr, dtype="f8")
        for name, arr in zip(_STAT_NAMES, stats):
            dtype = _STAT_SPECS[name][1]
            z = self._ensure(f"{phase}_stats", name, (), dtype)
            n0 = z.shape[1]
            z.resize((self._n_chains, n0 + arr.shape[1]))
            z[:, n0:] = np.asarray(arr, dtype=dtype)

    def length(self, phase: str) -> int:
        key = f"{phase}/{self._var_names[0]}"
        return self._root[key].shape[1] if key in self._root else 0

    def truncate(self, phase: str, n: int) -> None:
        """Drop any rows beyond ``n`` (un-checkpointed tail after a crash)."""
        for group in (phase, f"{phase}_stats"):
            names = self._var_names if group == phase else _STAT_NAMES
            for name in names:
                key = f"{group}/{name}"
                if key in self._root and self._root[key].shape[1] > n:
                    z = self._root[key]
                    z.resize((self._n_chains, n, *z.shape[2:]))

    def read(self, phase: str) -> dict[str, np.ndarray]:
        out = {}
        for name in self._var_names:
            key = f"{phase}/{name}"
            if key in self._root:
                out[name] = self._root[key][:]
        return out

    def read_stats(self, phase: str) -> dict[str, np.ndarray]:
        out = {}
        for name in _STAT_NAMES:
            key = f"{phase}_stats/{name}"
            if key in self._root:
                out[name] = self._root[key][:]
        return out


# --------------------------------------------------------------------------- #
# Orchestrator
# --------------------------------------------------------------------------- #
class ResumableSampler:
    """Runs/resumes a checkpointed NUTS chain under a wall-clock budget."""

    CHECKPOINT = "checkpoint.pkl"
    META = "checkpoint_meta.json"
    DRAWS = "draws.zarr"

    def __init__(
        self,
        logdensity_fn: Callable,
        initial_positions: list[np.ndarray],
        var_names: list[str],
        spec: SamplerSpec,
        run_dir: str | Path,
        config_hash: str,
    ):
        self._logp = logdensity_fn
        self._init_positions = [jnp.asarray(p) for p in initial_positions]
        self._var_names = var_names
        self._spec = spec
        self._run_dir = Path(run_dir)
        self._run_dir.mkdir(parents=True, exist_ok=True)
        self._config_hash = config_hash

        self._warmup_chunk, self._adapt_init, self._adapt_final = _make_warmup_chunk(
            logdensity_fn, spec.target_accept, spec.is_mass_matrix_diagonal
        )
        self._sampling_chunk = _make_sampling_chunk(logdensity_fn)
        self._store = _DrawStore(
            self._run_dir / self.DRAWS, var_names, spec.n_chains, spec.checkpoint_every
        )

    # -- checkpoint io ----------------------------------------------------- #
    @property
    def _ckpt_path(self) -> Path:
        return self._run_dir / self.CHECKPOINT

    def _load_checkpoint(self) -> dict[str, Any] | None:
        if not self._ckpt_path.exists():
            return None
        with open(self._ckpt_path, "rb") as fh:
            ckpt = pickle.load(fh)
        if ckpt.get("config_hash") != self._config_hash:
            raise ValueError(
                f"Refusing to resume: checkpoint config_hash {ckpt.get('config_hash')!r} "
                f"!= current {self._config_hash!r}. The solver config or model changed. "
                "Start a fresh run directory or restore the original config."
            )
        return ckpt

    def _write_checkpoint(self, ckpt: dict[str, Any]) -> None:
        tmp = self._ckpt_path.with_suffix(".pkl.tmp")
        with open(tmp, "wb") as fh:
            pickle.dump(ckpt, fh, protocol=pickle.HIGHEST_PROTOCOL)
        tmp.replace(self._ckpt_path)
        meta = {
            "phase": ckpt["phase"],
            "warmup_done": ckpt["warmup_done"],
            "sampling_done": ckpt["sampling_done"],
            "n_tune": ckpt["n_tune"],
            "n_draws": ckpt["n_draws"],
            "n_chains": self._spec.n_chains,
            "config_hash": self._config_hash,
            "updated": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        with open(self._run_dir / self.META, "w", encoding="utf-8") as fh:
            json.dump(meta, fh, indent=2)

    def _fresh_state(self) -> dict[str, Any]:
        positions = self._init_positions
        nuts_state = jax.vmap(lambda p: blackjax.nuts.init(list(p), self._logp))(positions)
        adapt_state = jax.vmap(
            lambda p: self._adapt_init(list(p), self._spec.initial_step_size)
        )(positions)
        keys = jax.random.split(jax.random.PRNGKey(self._spec.random_seed), self._spec.n_chains)

        # With no warmup, sampling runs on an identity metric at the initial
        # step size (no adaptation) -- seed the tuned params so the sampling
        # phase can start immediately.
        tuned_step_size = tuned_imm = None
        if self._spec.n_tune == 0:
            dim = int(sum(int(np.prod(np.asarray(p).shape[1:])) for p in positions))
            tuned_step_size = np.full(self._spec.n_chains, self._spec.initial_step_size)
            tuned_imm = np.ones((self._spec.n_chains, dim))

        return {
            "phase": "warmup" if self._spec.n_tune > 0 else "sampling",
            "warmup_done": 0,
            "sampling_done": 0,
            "n_tune": self._spec.n_tune,
            "n_draws": self._spec.n_draws,
            "config_hash": self._config_hash,
            "nuts_state": _to_numpy(nuts_state),
            "adapt_state": _to_numpy(adapt_state),
            "tuned_step_size": tuned_step_size,
            "tuned_imm": tuned_imm,
            "rng_keys": np.asarray(keys),
        }

    # -- main loop --------------------------------------------------------- #
    def run(self, max_seconds: float | None = None, extra_draws: int = 0) -> RunStatus:
        """Advance the chain until complete or the time budget is exhausted.

        Parameters
        ----------
        max_seconds
            Wall-clock budget for this invocation (checked between chunks). None
            runs to completion.
        extra_draws
            Increase the posterior draw target by this many before running. Lets
            a follow-up job say "that wasn't enough, run longer".
        """
        start = time.perf_counter()
        ckpt = self._load_checkpoint() or self._fresh_state()
        ckpt["n_invocations"] = ckpt.get("n_invocations", 0) + 1

        if extra_draws:
            ckpt["n_draws"] += int(extra_draws)
            # Reactivate a finished run so "that wasn't enough, run longer" keeps
            # the same chain (tuned metric already frozen in the checkpoint).
            if ckpt["phase"] == "done" and ckpt["sampling_done"] < ckpt["n_draws"]:
                ckpt["phase"] = "sampling"

        # Reconcile any un-checkpointed zarr tail from a crashed prior job.
        self._store.truncate("warmup", ckpt["warmup_done"])
        self._store.truncate("sampling", ckpt["sampling_done"])

        schedule = np.asarray(_wa.build_schedule(self._spec.n_tune)) if self._spec.n_tune else None

        did_work = False

        def can_run() -> bool:
            # Always make at least one chunk of progress per invocation so a
            # chain of dependent jobs never stalls; then honor the budget. The
            # budget is checked between chunks, so a call may overrun by up to
            # one chunk -- size max_hours with margin (a chunk is small).
            if not did_work:
                return True
            return max_seconds is None or (time.perf_counter() - start) < max_seconds

        # ---- warmup phase ---- #
        while ckpt["phase"] == "warmup" and ckpt["warmup_done"] < ckpt["n_tune"] and can_run():
            s = ckpt["warmup_done"]
            length = min(self._spec.checkpoint_every, ckpt["n_tune"] - s)
            sched_chunk = jnp.asarray(schedule[s : s + length])
            nuts_state = _to_jax(ckpt["nuts_state"])
            adapt_state = _to_jax(ckpt["adapt_state"])
            keys = jnp.asarray(ckpt["rng_keys"])
            fns, fast, fkeys, positions, stats = self._warmup_chunk(
                nuts_state, adapt_state, keys, sched_chunk
            )
            jax.block_until_ready(fns)
            self._store.append("warmup", [np.asarray(p) for p in positions], [np.asarray(s_) for s_ in stats])
            ckpt.update(
                nuts_state=_to_numpy(fns),
                adapt_state=_to_numpy(fast),
                rng_keys=np.asarray(fkeys),
                warmup_done=s + length,
            )
            self._write_checkpoint(ckpt)
            did_work = True

        # ---- warmup -> sampling transition (freeze tuned params) ---- #
        if ckpt["phase"] == "warmup" and ckpt["warmup_done"] >= ckpt["n_tune"]:
            adapt_state = _to_jax(ckpt["adapt_state"])
            step_sizes, imms = jax.vmap(self._adapt_final)(adapt_state)
            ckpt.update(
                phase="sampling",
                tuned_step_size=np.asarray(step_sizes),
                tuned_imm=np.asarray(imms),
            )
            self._write_checkpoint(ckpt)

        # ---- sampling phase ---- #
        while ckpt["phase"] == "sampling" and ckpt["sampling_done"] < ckpt["n_draws"] and can_run():
            length = min(self._spec.checkpoint_every, ckpt["n_draws"] - ckpt["sampling_done"])
            nuts_state = _to_jax(ckpt["nuts_state"])
            keys = jnp.asarray(ckpt["rng_keys"])
            step_sizes = jnp.asarray(ckpt["tuned_step_size"])
            imms = jnp.asarray(ckpt["tuned_imm"])
            fns, fkeys, positions, stats = self._sampling_chunk(
                nuts_state, keys, step_sizes, imms, length
            )
            jax.block_until_ready(fns)
            self._store.append("sampling", [np.asarray(p) for p in positions], [np.asarray(s_) for s_ in stats])
            ckpt.update(
                nuts_state=_to_numpy(fns),
                rng_keys=np.asarray(fkeys),
                sampling_done=ckpt["sampling_done"] + length,
            )
            self._write_checkpoint(ckpt)
            did_work = True

        if ckpt["phase"] == "sampling" and ckpt["sampling_done"] >= ckpt["n_draws"]:
            ckpt["phase"] = "done"
            self._write_checkpoint(ckpt)

        if ckpt["phase"] == "done":
            reason = "completed"
        elif not did_work:
            reason = "nothing_to_do"
        else:
            reason = "time_budget"

        status = RunStatus(
            phase=ckpt["phase"],
            warmup_done=ckpt["warmup_done"],
            sampling_done=ckpt["sampling_done"],
            n_tune=ckpt["n_tune"],
            n_draws=ckpt["n_draws"],
            stopped_reason=reason,
            n_invocations=ckpt["n_invocations"],
        )
        with open(self._run_dir / "status.json", "w", encoding="utf-8") as fh:
            json.dump(status.to_json_dict(), fh, indent=2)
        return status

    # -- readback (analysis / finalize) ------------------------------------ #
    def read_draws(self, phase: str = "sampling") -> dict[str, np.ndarray]:
        """Unconstrained-space draws {var: (chains, draws, *shape)}."""
        return self._store.read(phase)

    def read_stats(self, phase: str = "sampling") -> dict[str, np.ndarray]:
        return self._store.read_stats(phase)


@dataclass
class PyMCBridge:
    """Everything the core sampler needs, extracted from a built PyMC model.

    ``logdensity_fn`` is a jax function over a list of unconstrained value-var
    arrays (blackjax position pytree); ``initial_positions`` is one stacked
    array per value var with a leading ``chains`` axis. ``value_var_names`` are
    the transformed (unconstrained) names used for on-disk storage.
    """

    logdensity_fn: Callable
    initial_positions: list[np.ndarray]
    value_var_names: list[str]
    config_hash: str


def prepare_from_pymc(
    model: Any,
    n_chains: int,
    random_seed: int = 0,
    jitter: bool = True,
    config_signature: str | None = None,
) -> PyMCBridge:
    """Build a :class:`PyMCBridge` from a PyMC model.

    Reuses ``pymc.sampling.jax`` so the resumable sampler inherits the exact
    unconstrained parameterization, transforms, and jittered-init logic PyMC's
    own blackjax/numpyro paths use -- only the sampling loop is replaced. Works
    with the custom ODE ``SolOp`` because it registers a ``jax_funcify`` handler
    (same path as nutpie backend="jax").
    """
    import hashlib

    from pymc.sampling.jax import get_jaxified_logp, _get_batched_jittered_initial_points

    # BlackJAX samples exp(logdensity_fn), so it needs +log p. PyMC's
    # confusingly-named flag returns +log p when negative_logp=True (the
    # default, and what PyMC's own blackjax path uses); negative_logp=False
    # returns -log p, the potential energy numpyro wants. Using the wrong sign
    # makes NUTS climb *away* from the mode -> runaway chains and a collapsed
    # mass matrix.
    logdensity_fn = get_jaxified_logp(model)
    initial_points = _get_batched_jittered_initial_points(
        model=model,
        chains=n_chains,
        initvals=None,
        random_seed=random_seed,
        logp_fn=logdensity_fn,
        jitter=jitter,
    )
    if n_chains == 1:  # helper returns unstacked list for a single chain
        initial_positions = [np.asarray(v)[None, ...] for v in initial_points]
    else:
        initial_positions = [np.asarray(v) for v in initial_points]

    value_var_names = [v.name for v in model.value_vars]

    if config_signature is None:
        config_signature = "|".join(
            f"{v.name}:{tuple(np.asarray(p).shape[1:])}"
            for v, p in zip(model.value_vars, initial_positions)
        )
    config_hash = hashlib.sha256(config_signature.encode("utf-8")).hexdigest()[:16]

    return PyMCBridge(logdensity_fn, initial_positions, value_var_names, config_hash)


def load_draws(run_dir: str | Path, phase: str = "sampling") -> dict[str, np.ndarray]:
    """Read persisted draws for a phase without constructing a sampler.

    Used by finalize/analysis tooling and the self-test. Returns
    {var: (chains, draws, *shape)}.
    """
    store_path = Path(run_dir) / ResumableSampler.DRAWS
    root = zarr.open_group(store=zarr.storage.LocalStore(str(store_path)), mode="r")
    if phase not in root:
        return {}
    return {name: root[f"{phase}/{name}"][:] for name in root[phase].array_keys()}


def load_stats(run_dir: str | Path, phase: str = "sampling") -> dict[str, np.ndarray]:
    """Read persisted per-draw sample statistics for a phase.

    Returns {stat: (chains, draws)} for the stats in ``_STAT_NAMES``.
    """
    store_path = Path(run_dir) / ResumableSampler.DRAWS
    root = zarr.open_group(store=zarr.storage.LocalStore(str(store_path)), mode="r")
    group = f"{phase}_stats"
    if group not in root:
        return {}
    return {name: root[f"{group}/{name}"][:] for name in root[group].array_keys()}


# --------------------------------------------------------------------------- #
# Self-test: a chain of SUBPROCESS segments (a faithful SLURM-chain simulation)
# must reproduce a single uninterrupted run bit-for-bit. Subprocesses also match
# real usage where each 24h resume is a fresh process.
# --------------------------------------------------------------------------- #
# Toy target shared by every worker: correlated 3-D Gaussian, importable (not a
# lambda) so subprocesses reconstruct it deterministically from source.
_TOY_PREC = np.linalg.inv(
    np.array([[1.0, 0.6, 0.2], [0.6, 1.0, 0.3], [0.2, 0.3, 1.0]])
)
_TOY_NCHAINS = 4


def _toy_logdensity(pos):
    x = pos[0]
    return -0.5 * x @ jnp.asarray(_TOY_PREC) @ x


def _selftest_worker(run_dir: str, checkpoint_every: int, max_seconds: float, n_tune: int, n_draws: int) -> None:
    spec = SamplerSpec(
        n_tune=n_tune, n_draws=n_draws, n_chains=_TOY_NCHAINS, checkpoint_every=checkpoint_every
    )
    init = [np.zeros((_TOY_NCHAINS, 3))]
    status = ResumableSampler(_toy_logdensity, init, ["x"], spec, run_dir, "toy-hash").run(
        max_seconds=max_seconds
    )
    print(json.dumps(status.to_json_dict()))


def _selftest() -> None:
    import shutil
    import subprocess
    import sys
    import tempfile

    n_tune, n_draws = 120, 200
    base = Path(tempfile.mkdtemp(prefix="resumable_selftest_"))

    def seg(run_dir: Path, checkpoint_every: int, max_seconds: str) -> dict[str, Any]:
        proc = subprocess.run(
            [sys.executable, __file__, "_seg", str(run_dir), str(checkpoint_every),
             max_seconds, str(n_tune), str(n_draws)],
            check=True, capture_output=True, text=True,
            env={**os.environ, "JAX_PLATFORMS": "cpu", "JAX_ENABLE_X64": "1"},
        )
        return json.loads(proc.stdout.strip().splitlines()[-1])

    try:
        # (A) One uninterrupted segment, big chunks.
        dir_a = base / "uninterrupted"
        assert seg(dir_a, n_draws, "inf")["is_done"]

        # (B) A chain of tiny-budget segments (one chunk each), fresh process
        #     every time, DIFFERENT chunk size -> proves boundary invariance.
        dir_b = base / "chained"
        for _ in range(60):
            if seg(dir_b, 37, "1e-6")["is_done"]:
                break
        else:
            raise AssertionError("chained segments never completed")

        a = load_draws(dir_a, "sampling")["x"]
        b = load_draws(dir_b, "sampling")["x"]
        assert a.shape == (_TOY_NCHAINS, n_draws, 3), a.shape
        max_abs = float(np.max(np.abs(a - b)))
        print(f"[selftest] sampling draws shape={a.shape}  max|A-B|={max_abs:.3e}")
        assert max_abs == 0.0, f"chained resume diverged from uninterrupted (max|A-B|={max_abs})"

        wa_draws = load_draws(dir_a, "warmup")["x"]
        assert wa_draws.shape == (_TOY_NCHAINS, n_tune, 3), wa_draws.shape
        emp = np.cov(np.concatenate(a, axis=0).T)
        print(f"[selftest] warmup retained shape={wa_draws.shape} (retroactive window OK)")
        print(f"[selftest] empirical cov diag={np.diag(emp).round(2)} (target [1,1,1])")
        print("[selftest] PASS: chained-subprocess resume is bit-identical to one run.")
    finally:
        shutil.rmtree(base, ignore_errors=True)


def _split_rhat(x: np.ndarray) -> float:
    """Split-R-hat for x of shape (chains, draws)."""
    half = x.shape[1] // 2
    s = np.concatenate([x[:, :half], x[:, half : 2 * half]], axis=0)
    n = s.shape[1]
    w = s.var(axis=1, ddof=1).mean()
    b = n * s.mean(axis=1).var(ddof=1)
    return float(np.sqrt(((n - 1) / n * w + b / n) / w))


def _pymc_check() -> None:
    """End-to-end check of the PyMC bridge: a well-conditioned model must
    recover its known posterior with R-hat~1 and no divergences. Guards against
    log-density sign / transform regressions (BlackJAX needs +log p)."""
    import shutil
    import tempfile

    import pymc as pm

    with pm.Model() as model:
        a = pm.LogNormal("a", 0.0, 1.0)
        b = pm.LogNormal("b", 0.0, 1.0)
        pm.Normal("ya", mu=a, sigma=0.3, observed=np.full(30, 2.0))
        pm.Normal("yb", mu=b, sigma=0.3, observed=np.full(30, 3.0))

    bridge = prepare_from_pymc(model, n_chains=4, random_seed=0)
    run_dir = Path(tempfile.mkdtemp(prefix="resumable_pymccheck_"))
    try:
        spec = SamplerSpec(n_tune=500, n_draws=500, n_chains=4, checkpoint_every=1000,
                           target_accept=0.9, random_seed=0)
        ResumableSampler(bridge.logdensity_fn, bridge.initial_positions,
                         bridge.value_var_names, spec, run_dir, bridge.config_hash).run()
        draws = load_draws(run_dir, "sampling")
        stats = _DrawStore(run_dir / ResumableSampler.DRAWS, bridge.value_var_names, 4, 1)
        n_div = int(stats.read_stats("sampling")["diverging"].sum())
        a_c, b_c = np.exp(draws["a_log__"]), np.exp(draws["b_log__"])
        ra, rb = _split_rhat(a_c), _split_rhat(b_c)
        print(f"[pymc-check] a mean {a_c.mean():.3f} (2.0) rhat {ra:.3f} | "
              f"b mean {b_c.mean():.3f} (3.0) rhat {rb:.3f} | divergences {n_div}")
        assert abs(a_c.mean() - 2.0) < 0.1 and abs(b_c.mean() - 3.0) < 0.15, "posterior means off"
        assert ra < 1.05 and rb < 1.05, "R-hat too high"
        assert n_div == 0, "unexpected divergences"
        print("[pymc-check] PASS: bridge recovers the known posterior cleanly.")
    finally:
        shutil.rmtree(run_dir, ignore_errors=True)


if __name__ == "__main__":
    import sys

    if len(sys.argv) >= 2 and sys.argv[1] == "_seg":
        _, _, run_dir, cev, msec, ntune, ndraws = sys.argv
        _selftest_worker(run_dir, int(cev), float(msec), int(ntune), int(ndraws))
    elif len(sys.argv) >= 2 and sys.argv[1] == "pymc-check":
        _pymc_check()
    else:
        _selftest()
