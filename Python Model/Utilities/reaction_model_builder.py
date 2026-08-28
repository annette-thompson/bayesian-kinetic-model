"""Compact-aware reaction network model builder.
Reads reaction definitions from YAML or JSON files.
"""

from __future__ import annotations

import re
import types
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import sympy
import yaml
import json as _json

import jax.numpy as jnp
import jax
import equinox as eqx
jax.config.update("jax_enable_x64", True)


def _extract_scale_param_names(expr: str) -> list[str]:
    """Return parameter names used in a scale expression.

    Uses sympy to parse the expression; math functions (exp, log, sqrt, etc.)
    are recognised as sympy builtins and never appear in ``free_symbols``.
    """
    return sorted(str(s) for s in sympy.parse_expr(expr).free_symbols)


def _compile_scale_expr(expr: str, param_idx: dict[str, int], jnp: Any) -> Any:
    """Compile a math expression string into a JAX callable ``(theta,) -> scalar``.

    sympy parses the expression and ``lambdify`` with ``modules=jnp`` routes all
    math calls (exp, log, sqrt, etc.) to their JAX equivalents automatically.
    """
    sym_expr = sympy.parse_expr(expr)
    symbols = sorted(sym_expr.free_symbols, key=str)
    lam = sympy.lambdify(symbols, sym_expr, modules=jnp)

    def scale_fn(theta: Any) -> Any:
        return lam(*(theta[param_idx[str(s)]] for s in symbols))

    return scale_fn


def _eval_scale_expr(expr: str, params: "list[str]", theta: Any) -> float:
    """Evaluate a scaling expression to a plain float using current theta values.

    Uses Python's ``math`` module so the result is always a plain Python float
    regardless of whether ``theta`` is a NumPy or JAX array.
    """
    import math as _math
    sym_expr = sympy.parse_expr(expr)
    symbols = sorted(sym_expr.free_symbols, key=str)
    param_idx = {name: i for i, name in enumerate(params)}
    lam = sympy.lambdify(symbols, sym_expr, modules=_math)
    return float(lam(*(float(theta[param_idx[str(s)]]) for s in symbols)))


@dataclass(frozen=True)
class ElementaryReaction:
    """Minimal class-style reaction representation for readable display and auditing."""

    rxn_name: str
    reactants: dict[str, float]
    products: dict[str, float]
    rate_const_key: str
    rate_const_value: float
    reversible: bool = False
    rvs_rate_const_key: str | None = None
    rvs_rate_const_value: float | None = None
    scaling_group: str | None = None
    rvs_scaling_group: str | None = None

    @staticmethod
    def _format_species(stoich_map: dict[str, float]) -> str:
        parts: list[str] = []
        for species, coeff in stoich_map.items():
            coeff_f = float(coeff)
            if np.isclose(coeff_f, 1.0):
                parts.append(species)
            else:
                if float(coeff_f).is_integer():
                    coeff_str = str(int(coeff_f))
                else:
                    coeff_str = f"{coeff_f:g}"
                parts.append(f"{coeff_str}{species}")
        return " + ".join(parts)

    def reaction_expression(self) -> str:
        lhs = self._format_species(self.reactants)
        rhs = self._format_species(self.products)
        arrow = "<-->" if self.reversible else "-->"
        return f"{lhs} {arrow} {rhs}"

    @property
    def rate_expression(self) -> str:
        """Symbolic mass-action rate term, e.g. 'scaling_group * k3_1f * FabH * C2_CoA'."""
        terms = (
            ([self.scaling_group] if self.scaling_group is not None else [])
            + [self.rate_const_key]
            + list(self.reactants.keys())
        )
        return " * ".join(terms)

    @property
    def rvs_rate_expression(self) -> "str | None":
        """Symbolic reverse mass-action rate term, or None if not reversible."""
        if not self.reversible or self.rvs_rate_const_key is None:
            return None
        terms = (
            ([self.rvs_scaling_group] if self.rvs_scaling_group is not None else [])
            + [self.rvs_rate_const_key]
            + list(self.products.keys())
        )
        return " * ".join(terms)

    @property
    def full_rate_expression(self) -> str:
        """Forward rate term, plus reverse term if reversible: 'k_f * A * B  <-->  k_r * C * D'."""
        fwd = self.rate_expression
        rev = self.rvs_rate_expression
        if rev is None:
            return fwd
        return f"{fwd}  <-->  {rev}"

    def __str__(self) -> str:
        return f"{self.rxn_name}: {self.reaction_expression()}"

    def to_mapping(self) -> dict[str, Any]:
        mapping: dict[str, Any] = {
            "rxn_name": self.rxn_name,
            "reactants": dict(self.reactants),
            "products": dict(self.products),
            "rate_const_key": self.rate_const_key,
            "rate_const_value": float(self.rate_const_value),
            "reversible": bool(self.reversible),
        }
        if self.reversible:
            if self.rvs_rate_const_key is None or self.rvs_rate_const_value is None:
                raise ValueError(
                    f"Reversible elementary reaction '{self.rxn_name}' must include reverse rate key/value."
                )
            mapping["rvs_rate_const_key"] = self.rvs_rate_const_key
            mapping["rvs_rate_const_value"] = float(self.rvs_rate_const_value)
        if self.scaling_group is not None:
            mapping["scaling_group"] = self.scaling_group
        if self.rvs_scaling_group is not None:
            mapping["rvs_scaling_group"] = self.rvs_scaling_group
        return mapping

    @classmethod
    def from_mapping(cls, reaction: dict[str, Any]) -> "ElementaryReaction":
        return cls(
            rxn_name=str(reaction.get("rxn_name", "")),
            reactants={k: float(v) for k, v in reaction["reactants"].items()},
            products={k: float(v) for k, v in reaction["products"].items()},
            rate_const_key=str(reaction["rate_const_key"]),
            rate_const_value=float(reaction["rate_const_value"]),
            reversible=bool(reaction.get("reversible", False)),
            rvs_rate_const_key=(
                str(reaction["rvs_rate_const_key"]) if "rvs_rate_const_key" in reaction else None
            ),
            rvs_rate_const_value=(
                float(reaction["rvs_rate_const_value"]) if "rvs_rate_const_value" in reaction else None
            ),
            scaling_group=(
                str(reaction["scaling_group"]) if reaction.get("scaling_group") is not None else None
            ),
            rvs_scaling_group=(
                str(reaction["rvs_scaling_group"]) if reaction.get("rvs_scaling_group") is not None else None
            ),
        )


class ReactionList(list):  # type: ignore[type-arg]
    """A list of ElementaryReaction objects that also carries the species index
    and exposes convenience query methods.

    Returned by ``load_elementary_reactions``.  Behaves exactly like a plain
    ``list[ElementaryReaction]`` everywhere a list is accepted.
    """

    def __init__(self, rxns: list["ElementaryReaction"], species: list[str]) -> None:
        super().__init__(rxns)
        self._species = species

        # Auto-compute params and scaling_params from reactions in encounter order,
        # matching the ordering used by build_ode_system_from_reactions so that
        # pm.* and sg.* namespace indices are consistent.
        _params: list[str] = []
        _scaling_params: list[str] = []
        _seen: set[str] = set()

        def _add(name: str, *, scaling: bool = False) -> None:
            if name not in _seen:
                _seen.add(name)
                _params.append(name)
                if scaling:
                    _scaling_params.append(name)

        for r in rxns:
            _add(r.rate_const_key)
            if r.rvs_rate_const_key is not None:
                _add(r.rvs_rate_const_key)
            for sg_expr in (r.scaling_group, r.rvs_scaling_group):
                if sg_expr is not None:
                    for name in _extract_scale_param_names(sg_expr):
                        _add(name, scaling=True)

        self._params: list[str] = _params
        self._scaling_params: list[str] = _scaling_params

    @property
    def species(self) -> list[str]:
        """Species name list associated with this reaction set."""
        return self._species

    def set_params(
        self,
        params: list[str],
        scaling_params: "list[str] | None" = None,
    ) -> "ReactionList":
        """Override the auto-computed parameter lists used by integer queries.

        Rarely needed — parameter and scaling-parameter lists are computed
        automatically from the loaded reactions.  Use this only when the external
        ``params`` list has a different ordering than the auto-computed one (e.g.
        when reactions were loaded separately from the ODE build step).
        Returns ``self`` for optional chaining.
        """
        self._params = list(params)
        self._scaling_params = list(scaling_params) if scaling_params is not None else None
        return self

    def get_ode_terms(self, name: "str | int") -> "Any":
        """Return a DataFrame of every ODE term contributing to a species.

        Parameters
        ----------
        name:
            Species index (``sp.ACP``) or species name string (``'ACP'``).
        """
        _df = get_ode_terms(name, list(self), self._species)
        try:
            get_ipython  # type: ignore[name-defined]  # noqa: F821
            return None  # display() already fired inside _display_table
        except NameError:
            return _df

    def query(
        self,
        *criteria: "str | int",
        return_type: str = "expression",
        use_regex: bool = False,
        theta: "Any | None" = None,
    ) -> None:
        """Find and print reactions matching all supplied criteria (AND logic).

        Pass alternating ``query, query_type`` pairs::

            rxns.query("FabA", "species")                          # single criterion
            rxns.query("FabA", "species", "kcat", "parameter")     # two criteria (AND)
            rxns.query("AcACP", "reactant", "c2", "scaling")       # three or more supported

        Parameters
        ----------
        *criteria:
            Alternating ``(query, query_type)`` values.  Must contain an even
            number of elements (≥ 2).  Each ``query`` is a substring
            (``str``) or an index into the relevant list (``int``, e.g.
            ``sp.ACP``, ``pm.kcat_H``).  Each ``query_type`` is one of the
            built-in aliases ``"species"``, ``"reactant"``, ``"product"``,
            ``"parameter"``, ``"scaling"`` or any ``ElementaryReaction``
            attribute name.
        return_type:
            What to display for each match.  One of ``"expression"``
            (default), ``"parameters"``, ``"scaling"``, ``"ode_terms"``,
            or ``"scaled_rate_constants"``.
        theta:
            Current parameter array (NumPy or JAX) — required for
            ``"scaled_rate_constants"``.
        use_regex:
            When ``True`` and a query is a string, interpret it as a regular
            expression pattern. Integer index queries always use exact
            matching.
        """
        if len(criteria) < 2 or len(criteria) % 2 != 0:
            raise ValueError(
                "query() requires alternating (query, query_type) pairs: "
                "query(crit1, type1) or query(crit1, type1, crit2, type2, ...)"
            )
        pairs = [(criteria[i], criteria[i + 1]) for i in range(0, len(criteria), 2)]
        _df = reaction_query(
            pairs, list(self), self._species,
            return_type=return_type,
            use_regex=use_regex,
            params=self._params,
            scaling_params=self._scaling_params,
            theta=theta,
        )
        try:
            get_ipython  # type: ignore[name-defined]  # noqa: F821
            return None  # display() already fired inside _display_table
        except NameError:
            return _df


def load_elementary_reactions(
    reactions_path: "str | Path | Sequence[str | Path]",
    schemas: frozenset[str] | set[str] | None = None,
) -> ReactionList:
    """Load and expand reactions from a YAML or JSON file/directory/file list.

    Parameters
    ----------
    reactions_path:
        Path to a YAML/JSON file, a directory of such files, or an explicit
        list of individual file paths to compose a system from.
    schemas:
        Optional set of accepted ``schema_version`` strings. When provided, only
        files whose ``schema_version`` appears in this set contribute reactions;
        files with a non-matching (or absent) version are skipped with a warning.
        When ``None``, all files are loaded but a warning is issued if their
        ``schema_version`` values are inconsistent (including mixing ``None`` with
        an explicit version).
    """
    raw = _load_reactions(reactions_path, schemas=schemas)
    all_species: list[str] = []
    seen: set[str] = set()
    for rxn in raw:
        for sp in list(rxn["reactants"]) + list(rxn["products"]):
            if sp not in seen:
                seen.add(sp)
                all_species.append(sp)
    return ReactionList([ElementaryReaction.from_mapping(rxn) for rxn in raw], all_species)

class ReactionNetwork(eqx.Module):
    """Auto-generated reaction network ODE system."""

    param_idx_arr: jnp.ndarray
    scale_fns: tuple = eqx.field(static=True)
    reactant_idx_arr: jnp.ndarray
    reactant_stoich_arr: jnp.ndarray
    reactant_mask_arr: jnp.ndarray
    stoich_matrix: jnp.ndarray

    def __init__(
        self,
        param_idx_arr: jnp.ndarray,
        scale_fns: tuple,
        reactant_idx_arr: jnp.ndarray,
        reactant_stoich_arr: jnp.ndarray,
        reactant_mask_arr: jnp.ndarray,
        stoich_matrix: jnp.ndarray,
    ):
        self.param_idx_arr = param_idx_arr
        self.scale_fns = scale_fns
        self.reactant_idx_arr = reactant_idx_arr
        self.reactant_stoich_arr = reactant_stoich_arr
        self.reactant_mask_arr = reactant_mask_arr
        self.stoich_matrix = stoich_matrix

    def __call__(self, t, y, args):
        if self.param_idx_arr.shape[0] == 0:
            return jnp.zeros_like(y)

        theta = jnp.asarray(args)
        # Clamp concentrations to be non-negative before computing rates.
        # Concentrations are physically >= 0, but the adaptive solver can
        # produce tiny negative overshoots. Raised to a stoichiometric power
        # (possibly non-integer) those negatives yield NaN, and large negative
        # excursions during NUTS warmup make mass-action terms blow up to inf
        # (which then crashes the implicit linear solver). Flooring at 0 keeps
        # the rate law well-defined without changing the physical dynamics.
        reactant_conc = jnp.maximum(y[self.reactant_idx_arr], 0.0)
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
    
def build_ode_system_from_reactions(
    reactions_source: "str | Path | Sequence[str | Path] | Sequence[dict[str, Any]] | Sequence[ElementaryReaction]",
    schemas: frozenset[str] | set[str] | None = None,
    scaling_group: dict[str, float] | None = None,
):
    """
    Build a generic ODE system from explicit or compact reaction specs.

    Accepted inputs:
    - Path to a YAML file or directory
    - List of individual YAML/JSON file paths (composes a system from
      specific files, e.g. a subset of enzymes from a larger library)
    - Preloaded list of reaction mappings
    - Preloaded list of ElementaryReaction objects

    Parameters
    ----------
    schemas:
        Optional set of accepted ``schema_version`` strings. See
        ``load_elementary_reactions`` for full semantics. Only applied when
        ``reactions_source`` is a path.
    scaling_group:
        Optional dict mapping scaling group names to their initial values.
        Groups found in reactions but absent from this dict default to 1.0.
        Groups can also be updated after building via ``set_scaling_group_values``.
    """

    reactions = _normalize_reaction_input(reactions_source, schemas=schemas)

    species: list[str] = []
    params: list[str] = []
    param_values: dict[str, float] = {}
    scaling_params: list[str] = []

    def append_unique(target: list[str], item: str) -> None:
        if item not in target:
            target.append(item)

    for reaction in reactions:
        for state in reaction["reactants"].keys():
            append_unique(species, state)
        for state in reaction["products"].keys():
            append_unique(species, state)

        k_fwd = reaction["rate_const_key"]
        append_unique(params, k_fwd)
        param_values[k_fwd] = float(reaction["rate_const_value"])

        if reaction.get("reversible", False):
            k_rev = reaction["rvs_rate_const_key"]
            append_unique(params, k_rev)
            param_values[k_rev] = float(reaction["rvs_rate_const_value"])

        sg_expr = reaction.get("scaling_group")
        if sg_expr is not None:
            for _sg_name in _extract_scale_param_names(sg_expr):
                append_unique(params, _sg_name)
                append_unique(scaling_params, _sg_name)
                if _sg_name not in param_values:
                    param_values[_sg_name] = (scaling_group or {}).get(_sg_name, 1.0)

        sg_rev_expr = reaction.get("rvs_scaling_group")
        if sg_rev_expr is not None:
            for _sg_name in _extract_scale_param_names(sg_rev_expr):
                append_unique(params, _sg_name)
                append_unique(scaling_params, _sg_name)
                if _sg_name not in param_values:
                    param_values[_sg_name] = (scaling_group or {}).get(_sg_name, 1.0)

    species_idx = {state: i for i, state in enumerate(species)}
    param_idx = {param: i for i, param in enumerate(params)}

    # Compile reactions into irreversible channels (forward + reverse when reversible)
    channels: list[dict[str, Any]] = []

    def add_channel(
        param_name: str,
        reactants: dict[str, float],
        products: dict[str, float],
        scale_expr: str | None = None,
    ) -> None:
        reactant_idxs = [species_idx[state] for state in reactants.keys()]
        reactant_stoich = [float(stoich) for stoich in reactants.values()]

        delta = np.zeros(len(species), dtype=np.float64)
        for state, stoich in reactants.items():
            delta[species_idx[state]] -= float(stoich)
        for state, stoich in products.items():
            delta[species_idx[state]] += float(stoich)

        channels.append(
            {
                "param_i": param_idx[param_name],
                "scale_expr": scale_expr,
                "reactant_idxs": reactant_idxs,
                "reactant_stoich": reactant_stoich,
                "delta": delta,
            }
        )

    for reaction in reactions:
        add_channel(
            reaction["rate_const_key"],
            reaction["reactants"],
            reaction["products"],
            scale_expr=reaction.get("scaling_group"),
        )
        if reaction.get("reversible", False):
            add_channel(
                reaction["rvs_rate_const_key"],
                reaction["products"],
                reaction["reactants"],
                scale_expr=reaction.get("rvs_scaling_group"),
            )

    n_channels = len(channels)
    max_reactants = max((len(ch["reactant_idxs"]) for ch in channels), default=0)

    # Dense channel tensors for JAX-friendly runtime computation
    param_idx_arr = np.zeros((n_channels,), dtype=np.int64)
    reactant_idx_arr = np.zeros((n_channels, max_reactants), dtype=np.int64)
    reactant_stoich_arr = np.zeros((n_channels, max_reactants), dtype=np.float64)
    reactant_mask_arr = np.zeros((n_channels, max_reactants), dtype=bool)
    stoich_matrix = np.zeros((len(species), n_channels), dtype=np.float64)

    for j, ch in enumerate(channels):
        param_idx_arr[j] = ch["param_i"]
        stoich_matrix[:, j] = ch["delta"]
        r_len = len(ch["reactant_idxs"])
        if r_len > 0:
            reactant_idx_arr[j, :r_len] = np.asarray(ch["reactant_idxs"], dtype=np.int64)
            reactant_stoich_arr[j, :r_len] = np.asarray(ch["reactant_stoich"], dtype=np.float64)
            reactant_mask_arr[j, :r_len] = True

    scale_fns: tuple = tuple(
        _compile_scale_expr(ch["scale_expr"], param_idx, jnp) if ch["scale_expr"] is not None else None
        for ch in channels
    )

    param_idx_arr = jnp.asarray(param_idx_arr)
    reactant_idx_arr = jnp.asarray(reactant_idx_arr)
    reactant_stoich_arr = jnp.asarray(reactant_stoich_arr)
    reactant_mask_arr = jnp.asarray(reactant_mask_arr)
    stoich_matrix = jnp.asarray(stoich_matrix)

    return (
        ReactionNetwork(
            param_idx_arr, scale_fns,
            reactant_idx_arr, reactant_stoich_arr, reactant_mask_arr, stoich_matrix,
        ),
        species,
        params,
        param_values,
        scaling_params,
    )


def _normalize_reaction_input(
    reactions_source: "str | Path | Sequence[str | Path] | Sequence[dict[str, Any]] | Sequence[ElementaryReaction]",
    schemas: frozenset[str] | set[str] | None = None,
) -> list[dict[str, Any]]:
    """Normalize supported reaction source formats into reaction mapping dictionaries."""
    if isinstance(reactions_source, (str, Path)):
        return _load_reactions(reactions_source, schemas=schemas)

    reactions_list = list(reactions_source)
    if not reactions_list:
        return []

    first = reactions_list[0]
    if isinstance(first, (str, Path)):
        return _load_reactions(reactions_list, schemas=schemas)  # type: ignore[arg-type]

    if isinstance(first, ElementaryReaction):
        return [rxn.to_mapping() for rxn in reactions_list]  # type: ignore[union-attr]

    if isinstance(first, dict):
        return reactions_list  # type: ignore[return-value]

    raise TypeError(
        "Unsupported reactions_source type. Expected path, list[str|Path], "
        "list[dict], or list[ElementaryReaction]."
    )


# ─────────────────────────────────────────────────────────────────────────────
# Known-field sets for per-reaction validation
# ─────────────────────────────────────────────────────────────────────────────

_SCALAR_FIELDS: frozenset[str] = frozenset({
    "rxn_name", "reactants", "products",
    "rate_const_key", "rate_const_value",
    "reversible",
    "rvs_rate_const_key", "rvs_rate_const_value",
    "scaling_group", "rvs_scaling_group",
})

_CHAIN_TEMPLATE_FIELDS: frozenset[str] = _SCALAR_FIELDS | frozenset({
    "chain_template", "chain_lengths",
    "token_offsets", "carbon_offset", "template_context",
})

_SAT_TEMPLATE_FIELDS: frozenset[str] = _CHAIN_TEMPLATE_FIELDS | frozenset({
    "saturation_template",
    "sat_chain_lengths", "unsat_chain_lengths",
    "sat_rate_const_key", "sat_rate_const_value", "sat_rvs_rate_const_key",
    "unsat_rate_const_key", "unsat_rate_const_value",
    "unsat_rvs_rate_const_key", "unsat_rvs_rate_const_value",
    "fwd_unsat_params_same_as_sat", "rvs_unsat_params_same_as_sat",
    "sat_scaling_group", "unsat_scaling_group",
})


def _validate_known_fields(
    reaction: dict[str, Any], source_file: str, idx: int
) -> None:
    """Raise ValueError if a reaction dict contains any unrecognised field name.

    Validation uses the raw reaction dict (before defaults are merged) so that
    default keys like ``sat_chain_lengths`` do not trigger false positives on
    scalar reactions.
    """
    if reaction.get("saturation_template", False):
        known = _SAT_TEMPLATE_FIELDS
    elif reaction.get("chain_template", False):
        known = _CHAIN_TEMPLATE_FIELDS
    else:
        known = _SCALAR_FIELDS
    unknown = set(reaction.keys()) - known
    if unknown:
        name = reaction.get("rxn_name", f"index {idx}")
        raise ValueError(
            f"Reaction '{name}' in {source_file} contains unknown field(s): "
            + ", ".join(sorted(unknown))
            + ". Check for typos or use the correct template type."
        )


def _load_reactions(
    reactions_path: "str | Path | Sequence[str | Path]",
    schemas: frozenset[str] | set[str] | None = None,
) -> list[dict[str, Any]]:
    """Load reactions from a YAML/JSON file, directory, or explicit list of files.

    ``reactions_path`` may be a single file, a directory (all ``*.yaml``/``*.json``
    files in it are loaded), or a list/tuple of individual file paths — the
    latter lets you compose a reaction system from files spread across a
    directory (e.g. specific enzymes from a larger reaction library) without
    copying them into a new folder.
    """

    def _parse_file(file_path: Path) -> dict[str, Any]:
        with open(file_path, "r", encoding="utf-8") as fh:
            if file_path.suffix.lower() == ".json":
                return _json.load(fh)
            return yaml.safe_load(fh)

    file_specs: list[tuple[str, dict[str, Any]]] = []
    if isinstance(reactions_path, (list, tuple)):
        if not reactions_path:
            raise ValueError("Reaction source list is empty.")
        for entry in reactions_path:
            file_path = Path(entry).expanduser().resolve()
            if not file_path.exists():
                raise FileNotFoundError(f"Reaction source not found: {file_path}")
            if file_path.is_dir():
                raise ValueError(
                    f"Reaction source list entries must be files, got a directory: {file_path}"
                )
            file_specs.append((str(file_path), _parse_file(file_path)))
        source_label = ", ".join(spec[0] for spec in file_specs)
    else:
        source_path = Path(reactions_path).expanduser().resolve()
        if not source_path.exists():
            raise FileNotFoundError(f"Reaction source not found: {source_path}")

        if source_path.is_dir():
            reaction_files = sorted(source_path.glob("*.yaml")) + sorted(source_path.glob("*.json"))
            if not reaction_files:
                raise ValueError(f"No YAML or JSON reaction files found in directory: {source_path}")
            for file_path in sorted(reaction_files, key=lambda p: p.name):
                file_specs.append((str(file_path), _parse_file(file_path)))
        else:
            file_specs.append((str(source_path), _parse_file(source_path)))
        source_label = str(source_path)

    required_keys = {
        "rxn_name",
        "reactants",
        "products",
        "rate_const_key",
        "rate_const_value",
        "reversible",
    }

    reactions: list[dict[str, Any]] = []
    seen_param_values: dict[str, tuple[float, str, str]] = {}

    if schemas is None:
        found_versions: set[Any] = {spec.get("schema_version") for _, spec in file_specs}
        if len(found_versions) > 1:
            warnings.warn(
                f"Inconsistent schema_version values found across files in {source_label}: "
                f"{sorted(str(v) for v in found_versions)}. "
                "Pass schemas= to filter to specific versions.",
                UserWarning,
                stacklevel=2,
            )

    for source_file, spec in file_specs:
        schema_version = spec.get("schema_version")
        if schemas is not None and schema_version not in schemas:
            warnings.warn(
                f"Skipping {source_file}: schema_version {schema_version!r} not in "
                f"requested schemas {sorted(schemas)}.",
                UserWarning,
                stacklevel=2,
            )
            continue

        file_defaults: dict[str, Any] = spec.get("defaults", {})
        unsat_suffix: str = file_defaults.get("unsat_suffix", "_unsat")
        sat_suffix: str = file_defaults.get("sat_suffix", "")

        if "reactions" not in spec or not isinstance(spec["reactions"], list):
            raise KeyError(f"Missing or invalid 'reactions' list in {source_file}")

        file_reactions: list[dict[str, Any]] = []

        for idx, reaction in enumerate(spec["reactions"]):
            if not isinstance(reaction, dict):
                raise TypeError(f"Reaction index {idx} in {source_file} must be a dict.")

            _validate_known_fields(reaction, source_file, idx)
            reaction = {**file_defaults, **reaction}
            expanded = _expand_compact_reaction(reaction, source_file, idx, unsat_suffix, sat_suffix)
            for entry in expanded:
                missing_keys = sorted(required_keys.difference(entry.keys()))
                if missing_keys:
                    raise KeyError(
                        f"Reaction '{entry.get('rxn_name', idx)}' in {source_file} missing keys: {missing_keys}"
                    )

                if not isinstance(entry["reactants"], dict) or not isinstance(entry["products"], dict):
                    raise TypeError(
                        f"Reaction '{entry.get('rxn_name', idx)}' in {source_file} must use dict stoichiometry."
                    )

                _validate_param_conflict(
                    seen_param_values=seen_param_values,
                    param_key=entry["rate_const_key"],
                    param_value=float(entry["rate_const_value"]),
                    source_file=source_file,
                    reaction_name=entry.get("rxn_name", f"reaction_{idx}"),
                )

                if entry.get("reversible", False):
                    if "rvs_rate_const_key" not in entry or "rvs_rate_const_value" not in entry:
                        raise KeyError(
                            f"Reversible reaction '{entry.get('rxn_name', idx)}' in {source_file} is missing reverse-rate keys."
                        )
                    _validate_param_conflict(
                        seen_param_values=seen_param_values,
                        param_key=entry["rvs_rate_const_key"],
                        param_value=float(entry["rvs_rate_const_value"]),
                        source_file=source_file,
                        reaction_name=entry.get("rxn_name", f"reaction_{idx}"),
                    )

                file_reactions.append(entry)

        declared_enzymes: list[str] = spec.get("enzymes", [])
        if declared_enzymes:
            all_species: set[str] = set()
            for entry in file_reactions:
                all_species.update(entry["reactants"].keys())
                all_species.update(entry["products"].keys())
            for enzyme in declared_enzymes:
                if enzyme not in all_species:
                    raise ValueError(
                        f"Declared enzyme '{enzyme}' in {source_file} does not appear as a species "
                        "in any expanded reaction. Check enzyme name and reaction templates."
                    )

        reactions.extend(file_reactions)

    if not reactions:
        raise ValueError(f"No reactions loaded from source: {source_label}")
    return reactions


def _expand_sat_compact_reaction(
    reaction: dict[str, Any],
    source_file: str,
    idx: int,
    token_offsets: dict[str, int],
    unsat_suffix: str,
    sat_suffix: str,
    base_name: str,
    reactants_template: dict[str, Any],
    products_template: dict[str, Any],
    template_context: dict[str, Any],
) -> list[dict[str, Any]]:
    has_explicit_sat_fwd_key = "sat_rate_const_key" in reaction
    sat_fwd_key = reaction.get("sat_rate_const_key", reaction.get("rate_const_key"))
    if sat_fwd_key is None:
        raise KeyError(
            f"Compact_sat reaction '{base_name}' in {source_file} requires rate_const_key or sat_rate_const_key."
        )
    base_fwd_key = str(sat_fwd_key)
    has_explicit_sat_rev_key = "sat_rvs_rate_const_key" in reaction
    base_rev_key = str(reaction.get("sat_rvs_rate_const_key", reaction.get("rvs_rate_const_key", "")))

    sat_chain_lengths = reaction.get("sat_chain_lengths")
    unsat_chain_lengths = reaction.get("unsat_chain_lengths")
    sat_chains = [int(x) for x in sat_chain_lengths] if isinstance(sat_chain_lengths, list) else []
    unsat_chains = [int(x) for x in unsat_chain_lengths] if isinstance(unsat_chain_lengths, list) else []
    if not sat_chains and not unsat_chains:
        raise ValueError(
            f"Compact_sat reaction '{base_name}' in {source_file} requires at least one of "
            "sat_chain_lengths or unsat_chain_lengths to be non-empty."
        )
    sat_index = {chain: i for i, chain in enumerate(sat_chains)}

    fwd_sat_values: list[float] = []
    fwd_shared = False
    rev_sat_values: list[float] | None = None
    rev_shared = False
    if sat_chains:
        sat_fwd_values_raw = reaction.get("rate_const_value", reaction.get("sat_rate_const_value"))
        fwd_sat_values, fwd_shared = _as_list_and_shared(
            sat_fwd_values_raw,
            len(sat_chains),
            "rate_const_value",
            reaction,
            source_file,
            idx,
        )
        if reaction.get("reversible", False):
            sat_rev_values_raw = reaction.get("rvs_rate_const_value", reaction.get("sat_rvs_rate_const_value"))
            rev_sat_values, rev_shared = _as_list_and_shared(
                sat_rev_values_raw,
                len(sat_chains),
                "rvs_rate_const_value",
                reaction,
                source_file,
                idx,
            )

    reversible = bool(reaction.get("reversible", False))
    expanded: list[dict[str, Any]] = []
    # sat_scaling_group takes priority over scaling_group for sat branch;
    # unsat_scaling_group takes priority for unsat branch; both fall back to scaling_group.
    _sat_sg = reaction.get("sat_scaling_group", reaction.get("scaling_group"))
    _unsat_sg = reaction.get("unsat_scaling_group", reaction.get("scaling_group"))
    _rvs_sg = reaction.get("rvs_scaling_group")

    if "fwd_unsat_params_same_as_sat" in reaction:
        match_unsat_forward = bool(reaction["fwd_unsat_params_same_as_sat"]) and bool(sat_chains)
    else:
        # Infer: separate unsat keys provided → False; otherwise share sat params → True (only if sat_chains non-empty)
        match_unsat_forward = bool(sat_chains) and not (
            "unsat_rate_const_key" in reaction or "unsat_rate_const_value" in reaction
        )

    if "rvs_unsat_params_same_as_sat" in reaction:
        match_unsat_reverse = bool(reaction["rvs_unsat_params_same_as_sat"]) and bool(sat_chains)
    else:
        # Infer: separate unsat reverse keys provided → False; otherwise share sat params → True (only if sat_chains non-empty)
        match_unsat_reverse = bool(sat_chains) and not (
            "unsat_rvs_rate_const_key" in reaction or "unsat_rvs_rate_const_value" in reaction
        )

    if (match_unsat_forward or match_unsat_reverse) and unsat_chains:
        for chain in unsat_chains:
            if chain not in sat_index:
                raise ValueError(
                    f"Compact_sat reaction '{base_name}' in {source_file}: unsat chain {chain} missing in sat_chain_lengths."
                )

    # Determine actual sat keys: auto-append sat_suffix when using a shared rate_const_key and params differ
    actual_sat_fwd_key = base_fwd_key if (has_explicit_sat_fwd_key or match_unsat_forward) else base_fwd_key + sat_suffix
    actual_sat_rev_key = base_rev_key if (has_explicit_sat_rev_key or not reversible or match_unsat_reverse) else base_rev_key + sat_suffix

    if sat_chains:
        expanded.extend(
            _expand_compact_branch(
                base_name=f"{base_name}{sat_suffix}",
                chains=sat_chains,
                token_offsets=token_offsets,
                template_context=template_context,
                reactants_template=reactants_template,
                products_template=products_template,
                reversible=reversible,
                base_fwd_key=actual_sat_fwd_key,
                base_rev_key=actual_sat_rev_key,
                fwd_values=fwd_sat_values,
                rev_values=rev_sat_values,
                fwd_shared=fwd_shared,
                rev_shared=rev_shared,
                species_suffix=sat_suffix,
                scaling_group=_sat_sg,
                rvs_scaling_group=_rvs_sg,
            )
        )

    if not unsat_chains:
        return expanded

    if match_unsat_forward:
        unsat_fwd_key = actual_sat_fwd_key
        unsat_fwd_values = [float(fwd_sat_values[sat_index[chain]]) for chain in unsat_chains]
        unsat_fwd_shared = fwd_shared
    elif has_explicit_sat_fwd_key:
        # Explicit sat key was provided; require explicit unsat key too
        unsat_fwd_key = str(reaction.get("unsat_rate_const_key", ""))
        if not unsat_fwd_key:
            raise KeyError(
                f"Compact_sat reaction '{base_name}' in {source_file} with fwd_unsat_params_same_as_sat=false requires unsat_rate_const_key."
            )
        unsat_fwd_values, unsat_fwd_shared = _as_list_and_shared(
            reaction.get("unsat_rate_const_value"),
            len(unsat_chains),
            "unsat_rate_const_value",
            reaction,
            source_file,
            idx,
        )
    else:
        # Auto-suffix: derive unsat key from shared base key
        unsat_fwd_key = base_fwd_key + unsat_suffix
        unsat_fwd_values, unsat_fwd_shared = _as_list_and_shared(
            reaction.get("unsat_rate_const_value"),
            len(unsat_chains),
            "unsat_rate_const_value",
            reaction,
            source_file,
            idx,
        )

    unsat_rev_key = actual_sat_rev_key
    unsat_rev_shared = rev_shared
    if reversible:
        if match_unsat_reverse:
            if rev_sat_values is None:
                raise ValueError(
                    f"Compact_sat reaction '{base_name}' in {source_file} requested reverse sat mapping but is missing reverse sat values."
                )
            unsat_rev_values: list[float] | None = [float(rev_sat_values[sat_index[chain]]) for chain in unsat_chains]
            unsat_rev_shared = rev_shared
        elif has_explicit_sat_rev_key:
            # Explicit sat rev key was provided; require explicit unsat rev key too
            unsat_rev_key = str(reaction.get("unsat_rvs_rate_const_key", ""))
            if not unsat_rev_key:
                raise KeyError(
                    f"Compact_sat reaction '{base_name}' in {source_file} with rvs_unsat_params_same_as_sat=false requires unsat_rvs_rate_const_key."
                )
            unsat_rev_values, unsat_rev_shared = _as_list_and_shared(
                reaction.get("unsat_rvs_rate_const_value"),
                len(unsat_chains),
                "unsat_rvs_rate_const_value",
                reaction,
                source_file,
                idx,
            )
        else:
            # Auto-suffix: derive unsat rev key from shared base rev key
            unsat_rev_key = base_rev_key + unsat_suffix
            unsat_rev_values, unsat_rev_shared = _as_list_and_shared(
                reaction.get("unsat_rvs_rate_const_value"),
                len(unsat_chains),
                "unsat_rvs_rate_const_value",
                reaction,
                source_file,
                idx,
            )
    else:
        unsat_rev_values = None

    expanded.extend(
        _expand_compact_branch(
            base_name=f"{base_name}{unsat_suffix}",
            chains=unsat_chains,
            token_offsets=token_offsets,
            template_context=template_context,
            reactants_template=reactants_template,
            products_template=products_template,
            reversible=bool(reaction.get("reversible", False)),
            base_fwd_key=unsat_fwd_key,
            base_rev_key=unsat_rev_key,
            fwd_values=unsat_fwd_values,
            rev_values=unsat_rev_values,
            fwd_shared=unsat_fwd_shared,
            rev_shared=unsat_rev_shared,
            species_suffix=unsat_suffix,
            scaling_group=_unsat_sg,
            rvs_scaling_group=_rvs_sg,
        )
    )
    return expanded


def _expand_chain_compact_reaction(
    reaction: dict[str, Any],
    source_file: str,
    idx: int,
    token_offsets: dict[str, int],
    unsat_suffix: str,
    base_name: str,
    reactants_template: dict[str, Any],
    products_template: dict[str, Any],
    template_context: dict[str, Any],
) -> list[dict[str, Any]]:
    chain_lengths = reaction.get("chain_lengths")
    if not isinstance(chain_lengths, list) or not chain_lengths:
        raise ValueError(
            f"Compact_chain reaction '{base_name}' in {source_file} requires non-empty chain_lengths."
        )

    base_fwd_key = str(reaction["rate_const_key"])
    base_rev_key = str(reaction.get("rvs_rate_const_key", ""))

    fwd_values, fwd_shared = _as_list_and_shared(
        reaction.get("rate_const_value"), len(chain_lengths), "rate_const_value", reaction, source_file, idx
    )
    rev_values: list[float] | None = None
    rev_shared = False
    if reaction.get("reversible", False):
        rev_values, rev_shared = _as_list_and_shared(
            reaction.get("rvs_rate_const_value"), len(chain_lengths), "rvs_rate_const_value", reaction, source_file, idx
        )

    return _expand_compact_branch(
        base_name=base_name,
        chains=[int(x) for x in chain_lengths],
        token_offsets=token_offsets,
        template_context=template_context,
        reactants_template=reactants_template,
        products_template=products_template,
        reversible=bool(reaction.get("reversible", False)),
        base_fwd_key=base_fwd_key,
        base_rev_key=base_rev_key,
        fwd_values=fwd_values,
        rev_values=rev_values,
        fwd_shared=fwd_shared,
        rev_shared=rev_shared,
        scaling_group=reaction.get("scaling_group"),
        rvs_scaling_group=reaction.get("rvs_scaling_group"),
    )


def _expand_compact_reaction(
    reaction: dict[str, Any], source_file: str, idx: int, unsat_suffix: str, sat_suffix: str
) -> list[dict[str, Any]]:
    """
    Expand compact reaction definitions into explicit scalar reactions.

    Hard-cut compact rules:
    - chain_template: true is the only supported compact marker.
    - saturation_template: true enables combined sat/unsat expansion using sat_chain_lengths
      and unsat_chain_lengths.
        - fwd_unsat_params_same_as_sat: true maps unsat forward terms to sat parameter indices
            by chain value (for example unsat 12 uses sat C12 slot).
        - rvs_unsat_params_same_as_sat (optional) controls reverse-term mapping
            independently; defaults to fwd_unsat_params_same_as_sat when omitted.
    - rxn_name, rate_const_key, rvs_rate_const_key, scaling_group and all *_rate_const_key
      fields accept {chain} and any token_offsets tokens (e.g. {intermediate}) via
      str.format_map.  Example: rxn_name: "C{chain}_FabH_binding", rate_const_key: "C{chain}_kon".
    - token_offsets defines named tokens derived from chain (e.g. {"intermediate": 3}
      makes {intermediate} = chain+3 available in all template fields).
    - carbon_offset is accepted as legacy shorthand for token_offsets: {"carbon": n}.
    """
    if not reaction.get("chain_template", False):
        return [reaction]

    reactants_template = reaction.get("reactants")
    products_template = reaction.get("products")
    if not reactants_template or not products_template:
        raise KeyError(
            f"Compact reaction '{reaction.get('rxn_name', idx)}' in {source_file} requires reactants/products templates."
        )
    if not isinstance(reactants_template, dict) or not isinstance(products_template, dict):
        raise TypeError(
            f"Compact reaction '{reaction.get('rxn_name', idx)}' in {source_file} templates must be dicts."
        )

    token_offsets = _parse_token_offsets(reaction)
    base_name = str(reaction.get("rxn_name", f"reaction_{idx}"))
    template_context: dict[str, Any] = dict(reaction.get("template_context", {}))

    if reaction.get("saturation_template", False):
        return _expand_sat_compact_reaction(
            reaction, source_file, idx, token_offsets, unsat_suffix, sat_suffix, base_name,
            reactants_template, products_template, template_context
        )

    return _expand_chain_compact_reaction(
        reaction, source_file, idx, token_offsets, unsat_suffix, base_name,
        reactants_template, products_template, template_context
    )


def _expand_compact_branch(
    *,
    base_name: str,
    chains: list[int],
    token_offsets: dict[str, int],
    template_context: dict[str, Any],
    reactants_template: dict[str, Any],
    products_template: dict[str, Any],
    reversible: bool,
    base_fwd_key: str,
    base_rev_key: str,
    fwd_values: list[float],
    rev_values: list[float] | None,
    fwd_shared: bool,
    rev_shared: bool,
    species_suffix: str = "",
    scaling_group: str | None = None,
    rvs_scaling_group: str | None = None,
) -> list[dict[str, Any]]:
    # A species gets the suffix only if its template contains a chain-varying token.
    # Fixed species like "C3_MalACP", "ACP", "CoA" have no tokens and are never suffixed.
    # template_context provides static tokens; chain and token_offset keys take precedence.
    chain_tokens = frozenset({"chain"} | token_offsets.keys())
    expanded: list[dict[str, Any]] = []
    for i, chain in enumerate(chains):
        token_values: dict[str, Any] = {**template_context, "chain": chain, **{k: chain + v for k, v in token_offsets.items()}}

        exp_reactants: dict[str, float] = {}
        for name, stoich in reactants_template.items():
            rendered = _render_species_name(name, token_values)
            if species_suffix and any(f"{{{t}}}" in name for t in chain_tokens):
                rendered = rendered + species_suffix
            exp_reactants[rendered] = float(stoich)

        exp_products: dict[str, float] = {}
        for name, stoich in products_template.items():
            rendered = _render_species_name(name, token_values)
            if species_suffix and any(f"{{{t}}}" in name for t in chain_tokens):
                rendered = rendered + species_suffix
            exp_products[rendered] = float(stoich)

        fwd_key = base_fwd_key.format_map(token_values)
        exp: dict[str, Any] = {
            "rxn_name": base_name.format_map(token_values),
            "reactants": exp_reactants,
            "products": exp_products,
            "rate_const_key": fwd_key,
            "rate_const_value": float(fwd_values[i]),
            "reversible": reversible,
        }

        if reversible:
            if not base_rev_key:
                raise KeyError(f"Compact reversible reaction '{base_name}' is missing rvs_rate_const_key.")
            exp["rvs_rate_const_key"] = base_rev_key.format_map(token_values)
            exp["rvs_rate_const_value"] = float(rev_values[i]) if rev_values is not None else 0.0

        if scaling_group is not None:
            exp["scaling_group"] = scaling_group.format_map(token_values)
        if rvs_scaling_group is not None:
            exp["rvs_scaling_group"] = rvs_scaling_group.format_map(token_values)

        expanded.append(exp)

    return expanded


def _parse_token_offsets(reaction: dict[str, Any]) -> dict[str, int]:
    """Parse chain-offset tokens from a reaction dict.

    Accepts ``token_offsets`` (explicit dict mapping token name to offset from chain)
    or ``carbon_offset`` (legacy scalar shorthand, mapped to ``{"carbon": n}``).
    Returns an empty dict when neither field is set.
    """
    if "token_offsets" in reaction:
        return {k: int(v) for k, v in reaction["token_offsets"].items()}
    if "carbon_offset" in reaction:
        return {"carbon": int(reaction["carbon_offset"])}
    return {}


def _as_list_and_shared(
    value: Any,
    expected_len: int,
    field_name: str,
    reaction: dict[str, Any],
    source_file: str,
    idx: int,
) -> tuple[list[float], bool]:
    """Return (values_list, is_shared). Scalar and length-1 list inputs are broadcast and marked shared."""
    if isinstance(value, list):
        if len(value) == 1:
            return [float(value[0])] * expected_len, True
        if len(value) != expected_len:
            raise ValueError(
                f"Compact reaction '{reaction.get('rxn_name', idx)}' in {source_file}: "
                f"{field_name} length {len(value)} does not match chain_lengths length {expected_len}."
            )
        return [float(v) for v in value], False
    # Scalar: broadcast and mark as shared.
    return [float(value)] * expected_len, True


def _render_species_name(template: str, values: dict[str, int]) -> str:
    return template.format_map(values)


def _validate_param_conflict(
    seen_param_values: dict[str, tuple[float, str, str]],
    param_key: str,
    param_value: float,
    source_file: str,
    reaction_name: str,
) -> None:
    """Reject conflicting values when the same kinetic key appears multiple times."""
    if param_key not in seen_param_values:
        seen_param_values[param_key] = (param_value, source_file, reaction_name)
        return

    previous_value, previous_file, previous_reaction = seen_param_values[param_key]
    if not np.isclose(previous_value, param_value):
        raise ValueError(
            "Conflicting nominal values for parameter "
            f"'{param_key}': {previous_value} ({previous_reaction} in {previous_file}) vs "
            f"{param_value} ({reaction_name} in {source_file})."
        )


def make_namespace(names: Sequence[str]) -> types.SimpleNamespace:
    """Return a SimpleNamespace mapping each name to its integer index.

    Enables tab-completion in Jupyter: ``sp.ACP``, ``sp.C3_MalACP``, ``pm.k3_1f``.
    Raises ValueError if any name is not a valid Python identifier.
    """
    invalid = [n for n in names if not n.isidentifier()]
    if invalid:
        raise ValueError(
            f"make_namespace: the following names are not valid Python identifiers "
            f"and would not be tab-completable: {invalid}"
        )
    return types.SimpleNamespace(**{name: i for i, name in enumerate(names)})


def make_reaction_namespace(rxns: Sequence["ElementaryReaction"]) -> types.SimpleNamespace:
    """Return a SimpleNamespace mapping each reaction name to its ElementaryReaction object.

    Enables tab-completion in Jupyter: ``rn.FabG_binding_NADPH``, ``print(rn.C4_FabB_binding_AcACP_sat)``.
    Raises ValueError if any reaction name is not a valid Python identifier.
    """
    names = [r.rxn_name for r in rxns]
    invalid = [n for n in names if not n.isidentifier()]
    if invalid:
        raise ValueError(
            f"make_reaction_namespace: the following reaction names are not valid Python identifiers "
            f"and would not be tab-completable: {invalid}"
        )
    return types.SimpleNamespace(**{r.rxn_name: r for r in rxns})


def set_scaling_group_values(
    theta: Any,
    params: list[str],
    scaling_group: dict[str, float],
) -> Any:
    """Return a new theta with scaling group parameters updated to the given values.

    Parameters
    ----------
    theta:
        Current parameter array returned by ``build_ode_system_from_reactions``.
    params:
        Parameter name list returned by ``build_ode_system_from_reactions``.
    scaling_group:
        Mapping of scaling group name → new value. Every key must exist in
        ``params``; a ``KeyError`` is raised otherwise to catch typos early.

    Examples
    --------
    >>> theta = set_scaling_group_values(theta, params, {"FabD_enzyme": 10.0, "FabH_enzyme": 2.0})
    """

    param_index = {name: i for i, name in enumerate(params)}
    for group in scaling_group:
        if group not in param_index:
            raise KeyError(
                f"set_scaling_group_values: '{group}' not found in params. "
                "Check that the scaling group name matches what is declared in the reaction files."
            )

    for group, value in scaling_group.items():
        theta = theta.at[param_index[group]].set(float(value))

    return theta


def get_ode_terms(
    name: "str | int",
    rxns: "Sequence[ElementaryReaction]",
    species: list[str],
) -> "Any":
    """Return grouped ODE-term DataFrames for a species.

    Contributions are split into two groups and displayed as separate tables:
    consumed and generated. Each row is one contribution (fwd or rev).
    In Jupyter both DataFrames render automatically; in plain Python they are
    printed as text.

    Parameters
    ----------
    name:
        Species index (``sp.ACP``) or species name string (``'ACP'``).
    rxns:
        List of ``ElementaryReaction`` objects from ``load_elementary_reactions``.
    species:
        Species name list from ``build_ode_system_from_reactions``.
    """
    sp_name = species[name] if isinstance(name, int) else name

    records: list[dict] = []

    for rxn in rxns:
        if sp_name in rxn.products:
            records.append({"species": sp_name, "role": "generated", "direction": "fwd",
                            "stoich": rxn.products[sp_name], "rxn_name": rxn.rxn_name,
                            "rate_expression": rxn.rate_expression})
        if sp_name in rxn.reactants:
            records.append({"species": sp_name, "role": "consumed", "direction": "fwd",
                            "stoich": rxn.reactants[sp_name], "rxn_name": rxn.rxn_name,
                            "rate_expression": rxn.rate_expression})
        if rxn.reversible and rxn.rvs_rate_expression is not None:
            if sp_name in rxn.reactants:
                records.append({"species": sp_name, "role": "generated", "direction": "rev",
                                "stoich": rxn.reactants[sp_name], "rxn_name": rxn.rxn_name,
                                "rate_expression": rxn.rvs_rate_expression})
            if sp_name in rxn.products:
                records.append({"species": sp_name, "role": "consumed", "direction": "rev",
                                "stoich": rxn.products[sp_name], "rxn_name": rxn.rxn_name,
                                "rate_expression": rxn.rvs_rate_expression})

    if not records:
        print(f"{sp_name}: not found in any reaction")
        return None

    consumed_records = [r for r in records if r["role"] == "consumed"]
    generated_records = [r for r in records if r["role"] == "generated"]

    print(f"{sp_name} | consumed")
    consumed_df = _display_table(consumed_records)
    print(f"{sp_name} | generated")
    generated_df = _display_table(generated_records)
    return {"consumed": consumed_df, "generated": generated_df}


# ---------------------------------------------------------------------------
# reaction_query dispatch tables
# ---------------------------------------------------------------------------

def _name_in_expr(name: str, expr: "str | None") -> bool:
    """True if *name* appears as a free-standing identifier in *expr*.

    Uses a word-boundary regex so that e.g. ``"b2"`` matches ``"(1/b2)"`` and
    ``"exp(b2*c1)"`` but does **not** match ``"b20"`` or ``"ab2"``.
    """
    if not expr:
        return False
    return bool(re.search(r"\b" + re.escape(name) + r"\b", expr))


# Exact-match predicates — used when query is an int (resolved to a name string).
# For "parameter" and "scaling", the name may appear inside a math expression
# (e.g. ``scaling_group = "(1/b2)"``), so we use word-boundary matching instead
# of equality, which would miss those cases.
_EXACT: "dict[str, Any]" = {
    "species":   lambda r, n: n in r.reactants or n in r.products,
    "reactant":  lambda r, n: n in r.reactants,
    "product":   lambda r, n: n in r.products,
    "parameter": lambda r, n: (
        n == r.rate_const_key
        or n == r.rvs_rate_const_key
        or _name_in_expr(n, r.scaling_group)
        or _name_in_expr(n, r.rvs_scaling_group)
    ),
    "scaling":   lambda r, n: (
        _name_in_expr(n, r.scaling_group)
        or _name_in_expr(n, r.rvs_scaling_group)
    ),
}

# Substring-match predicates — used when query is a str
_SUBSTR: "dict[str, Any]" = {
    "species":   lambda r, n: any(n in s for s in (*r.reactants, *r.products)),
    "reactant":  lambda r, n: any(n in s for s in r.reactants),
    "product":   lambda r, n: any(n in s for s in r.products),
    "parameter": lambda r, n: n in r.rate_const_key or n in (r.rvs_rate_const_key or ""),
    "scaling":   lambda r, n: n in (r.scaling_group or "") or n in (r.rvs_scaling_group or ""),
}

# Regex-match predicates — used when query is a str and use_regex=True
_REGEX: "dict[str, Any]" = {
    "species":   lambda r, p: any(p.search(s) for s in (*r.reactants, *r.products)),
    "reactant":  lambda r, p: any(p.search(s) for s in r.reactants),
    "product":   lambda r, p: any(p.search(s) for s in r.products),
    "parameter": lambda r, p: bool(p.search(r.rate_const_key)) or bool(p.search(r.rvs_rate_const_key or "")),
    "scaling":   lambda r, p: bool(p.search(r.scaling_group or "")) or bool(p.search(r.rvs_scaling_group or "")),
}

# Which context list to index for each query_type (int path only)
_INT_LIST_KEY: dict[str, str] = {
    "species":   "species",
    "reactant":  "species",
    "product":   "species",
    "parameter": "params",
    "scaling":   "scaling_params",
}


def _record_expression(r: "ElementaryReaction") -> dict:
    return {"rxn_name": r.rxn_name, "reaction": r.reaction_expression(), "rate_expression": r.full_rate_expression}


def _record_parameters(r: "ElementaryReaction") -> dict:
    return {
        "rxn_name":  r.rxn_name,
        "reaction":  r.reaction_expression(),
        "fwd_key":   r.rate_const_key,
        "fwd_value": r.rate_const_value,
        "rvs_key":   r.rvs_rate_const_key,
        "rvs_value": r.rvs_rate_const_value,
    }


def _record_scaling(r: "ElementaryReaction") -> dict:
    return {
        "rxn_name":          r.rxn_name,
        "reaction":          r.reaction_expression(),
        "scaling_group":     r.scaling_group,
        "rvs_scaling_group": r.rvs_scaling_group,
    }


# Record-builder functions indexed by return_type (all except "ode_terms" / "scaled_rate_constants")
_RECORD_FNS: "dict[str, Any]" = {
    "expression": _record_expression,
    "parameters": _record_parameters,
    "scaling":    _record_scaling,
}


def _display_table(records: "list[dict]") -> "Any":
    """Return a DataFrame and display it appropriately.

    In Jupyter each call to ``_display_table`` renders the DataFrame immediately
    via ``IPython.display.display``, so multiple queries in one cell all appear.
    In a plain Python script the DataFrame is printed as text.
    Returns the DataFrame when pandas is available, otherwise ``None``.
    """
    try:
        import pandas as pd
        df = pd.DataFrame(records).fillna("---")
        try:
            from IPython.display import display as _ipy_display, HTML as _HTML
            html = (
                '<div style="max-height:300px;overflow-y:auto;">'
                + df.to_html(index=False)
                + "</div>"
            )
            _ipy_display(_HTML(html))
        except ImportError:
            # Plain Python (no IPython kernel) — print explicitly.
            print(df.to_string(index=False))
        return df
    except ImportError:
        if not records:
            return None
        cols = list(records[0].keys())
        rows = [[str(r.get(c, "") if r.get(c) is not None else "") for c in cols] for r in records]
        widths = [max(len(c), *(len(row[i]) for row in rows)) for i, c in enumerate(cols)]
        sep = "  "
        print(sep.join(c.ljust(w) for c, w in zip(cols, widths)))
        print("-" * (sum(widths) + len(sep) * (len(cols) - 1)))
        for row in rows:
            print(sep.join(s.ljust(w) for s, w in zip(row, widths)))
        return None


def _resolve_criterion(
    query: "str | int",
    query_type: str,
    ctx: dict,
    use_regex: bool,
) -> tuple:
    """Return (name, predicate) for a single (query, query_type) criterion.

    predicate(r) -> bool tests one ElementaryReaction against this criterion.
    """
    if isinstance(query, (int, np.integer)):
        list_key = _INT_LIST_KEY.get(query_type)
        if list_key is None:
            raise ValueError(
                f"Integer queries are not supported for query_type={query_type!r}. "
                f"Supported: {sorted(_INT_LIST_KEY)}"
            )
        lst = ctx[list_key]
        if lst is None:
            raise ValueError(
                f"query_type={query_type!r} with an integer index has no associated list."
            )
        name = lst[int(query)]
        search_fn = _EXACT.get(query_type) or (lambda r, n: str(getattr(r, query_type, "")) == n)
        predicate = lambda r, fn=search_fn, n=name: fn(r, n)
    else:
        name = query
        if use_regex:
            pattern = re.compile(name)
            search_fn = _REGEX.get(query_type) or (lambda r, p: bool(p.search(str(getattr(r, query_type) or ""))))
            predicate = lambda r, fn=search_fn, p=pattern: fn(r, p)
        else:
            search_fn = _SUBSTR.get(query_type) or (lambda r, n: n in str(getattr(r, query_type) or ""))
            predicate = lambda r, fn=search_fn, n=name: fn(r, n)
    return name, predicate


def reaction_query(
    criteria: "list[tuple] | tuple",
    rxns: "Sequence[ElementaryReaction]",
    species: list[str],
    *,
    return_type: str = "expression",
    use_regex: bool = False,
    params: "list[str] | None" = None,
    scaling_params: "list[str] | None" = None,
    theta: "Any | None" = None,
) -> None:
    """Find and print reactions matching all supplied (query, query_type) criteria.

    Parameters
    ----------
    criteria:
        A list of ``(query, query_type)`` tuples.  All criteria are ANDed:
        only reactions matching every criterion are returned.
        Pass a single ``(query, query_type)`` tuple for a one-criterion search.
    rxns:
        Reactions from ``load_elementary_reactions``.
    species:
        Species name list from ``build_ode_system_from_reactions``.
    return_type:
        How to display each match: ``"expression"`` (default), ``"parameters"``,
        ``"scaling"``, ``"ode_terms"``, or ``"scaled_rate_constants"``.
    use_regex:
        When ``True`` and a query is a string, treat it as a regex.
        Integer index queries always use exact matching.
    params:
        Full parameter list — required for integer ``"parameter"`` queries.
    scaling_params:
        Scaling-parameter list — required for integer ``"scaling"`` queries.
    theta:
        Current parameter array (NumPy or JAX) — required for
        ``"scaled_rate_constants"``.
    """
    # Normalise: single (q, t) tuple → one-element list
    if (
        isinstance(criteria, tuple)
        and len(criteria) == 2
        and not isinstance(criteria[0], tuple)
    ):
        criteria = [criteria]
    criteria = list(criteria)

    ctx = {"species": species, "params": params, "scaling_params": scaling_params}

    # Resolve each criterion to (name, predicate)
    resolved: list[tuple] = []
    for query, query_type in criteria:
        name, predicate = _resolve_criterion(query, query_type, ctx, use_regex)
        resolved.append((query, query_type, name, predicate))

    # AND-filter: keep reactions satisfying every criterion
    matches = [r for r in rxns if all(pred(r) for _, _, _, pred in resolved)]

    criteria_str = ", ".join(f"{qt}={nm!r}" for _, qt, nm, _ in resolved)
    print(f"{len(matches)} match(es) | {criteria_str} | return={return_type}")

    if return_type == "ode_terms":
        sp_criteria = [(q, qt, nm) for q, qt, nm, _ in resolved if qt in ("species", "reactant", "product")]
        if not sp_criteria:
            raise ValueError(
                "return_type='ode_terms' requires at least one criterion with "
                "query_type in ('species', 'reactant', 'product')."
            )
        sp_names: list[str] = []
        for query, query_type, name in sp_criteria:
            if isinstance(query, (int, np.integer)):
                if name not in sp_names:
                    sp_names.append(name)
            elif use_regex:
                pattern = re.compile(name)
                for s in species:
                    if pattern.search(s) and s not in sp_names:
                        sp_names.append(s)
            else:
                for s in species:
                    if name in s and s not in sp_names:
                        sp_names.append(s)
        for sp_name in sp_names:
            get_ode_terms(sp_name, list(rxns), species)
        return _display_table([]) if not sp_names else None

    if return_type == "scaled_rate_constants":
        if theta is None or params is None:
            raise ValueError(
                "return_type='scaled_rate_constants' requires theta= to be provided. "
                "Call: rxns.query(..., return_type='scaled_rate_constants', theta=theta)"
            )

        # Build per-criterion include_fwd/include_rvs functions; a direction is
        # shown if ANY criterion (of parameter or scaling type) matches it.
        all_param_crits  = [(q, qt, nm) for q, qt, nm, _ in resolved if qt == "parameter"]
        all_scaling_crits = [(q, qt, nm) for q, qt, nm, _ in resolved if qt == "scaling"]

        fwd_checks: list = []
        rvs_checks: list = []

        for query, query_type, name in all_param_crits:
            if isinstance(query, (int, np.integer)):
                matched = {str(name)}
            elif use_regex and isinstance(query, str):
                qpat = re.compile(name)
                matched = {p for p in (params or []) if qpat.search(str(p))}
            else:
                matched = {p for p in (params or []) if str(name) in str(p)}

            if matched:
                fwd_checks.append(lambda r, m=matched: str(r.rate_const_key or "") in m)
                rvs_checks.append(lambda r, m=matched: str(r.rvs_rate_const_key or "") in m)
            elif use_regex and isinstance(query, str):
                qpat = re.compile(name)
                fwd_checks.append(lambda r, p=qpat: bool(p.search(str(r.rate_const_key or ""))))
                rvs_checks.append(lambda r, p=qpat: bool(p.search(str(r.rvs_rate_const_key or ""))))
            elif isinstance(query, (int, np.integer)):
                fwd_checks.append(lambda r, n=name: str(r.rate_const_key or "") == str(n))
                rvs_checks.append(lambda r, n=name: str(r.rvs_rate_const_key or "") == str(n))
            else:
                fwd_checks.append(lambda r, n=name: str(n) in str(r.rate_const_key or ""))
                rvs_checks.append(lambda r, n=name: str(n) in str(r.rvs_rate_const_key or ""))

        for query, query_type, name in all_scaling_crits:
            if isinstance(query, (int, np.integer)):
                matched_s = {str(name)}
            elif use_regex and isinstance(query, str):
                qpat = re.compile(name)
                matched_s = {g for g in (scaling_params or []) if qpat.search(str(g))}
            else:
                matched_s = {g for g in (scaling_params or []) if str(name) in str(g)}

            if matched_s:
                fwd_checks.append(lambda r, ms=matched_s: any(_name_in_expr(g, r.scaling_group) for g in ms))
                rvs_checks.append(lambda r, ms=matched_s: any(_name_in_expr(g, r.rvs_scaling_group) for g in ms))
            elif use_regex and isinstance(query, str):
                qpat = re.compile(name)
                fwd_checks.append(lambda r, p=qpat: bool(p.search(str(r.scaling_group or ""))))
                rvs_checks.append(lambda r, p=qpat: bool(p.search(str(r.rvs_scaling_group or ""))))
            else:
                fwd_checks.append(lambda r, n=name: _name_in_expr(str(n), r.scaling_group))
                rvs_checks.append(lambda r, n=name: _name_in_expr(str(n), r.rvs_scaling_group))

        if fwd_checks:
            # Direction shown if ANY param/scaling criterion matches that direction
            include_fwd = lambda r, checks=fwd_checks: any(fn(r) for fn in checks)
            include_rvs = lambda r, checks=rvs_checks: any(fn(r) for fn in checks)
        else:
            include_fwd = lambda r: True
            include_rvs = lambda r: True

        records: list[dict] = []
        for r in matches:
            if include_fwd(r):
                fwd_scale = _eval_scale_expr(r.scaling_group, params, theta) if r.scaling_group is not None else None
                records.append({
                    "rxn_name":   r.rxn_name,
                    "reaction":   r.reaction_expression(),
                    "direction":  "fwd",
                    "key":        r.rate_const_key,
                    "base":       r.rate_const_value,
                    "scale_expr": r.scaling_group,
                    "scale":      fwd_scale,
                    "effective":  r.rate_const_value * fwd_scale if fwd_scale is not None else r.rate_const_value,
                })
            if include_rvs(r) and r.rvs_rate_const_key is not None and r.rvs_rate_const_value is not None:
                rvs_scale = _eval_scale_expr(r.rvs_scaling_group, params, theta) if r.rvs_scaling_group is not None else None
                records.append({
                    "rxn_name":   r.rxn_name,
                    "reaction":   r.reaction_expression(),
                    "direction":  "rvs",
                    "key":        r.rvs_rate_const_key,
                    "base":       r.rvs_rate_const_value,
                    "scale_expr": r.rvs_scaling_group,
                    "scale":      rvs_scale,
                    "effective":  r.rvs_rate_const_value * rvs_scale if rvs_scale is not None else r.rvs_rate_const_value,
                })
        return _display_table(records)

    if return_type not in _RECORD_FNS:
        raise ValueError(
            f"Unknown return_type={return_type!r}. "
            f"Supported: {sorted(_RECORD_FNS)}, 'ode_terms', 'scaled_rate_constants'"
        )
    record_fn = _RECORD_FNS[return_type]
    return _display_table([record_fn(r) for r in matches])
