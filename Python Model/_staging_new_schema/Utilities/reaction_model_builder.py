"""Compact-aware reaction network model builder.
Reads reaction definitions from YAML or JSON files.

Reactions vary over a small set of named "axes" (exactly one numeric axis,
e.g. fatty-acid chain length, plus any number of categorical axes, e.g.
saturation state), declared via ``axis_definitions`` and a per-reaction
``axes`` field -- nothing about axis names, values, or count is hardcoded
here. Species names and rate-constant keys use the identical ``{axis_name}``
placeholder syntax (e.g. ``{chain}``, ``{chain+2}``, ``{saturation}``); a rate
constant's key/value pair is not separately declared as "shared" or
"independent" -- it's inferred purely from the *shape* of its value (a plain
number is shared across every instance, a dict is independent across
whichever axis its keys match), and any independent axis not already placed
explicitly in the key template is appended automatically.
"""

from __future__ import annotations

import itertools
import re
import types
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
import sympy
import yaml
import json as _json

import jax.numpy as jnp
import jax
import equinox as eqx
jax.config.update("jax_enable_x64", True)


GLOSSARY_FILENAME = "species_components.yaml"


# ─────────────────────────────────────────────────────────────────────────────
# Shared math-expression helpers (used for scaling_group expressions and,
# via _render/_render_formula_value, for every other templated field)
# ─────────────────────────────────────────────────────────────────────────────

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


def _render(template: str, token_values: dict[str, Any]) -> str:
    """Render a template string as an f-string, with each axis token bound as a
    real local variable -- ``{chain}``, ``{chain+2}``, ``{saturation}`` all just
    evaluate as native Python f-string expressions (arithmetic for numeric axis
    tokens, plain substitution for categorical axis tokens, which are strings).
    Text outside ``{}`` (e.g. free kinetic-parameter symbols in a scaling_group
    expression like ``b3``) is untouched, exactly as with any f-string.

    Templates come only from this repo's own YAML files, not external input, so
    ``eval`` here is a deliberate simplification (Python's own expression
    grammar, no hand-rolled parser) rather than a general-purpose sandbox;
    ``__builtins__`` is stripped to keep the evaluated expression to axis-token
    arithmetic only.
    """
    try:
        return eval(f'f"""{template}"""', {"__builtins__": {}}, dict(token_values))  # noqa: S307
    except Exception as exc:
        raise ValueError(f"Could not render template {template!r} with tokens {token_values}: {exc}") from exc


def _render_formula_value(template: str, token_values: dict[str, Any]) -> float:
    """Render a glossary ``formula`` entry (a number, or a token/arithmetic string) to a float."""
    substituted = _render(template, token_values)
    try:
        return float(eval(substituted, {"__builtins__": {}}, {}))  # noqa: S307
    except Exception as exc:
        raise ValueError(f"Could not evaluate formula {template!r} (rendered {substituted!r}): {exc}") from exc


# ─────────────────────────────────────────────────────────────────────────────
# Axis model
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class NumericAxis:
    """A numeric compaction axis (e.g. chain length). Exactly one is allowed per
    axis set -- it is the only axis kind that supports inline arithmetic tokens
    like ``{chain+2}`` and derived-token math."""
    name: str
    values: list[int]


@dataclass(frozen=True)
class CategoricalValue:
    """One named value of a categorical axis (e.g. "sat"/"unsat")."""
    token: str
    restrict: "dict[str, list[int]] | None" = None


@dataclass(frozen=True)
class CategoricalAxis:
    """A categorical compaction axis (e.g. saturation state). Any number of
    these may appear in an axis set; nothing about their names or values is
    hardcoded."""
    name: str
    values: "dict[str, CategoricalValue]"


Axis = "NumericAxis | CategoricalAxis"


@dataclass(frozen=True)
class AxisPoint:
    """One concrete combination of axis values a templated reaction/species expands to."""
    numeric_values: dict[str, int]
    categorical_values: dict[str, str]
    categorical_tokens: dict[str, str]

    def render_tokens(self) -> dict[str, Any]:
        """Token dict for ``_render``: numeric axes -> int, categorical axes -> token string."""
        out: dict[str, Any] = dict(self.numeric_values)
        out.update(self.categorical_tokens)
        return out

    def raw_values(self) -> dict[str, Any]:
        """Token dict for provenance/display: numeric axes -> int, categorical axes -> value name."""
        out: dict[str, Any] = dict(self.numeric_values)
        out.update(self.categorical_values)
        return out


class AxisSet:
    """The full collection of named axes declared in a file's ``axis_definitions``."""

    def __init__(self, axes: "dict[str, Axis]") -> None:
        self.axes = axes
        numeric = [n for n, a in axes.items() if isinstance(a, NumericAxis)]
        if len(numeric) > 1:
            raise ValueError(f"Only one numeric axis is supported per file; got {numeric}")

    @classmethod
    def from_dict(cls, spec: "dict[str, Any]") -> "AxisSet":
        axes: "dict[str, Axis]" = {}
        for name, ax_spec in spec.items():
            kind = ax_spec.get("type")
            if kind == "numeric":
                axes[name] = NumericAxis(name=name, values=[int(v) for v in ax_spec["values"]])
            elif kind == "categorical":
                values: dict[str, CategoricalValue] = {}
                for val_name, val_spec in ax_spec["values"].items():
                    raw_restrict = val_spec.get("restrict")
                    restrict = (
                        {k: [int(x) for x in v] for k, v in raw_restrict.items()}
                        if raw_restrict else None
                    )
                    values[val_name] = CategoricalValue(token=str(val_spec.get("token", "")), restrict=restrict)
                axes[name] = CategoricalAxis(name=name, values=values)
            else:
                raise ValueError(f"axis {name!r} has unknown type {kind!r}; expected 'numeric' or 'categorical'")
        return cls(axes)

    def combinations(
        self,
        axis_names: "list[str]",
        overrides: "dict[str, dict[str, dict[str, list[int]]]] | None" = None,
    ) -> "list[AxisPoint]":
        """Enumerate every valid AxisPoint for the given axes.

        For each combination of categorical-axis values, the active numeric
        range is the intersection of every included categorical value's
        ``restrict`` for the numeric axis (``axis_overrides`` takes priority
        over a categorical value's own default ``restrict``); with no
        categorical axes involved, or none of them restricting the numeric
        axis, the full numeric axis ``values`` list applies.
        """
        overrides = overrides or {}
        for name in axis_names:
            if name not in self.axes:
                raise KeyError(f"Unknown axis {name!r}; declared axes are {sorted(self.axes)}")

        numeric_axes = [a for a in axis_names if isinstance(self.axes[a], NumericAxis)]
        categorical_axes = [a for a in axis_names if isinstance(self.axes[a], CategoricalAxis)]
        numeric_axis_name = numeric_axes[0] if numeric_axes else None

        cat_value_lists = [list(self.axes[a].values.items()) for a in categorical_axes]
        combos = list(itertools.product(*cat_value_lists)) if cat_value_lists else [()]

        points: list[AxisPoint] = []
        for combo in combos:
            categorical_values = {axis: val_name for axis, (val_name, _) in zip(categorical_axes, combo)}
            categorical_tokens = {axis: cv.token for axis, (_, cv) in zip(categorical_axes, combo)}

            if numeric_axis_name is None:
                points.append(AxisPoint({}, categorical_values, categorical_tokens))
                continue

            active_range: "set[int] | None" = None
            for axis, (val_name, cv) in zip(categorical_axes, combo):
                r = None
                if axis in overrides and val_name in overrides[axis]:
                    r = overrides[axis][val_name].get(numeric_axis_name)
                elif cv.restrict is not None:
                    r = cv.restrict.get(numeric_axis_name)
                if r is not None:
                    r_set = set(r)
                    active_range = r_set if active_range is None else (active_range & r_set)
            if active_range is None:
                active_range = set(self.axes[numeric_axis_name].values)

            for v in sorted(active_range):
                points.append(AxisPoint({numeric_axis_name: v}, dict(categorical_values), dict(categorical_tokens)))
        return points


# ─────────────────────────────────────────────────────────────────────────────
# Parameter linkage
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class LinkageSpec:
    """Per-axis record of whether a rate constant is shared ("linked") or
    independently-fittable ("independent") across that axis's values.

    Never declared in YAML -- inferred purely from the *shape* of the
    rate_const_value/rvs_rate_const_value that was actually given (see
    ``_infer_linkage``): a plain number means every axis is linked, a dict
    means whichever axis its keys correspond to is independent. There is
    exactly one source of truth (the value), so there's nothing to fall out
    of sync with.
    """
    per_axis: "dict[str, str]"

    def independent_axes(self, axes: "list[str]") -> "list[str]":
        return [a for a in axes if self.per_axis.get(a) == "independent"]

    def linked_axes(self, axes: "list[str]") -> "list[str]":
        return [a for a in axes if self.per_axis.get(a) == "linked"]

    def derive_key(self, base_name: str, point: AxisPoint, axes: "list[str]", axis_set: AxisSet) -> str:
        """Render base_name (which may itself contain explicit {axis} placeholders,
        same syntax as species names), then append any independent axis's value
        that the author did *not* already place explicitly -- same convention as
        before, just skipped wherever the template already covers it."""
        rendered = _render(base_name, point.render_tokens())
        parts = [rendered]
        for a in axes:
            if self.per_axis.get(a) != "independent":
                continue
            if _axis_referenced(base_name, a):
                continue
            if isinstance(axis_set.axes[a], NumericAxis):
                parts.append(f"C{point.numeric_values[a]}")
            else:
                parts.append(str(point.categorical_values[a]))
        return "_".join(parts)


def _axis_referenced(template: str, axis_name: str) -> bool:
    """True if the raw (unrendered) template string already places this axis's
    token explicitly, e.g. via ``{chain}`` or ``{chain+2}``."""
    return f"{{{axis_name}" in template


def _infer_linkage(raw_value: Any, axes: "list[str]", axis_set: AxisSet) -> LinkageSpec:
    """Infer which axes a rate constant is independent vs. linked across from
    the shape of its value alone: a plain number is fully linked/shared; a
    dict is independent across whichever axis its keys match (categorical
    axes are matched by value-name, the numeric axis by membership in its
    declared values), recursing through nested dicts (categorical levels
    first, numeric innermost -- the same convention used when writing them).
    Any axis in ``axes`` not implicated by the value's shape is linked.
    """
    categorical_axes = [a for a in axes if isinstance(axis_set.axes[a], CategoricalAxis)]
    numeric_axes = [a for a in axes if isinstance(axis_set.axes[a], NumericAxis)]
    numeric_axis_name = numeric_axes[0] if numeric_axes else None

    independent: set[str] = set()
    value = raw_value
    remaining_categorical = list(categorical_axes)
    while isinstance(value, dict):
        outer_keys = set(value.keys())
        matched = next(
            (a for a in remaining_categorical if outer_keys and outer_keys <= set(axis_set.axes[a].values.keys())),
            None,
        )
        if matched is not None:
            independent.add(matched)
            remaining_categorical.remove(matched)
            value = next(iter(value.values()))
            continue
        if (
            numeric_axis_name is not None
            and numeric_axis_name not in independent
            and outer_keys
            and outer_keys <= set(axis_set.axes[numeric_axis_name].values)
        ):
            independent.add(numeric_axis_name)
            value = next(iter(value.values()))
            continue
        raise ValueError(
            f"Could not infer which axis a rate value's dict keys {sorted(outer_keys, key=str)} "
            f"correspond to among {axes}."
        )

    return LinkageSpec({a: ("independent" if a in independent else "linked") for a in axes})


def _broaden_with_key_references(value_linkage: LinkageSpec, base_key_template: str, axes: "list[str]") -> LinkageSpec:
    """Union value-shape-inferred independence with any axis the author placed
    explicitly in the key template.

    Writing e.g. ``rate_const_key: "kcat_B_BKeAcACP_C{chain}"`` with a plain
    scalar ``rate_const_value`` is a deliberate, supported pattern: distinct,
    individually-addressable parameter keys per chain value (so each can later
    be fit independently) that for now all share one starting value. The value
    lookup itself (``_lookup_rate_value``) still uses the narrower,
    value-shape-only linkage -- broadening only affects which axes get their
    own key string and how ``param_linkage`` reports the parameter.
    """
    per_axis = dict(value_linkage.per_axis)
    for a in axes:
        if per_axis.get(a) != "independent" and _axis_referenced(base_key_template, a):
            per_axis[a] = "independent"
    return LinkageSpec(per_axis)


def _independent_categorical_numeric(
    linkage: LinkageSpec, axes: "list[str]", axis_set: AxisSet
) -> "tuple[list[str], list[str]]":
    indep = linkage.independent_axes(axes)
    categorical = [a for a in indep if isinstance(axis_set.axes[a], CategoricalAxis)]
    numeric = [a for a in indep if isinstance(axis_set.axes[a], NumericAxis)]
    return categorical, numeric


def _lookup_rate_value(
    raw_value: Any, linkage: LinkageSpec, axes: "list[str]", axis_set: AxisSet, point: AxisPoint
) -> float:
    if not isinstance(raw_value, dict):
        return float(raw_value)
    indep_categorical, indep_numeric = _independent_categorical_numeric(linkage, axes, axis_set)
    cur = raw_value
    for a in indep_categorical:
        cur = cur[point.categorical_values[a]]
    for a in indep_numeric:
        cur = cur[point.numeric_values[a]]
    return float(cur)


def _validate_value_shape(
    raw_value: Any,
    linkage: LinkageSpec,
    axes: "list[str]",
    points: "list[AxisPoint]",
    axis_set: AxisSet,
    field_label: str,
    rxn_label: str,
) -> None:
    """Raise a clear error if raw_value's dict nesting doesn't exactly match the
    independent axes implied by ``linkage`` and the enumerated ``points``."""
    indep_categorical, indep_numeric = _independent_categorical_numeric(linkage, axes, axis_set)
    indep_order = indep_categorical + indep_numeric

    if not indep_order:
        if isinstance(raw_value, dict):
            raise ValueError(f"{rxn_label}: {field_label} is fully linked, expected a scalar, got a dict.")
        return

    def _path_for(p: AxisPoint) -> tuple:
        path = []
        for a in indep_order:
            path.append(p.categorical_values[a] if a in p.categorical_values else p.numeric_values[a])
        return tuple(path)

    expected_paths = {_path_for(p) for p in points}

    def _walk(value: Any, depth: int, prefix: list) -> None:
        if depth == len(indep_order):
            if isinstance(value, dict):
                raise ValueError(f"{rxn_label}: {field_label}{prefix} should be a number, got a dict.")
            return
        if not isinstance(value, dict):
            raise ValueError(
                f"{rxn_label}: {field_label}{prefix} should be a dict keyed by {indep_order[depth]!r} values, "
                f"got {type(value).__name__}."
            )
        expected_here = {p[depth] for p in expected_paths if p[:depth] == tuple(prefix)}
        actual = set(value.keys())
        if actual != expected_here:
            raise ValueError(
                f"{rxn_label}: {field_label}{prefix} keys mismatch at axis {indep_order[depth]!r} -- "
                f"missing {expected_here - actual}, unexpected {actual - expected_here}."
            )
        for k, sub in value.items():
            _walk(sub, depth + 1, prefix + [k])

    _walk(raw_value, 0, [])


# ─────────────────────────────────────────────────────────────────────────────
# Compact reaction expansion
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ParamLinkage:
    """Which axes a parameter key is shared vs. independent across, and which
    expanded reaction instances use it -- the queryable answer to "is this
    rate constant currently fit jointly or individually"."""
    key: str
    base_name: str
    role: str
    linked_axes: "list[str]"
    independent_axes: "list[str]"
    instances: "list[dict[str, Any]]" = field(default_factory=list)


@dataclass
class ExpansionResult:
    reactions: "list[dict[str, Any]]"
    species_axis_provenance: "dict[str, dict[str, Any]]"
    param_linkage: "dict[str, ParamLinkage]"


_TEMPLATE_STRIP_RE = re.compile(r"\{[^{}]*\}")


def _register_species_provenance(
    provenance: "dict[str, dict[str, Any]]", rendered: str, template: str, point: AxisPoint
) -> None:
    provenance[rendered] = {
        "template": _TEMPLATE_STRIP_RE.sub("", template),
        "axis_values": point.raw_values(),
    }


def _register_linkage(
    registry: "dict[str, ParamLinkage]",
    key: str,
    base_name: str,
    role: str,
    linkage: LinkageSpec,
    axes: "list[str]",
    point: AxisPoint,
) -> None:
    entry = registry.setdefault(
        key,
        ParamLinkage(
            key=key,
            base_name=base_name,
            role=role,
            linked_axes=linkage.linked_axes(axes),
            independent_axes=linkage.independent_axes(axes),
        ),
    )
    entry.instances.append(point.raw_values())


class CompactReaction:
    """One ``axes:``-bearing reaction template, ready to expand across an AxisSet."""

    def __init__(self, raw: "dict[str, Any]", source_file: str, idx: int) -> None:
        self.raw = raw
        self.source_file = source_file
        self.idx = idx
        self.axes: list[str] = list(raw["axes"])
        self.base_name = str(raw.get("rxn_name", f"reaction_{idx}"))

    def expand(self, axis_set: AxisSet) -> ExpansionResult:
        raw = self.raw
        label = f"'{self.base_name}' in {self.source_file}"

        reactants_template = raw.get("reactants")
        products_template = raw.get("products")
        if not isinstance(reactants_template, dict) or not isinstance(products_template, dict):
            raise TypeError(f"Reaction {label} must have dict-shaped reactants/products templates.")

        overrides = raw.get("axis_overrides")
        points = axis_set.combinations(self.axes, overrides=overrides)
        if not points:
            raise ValueError(f"Reaction {label}: no valid axis combinations were enumerated.")

        reversible = bool(raw.get("reversible", False))

        fwd_base_key = str(raw["rate_const_key"])
        fwd_value_raw = raw["rate_const_value"]
        fwd_value_linkage = _infer_linkage(fwd_value_raw, self.axes, axis_set)
        _validate_value_shape(fwd_value_raw, fwd_value_linkage, self.axes, points, axis_set, "rate_const_value", label)
        fwd_key_linkage = _broaden_with_key_references(fwd_value_linkage, fwd_base_key, self.axes)

        rvs_key_linkage = rvs_value_linkage = rvs_base_key = rvs_value_raw = None
        if reversible:
            rvs_base_key = str(raw["rvs_rate_const_key"])
            rvs_value_raw = raw["rvs_rate_const_value"]
            rvs_value_linkage = _infer_linkage(rvs_value_raw, self.axes, axis_set)
            _validate_value_shape(rvs_value_raw, rvs_value_linkage, self.axes, points, axis_set, "rvs_rate_const_value", label)
            rvs_key_linkage = _broaden_with_key_references(rvs_value_linkage, rvs_base_key, self.axes)

        scaling_group_template = raw.get("scaling_group")
        rvs_scaling_group_template = raw.get("rvs_scaling_group")

        expanded: list[dict[str, Any]] = []
        species_axis_provenance: dict[str, dict[str, Any]] = {}
        param_linkage: dict[str, ParamLinkage] = {}

        for point in points:
            token_values = point.render_tokens()

            exp_reactants: dict[str, float] = {}
            for name, stoich in reactants_template.items():
                rendered = _render(name, token_values)
                exp_reactants[rendered] = float(stoich)
                _register_species_provenance(species_axis_provenance, rendered, name, point)

            exp_products: dict[str, float] = {}
            for name, stoich in products_template.items():
                rendered = _render(name, token_values)
                exp_products[rendered] = float(stoich)
                _register_species_provenance(species_axis_provenance, rendered, name, point)

            fwd_key = fwd_key_linkage.derive_key(fwd_base_key, point, self.axes, axis_set)
            fwd_value = _lookup_rate_value(fwd_value_raw, fwd_value_linkage, self.axes, axis_set, point)

            entry: dict[str, Any] = {
                "rxn_name": _render(self.base_name, token_values),
                "reactants": exp_reactants,
                "products": exp_products,
                "rate_const_key": fwd_key,
                "rate_const_value": fwd_value,
                "reversible": reversible,
            }
            if scaling_group_template is not None:
                entry["scaling_group"] = _render(scaling_group_template, token_values)

            _register_linkage(param_linkage, fwd_key, fwd_base_key, "fwd", fwd_key_linkage, self.axes, point)

            if reversible:
                rvs_key = rvs_key_linkage.derive_key(rvs_base_key, point, self.axes, axis_set)
                rvs_value = _lookup_rate_value(rvs_value_raw, rvs_value_linkage, self.axes, axis_set, point)
                entry["rvs_rate_const_key"] = rvs_key
                entry["rvs_rate_const_value"] = rvs_value
                if rvs_scaling_group_template is not None:
                    entry["rvs_scaling_group"] = _render(rvs_scaling_group_template, token_values)
                _register_linkage(param_linkage, rvs_key, rvs_base_key, "rvs", rvs_key_linkage, self.axes, point)

            expanded.append(entry)

        return ExpansionResult(expanded, species_axis_provenance, param_linkage)


# ─────────────────────────────────────────────────────────────────────────────
# ElementaryReaction / ReactionList
# ─────────────────────────────────────────────────────────────────────────────

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

    Returned by ``load_elementary_reactions``. Behaves exactly like a plain
    ``list[ElementaryReaction]`` everywhere a list is accepted. ``species_components``,
    ``species_axis_values``, and ``param_linkage`` are populated by the loader
    (not computed here) since only the loader knows what produced the reactions.
    """

    def __init__(self, rxns: list["ElementaryReaction"], species: list[str]) -> None:
        super().__init__(rxns)
        self._species = species

        self.species_components: dict[str, dict[str, float]] = {}
        self.species_axis_values: dict[str, dict[str, Any]] = {}
        self.param_linkage: dict[str, ParamLinkage] = {}

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

        Rarely needed -- parameter and scaling-parameter lists are computed
        automatically from the loaded reactions. Use this only when the external
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
            Alternating ``(query, query_type)`` values. Must contain an even
            number of elements (>= 2). Each ``query`` is a substring
            (``str``) or an index into the relevant list (``int``, e.g.
            ``sp.ACP``, ``pm.kcat_H``). Each ``query_type`` is one of the
            built-in aliases ``"species"``, ``"reactant"``, ``"product"``,
            ``"parameter"``, ``"scaling"`` or any ``ElementaryReaction``
            attribute name.
        return_type:
            What to display for each match. One of ``"expression"``
            (default), ``"parameters"``, ``"scaling"``, ``"linkage"``,
            ``"ode_terms"``, or ``"scaled_rate_constants"``.
        theta:
            Current parameter array (NumPy or JAX) -- required for
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
            param_linkage=self.param_linkage,
        )
        try:
            get_ipython  # type: ignore[name-defined]  # noqa: F821
            return None  # display() already fired inside _display_table
        except NameError:
            return _df


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

_AXIS_TEMPLATE_FIELDS: frozenset[str] = _SCALAR_FIELDS | frozenset({
    "axes", "axis_overrides",
})


def _validate_known_fields(reaction: dict[str, Any], source_file: str, idx: int) -> None:
    """Raise ValueError if a reaction dict contains any unrecognised field name."""
    known = _AXIS_TEMPLATE_FIELDS if "axes" in reaction else _SCALAR_FIELDS
    unknown = set(reaction.keys()) - known
    if unknown:
        name = reaction.get("rxn_name", f"index {idx}")
        raise ValueError(
            f"Reaction '{name}' in {source_file} contains unknown field(s): "
            + ", ".join(sorted(unknown))
            + ". Check for typos or use the correct field set."
        )


# ─────────────────────────────────────────────────────────────────────────────
# Loading
# ─────────────────────────────────────────────────────────────────────────────

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
    species_axis_provenance: dict[str, dict[str, Any]] = {}
    param_linkage: dict[str, ParamLinkage] = {}
    raw = _load_reactions(
        reactions_path, schemas=schemas,
        species_axis_provenance=species_axis_provenance,
        param_linkage=param_linkage,
    )
    all_species: list[str] = []
    seen: set[str] = set()
    for rxn in raw:
        for sp in list(rxn["reactants"]) + list(rxn["products"]):
            if sp not in seen:
                seen.add(sp)
                all_species.append(sp)
    rxns = ReactionList([ElementaryReaction.from_mapping(rxn) for rxn in raw], all_species)
    rxns.species_components = load_species_components(reactions_path, schemas=schemas)
    rxns.species_axis_values = species_axis_provenance
    rxns.param_linkage = param_linkage
    return rxns


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


def _parse_file(file_path: Path) -> dict[str, Any]:
    with open(file_path, "r", encoding="utf-8") as fh:
        if file_path.suffix.lower() == ".json":
            return _json.load(fh)
        return yaml.safe_load(fh)


def _collect_file_specs(reactions_path: "str | Path | Sequence[str | Path]") -> "tuple[list[tuple[str, dict[str, Any]]], str]":
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
            reaction_files = [
                p for p in sorted(source_path.glob("*.yaml")) + sorted(source_path.glob("*.json"))
                if p.name != GLOSSARY_FILENAME
            ]
            if not reaction_files:
                raise ValueError(f"No YAML or JSON reaction files found in directory: {source_path}")
            for file_path in sorted(reaction_files, key=lambda p: p.name):
                file_specs.append((str(file_path), _parse_file(file_path)))
        else:
            file_specs.append((str(source_path), _parse_file(source_path)))
        source_label = str(source_path)
    return file_specs, source_label


def _load_reactions(
    reactions_path: "str | Path | Sequence[str | Path]",
    schemas: frozenset[str] | set[str] | None = None,
    species_axis_provenance: "dict[str, dict[str, Any]] | None" = None,
    param_linkage: "dict[str, ParamLinkage] | None" = None,
) -> list[dict[str, Any]]:
    """Load reactions from a YAML/JSON file, directory, or explicit list of files.

    ``reactions_path`` may be a single file, a directory (all ``*.yaml``/``*.json``
    files in it are loaded, except ``species_components.yaml``), or a list/tuple
    of individual file paths -- the latter lets you compose a reaction system
    from files spread across a directory (e.g. specific enzymes from a larger
    reaction library) without copying them into a new folder.
    """
    file_specs, source_label = _collect_file_specs(reactions_path)

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

        axis_set: "AxisSet | None" = None
        if "axis_definitions" in spec:
            axis_set = AxisSet.from_dict(spec["axis_definitions"])

        if "reactions" not in spec or not isinstance(spec["reactions"], list):
            raise KeyError(f"Missing or invalid 'reactions' list in {source_file}")

        file_reactions: list[dict[str, Any]] = []

        for idx, reaction in enumerate(spec["reactions"]):
            if not isinstance(reaction, dict):
                raise TypeError(f"Reaction index {idx} in {source_file} must be a dict.")

            _validate_known_fields(reaction, source_file, idx)

            if "axes" in reaction:
                if axis_set is None:
                    raise KeyError(
                        f"Reaction '{reaction.get('rxn_name', idx)}' in {source_file} uses 'axes' "
                        "but the file has no 'axis_definitions' block."
                    )
                result = CompactReaction(reaction, source_file, idx).expand(axis_set)
                expanded_entries = result.reactions
                if species_axis_provenance is not None:
                    species_axis_provenance.update(result.species_axis_provenance)
                if param_linkage is not None:
                    for key, link in result.param_linkage.items():
                        if key in param_linkage:
                            param_linkage[key].instances.extend(link.instances)
                        else:
                            param_linkage[key] = link
            else:
                expanded_entries = [reaction]

            for entry in expanded_entries:
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


# ─────────────────────────────────────────────────────────────────────────────
# Species-components glossary
# ─────────────────────────────────────────────────────────────────────────────

def _discover_glossary_dirs(reactions_path: "str | Path | Sequence[str | Path]") -> "list[Path]":
    dirs: set[Path] = set()
    if isinstance(reactions_path, (list, tuple)):
        for entry in reactions_path:
            p = Path(entry).expanduser().resolve()
            dirs.add(p.parent if p.is_file() else p)
    else:
        p = Path(reactions_path).expanduser().resolve()
        dirs.add(p.parent if p.is_file() else p)
    return sorted(dirs)


def load_species_components(
    reactions_path: "str | Path | Sequence[str | Path]",
    schemas: frozenset[str] | set[str] | None = None,
) -> "dict[str, dict[str, float]]":
    """Discover and resolve ``species_components.yaml`` file(s) alongside ``reactions_path``.

    The glossary is a directory-wide superset: it is fine (and expected) for it
    to declare species that don't appear in whatever subset of reaction files is
    actually being loaded, and fine for it to not (yet) cover every species a
    given subset uses -- ``reaction_sanity_check.check_balances`` reports
    uncovered reactions separately rather than erroring. Returns ``{}`` if no
    glossary file is found.
    """
    components: dict[str, dict[str, float]] = {}
    for directory in _discover_glossary_dirs(reactions_path):
        gpath = directory / GLOSSARY_FILENAME
        if not gpath.exists():
            continue
        with open(gpath, "r", encoding="utf-8") as fh:
            spec = yaml.safe_load(fh)

        axis_set = AxisSet.from_dict(spec.get("axis_definitions", {})) if "axis_definitions" in spec else None

        for entry in spec.get("components", []):
            pattern = str(entry["pattern"])
            formula = entry.get("formula", {})

            if "axes" in entry:
                if axis_set is None:
                    raise KeyError(f"Component {pattern!r} in {gpath} uses 'axes' but no 'axis_definitions' is declared.")
                axes = list(entry["axes"])
                overrides = entry.get("axis_overrides")
                for point in axis_set.combinations(axes, overrides=overrides):
                    token_values = point.render_tokens()
                    name = _render(pattern, token_values)
                    components[name] = {
                        group: (
                            _render_formula_value(val, token_values) if isinstance(val, str) else float(val)
                        )
                        for group, val in formula.items()
                    }
            else:
                components[pattern] = {group: float(val) for group, val in formula.items()}

    return components


# ─────────────────────────────────────────────────────────────────────────────
# Namespaces / scaling-group helpers
# ─────────────────────────────────────────────────────────────────────────────

def make_namespace(names: Sequence[str]) -> types.SimpleNamespace:
    """Return a SimpleNamespace mapping each name to its integer index.

    Enables tab-completion in Jupyter: ``sp.ACP``, ``pm.k3_1f``.
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
        Mapping of scaling group name -> new value. Every key must exist in
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

    return _display_table(records)


# ─────────────────────────────────────────────────────────────────────────────
# Query system
# ─────────────────────────────────────────────────────────────────────────────

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


def _parameter_candidates(r: "ElementaryReaction") -> "list[str]":
    names = [k for k in (r.rate_const_key, r.rvs_rate_const_key) if k]
    for expr in (r.scaling_group, r.rvs_scaling_group):
        if expr:
            names.extend(_extract_scale_param_names(expr))
    return names


def _scaling_candidates(r: "ElementaryReaction") -> "list[str]":
    names: list[str] = []
    for expr in (r.scaling_group, r.rvs_scaling_group):
        if expr:
            names.extend(_extract_scale_param_names(expr))
    return names


# What strings a query_type refers to on a given reaction.
_QUERY_FIELDS: "dict[str, Callable[[ElementaryReaction], list[str]]]" = {
    "species": lambda r: list(r.reactants) + list(r.products),
    "reactant": lambda r: list(r.reactants),
    "product": lambda r: list(r.products),
    "parameter": _parameter_candidates,
    "scaling": _scaling_candidates,
}

_INT_LIST_KEY: dict[str, str] = {
    "species": "species",
    "reactant": "species",
    "product": "species",
    "parameter": "params",
    "scaling": "scaling_params",
}


def _matches(candidates: "list[str]", query: Any, mode: str) -> bool:
    if mode == "exact":
        return query in candidates
    if mode == "substr":
        return any(query in c for c in candidates)
    if mode == "regex":
        return any(query.search(c) for c in candidates)
    raise ValueError(f"Unknown match mode {mode!r}")


def _field_candidates(r: "ElementaryReaction", query_type: str) -> "list[str]":
    fn = _QUERY_FIELDS.get(query_type)
    if fn is not None:
        return fn(r)
    return [str(getattr(r, query_type, ""))]


def _resolve_criterion(
    query: "str | int",
    query_type: str,
    ctx: dict,
    use_regex: bool,
) -> "Callable[[ElementaryReaction], bool]":
    if isinstance(query, (int, np.integer)):
        list_key = _INT_LIST_KEY.get(query_type)
        if list_key is None:
            raise ValueError(
                f"Integer queries are not supported for query_type={query_type!r}. "
                f"Supported: {sorted(_INT_LIST_KEY)}"
            )
        lst = ctx[list_key]
        if lst is None:
            raise ValueError(f"query_type={query_type!r} with an integer index has no associated list.")
        name = lst[int(query)]
        return lambda r, n=name, qt=query_type: _matches(_field_candidates(r, qt), n, "exact")

    mode = "regex" if use_regex else "substr"
    q = re.compile(query) if use_regex else query
    return lambda r, q=q, qt=query_type, m=mode: _matches(_field_candidates(r, qt), q, m)


def _record_expression(r: "ElementaryReaction", ctx: dict) -> "list[dict]":
    return [{"rxn_name": r.rxn_name, "reaction": r.reaction_expression(), "rate_expression": r.full_rate_expression}]


def _record_parameters(r: "ElementaryReaction", ctx: dict) -> "list[dict]":
    return [{
        "rxn_name": r.rxn_name,
        "reaction": r.reaction_expression(),
        "fwd_key": r.rate_const_key,
        "fwd_value": r.rate_const_value,
        "rvs_key": r.rvs_rate_const_key,
        "rvs_value": r.rvs_rate_const_value,
    }]


def _record_scaling(r: "ElementaryReaction", ctx: dict) -> "list[dict]":
    return [{
        "rxn_name": r.rxn_name,
        "reaction": r.reaction_expression(),
        "scaling_group": r.scaling_group,
        "rvs_scaling_group": r.rvs_scaling_group,
    }]


def _record_linkage(r: "ElementaryReaction", ctx: dict) -> "list[dict]":
    param_linkage: dict[str, ParamLinkage] = ctx.get("param_linkage") or {}
    rows = []
    for key, role in ((r.rate_const_key, "fwd"), (r.rvs_rate_const_key, "rvs")):
        if not key:
            continue
        info = param_linkage.get(key)
        if info is None:
            continue
        rows.append({
            "rxn_name": r.rxn_name,
            "parameter": key,
            "role": role,
            "linked_axes": ", ".join(info.linked_axes) or "(none)",
            "independent_axes": ", ".join(info.independent_axes) or "(none)",
            "n_instances": len(info.instances),
        })
    return rows


def _record_ode_terms(r: "ElementaryReaction", ctx: dict) -> "list[dict]":
    rows = []
    for sp_name in set(r.reactants) | set(r.products):
        if sp_name in r.products:
            rows.append({"species": sp_name, "role": "generated", "direction": "fwd",
                         "rxn_name": r.rxn_name, "rate_expression": r.rate_expression})
        if sp_name in r.reactants:
            rows.append({"species": sp_name, "role": "consumed", "direction": "fwd",
                         "rxn_name": r.rxn_name, "rate_expression": r.rate_expression})
        if r.reversible and r.rvs_rate_expression is not None:
            if sp_name in r.reactants:
                rows.append({"species": sp_name, "role": "generated", "direction": "rev",
                             "rxn_name": r.rxn_name, "rate_expression": r.rvs_rate_expression})
            if sp_name in r.products:
                rows.append({"species": sp_name, "role": "consumed", "direction": "rev",
                             "rxn_name": r.rxn_name, "rate_expression": r.rvs_rate_expression})
    return rows


def _record_scaled_rate_constants(r: "ElementaryReaction", ctx: dict) -> "list[dict]":
    theta = ctx.get("theta")
    params = ctx.get("params")
    if theta is None or params is None:
        raise ValueError("return_type='scaled_rate_constants' requires theta=... to be passed to query().")
    fwd_scale = _eval_scale_expr(r.scaling_group, params, theta) if r.scaling_group else 1.0
    row = {
        "rxn_name": r.rxn_name,
        "reaction": r.reaction_expression(),
        "fwd_rate_const": r.rate_const_value * fwd_scale,
    }
    if r.reversible and r.rvs_rate_const_value is not None:
        rvs_scale = _eval_scale_expr(r.rvs_scaling_group, params, theta) if r.rvs_scaling_group else 1.0
        row["rvs_rate_const"] = r.rvs_rate_const_value * rvs_scale
    return [row]


_ROW_BUILDERS: "dict[str, Callable[[ElementaryReaction, dict], list[dict]]]" = {
    "expression": _record_expression,
    "parameters": _record_parameters,
    "scaling": _record_scaling,
    "linkage": _record_linkage,
    "ode_terms": _record_ode_terms,
    "scaled_rate_constants": _record_scaled_rate_constants,
}


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
    param_linkage: "dict[str, ParamLinkage] | None" = None,
) -> "Any":
    """Find and print reactions matching all supplied (query, query_type) criteria."""
    ctx = {"species": species, "params": params, "scaling_params": scaling_params,
           "theta": theta, "param_linkage": param_linkage}

    predicates = [_resolve_criterion(q, qt, ctx, use_regex) for q, qt in criteria]
    matches = [r for r in rxns if all(p(r) for p in predicates)]

    builder = _ROW_BUILDERS.get(return_type)
    if builder is None:
        raise ValueError(f"Unknown return_type {return_type!r}. Supported: {sorted(_ROW_BUILDERS)}")

    rows: list[dict] = []
    for r in matches:
        rows.extend(builder(r, ctx))

    label = ", ".join(f"{qt}={q!r}" for q, qt in criteria)
    print(f"{len(matches)} match(es) | {label} | return={return_type}")
    return _display_table(rows)
