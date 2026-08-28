from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import arviz as az
import numpy as np
import pandas as pd

from reaction_model_builder import (
    ElementaryReaction,
    _eval_scale_expr,
    _extract_scale_param_names,
    build_ode_system_from_reactions,
    load_elementary_reactions,
)


@dataclass(frozen=True)
class EffectiveParameterLimitResult:
    feasibility_mask: np.ndarray
    draw_summary: pd.DataFrame
    channel_summary: pd.DataFrame
    effective_values: pd.DataFrame
    rules: list[dict[str, Any]]
    free_params: list[str]
    param_names: list[str]


def evaluate_effective_parameter_limits(
    inf_data: az.InferenceData,
    reactions_source: str,
    free_params: Sequence[str],
    effective_parameter_limits: Sequence[Mapping[str, Any]],
    max_effective_value_records: int = 200_000,
) -> EffectiveParameterLimitResult:
    """Evaluate post-scaling effective-parameter limits against posterior draws.

    Limits are diagnostic: they are recomputed from posterior samples and reaction
    definitions, so changing the rules does not require rerunning inference.
    Each rule is matched against reaction channels and then applied to the final
    effective value: base rate constant * scaling expression(theta).
    """
    if not effective_parameter_limits:
        raise ValueError("At least one effective parameter limit rule is required.")

    rxns = load_elementary_reactions(reactions_source)
    _, _, param_names, param_values, scaling_params = build_ode_system_from_reactions(
        reactions_source
    )

    selected_free_params = [name for name in free_params if name in inf_data.posterior.data_vars]
    if not selected_free_params:
        raise ValueError("None of the provided free_params are present in posterior variables.")

    posterior_values = {
        name: np.asarray(inf_data.posterior[name].values, dtype=np.float64).reshape(-1)
        for name in selected_free_params
    }
    n_draws = len(next(iter(posterior_values.values())))
    for name, values in posterior_values.items():
        if len(values) != n_draws:
            raise ValueError(
                f"Posterior variable '{name}' has {len(values)} flattened draws; expected {n_draws}."
            )

    matched_channels = _build_limited_channels(
        rxns=rxns,
        params=param_names,
        scaling_params=scaling_params,
        rules=[dict(rule) for rule in effective_parameter_limits],
    )
    if not matched_channels:
        empty_mask = np.ones((n_draws,), dtype=bool)
        return EffectiveParameterLimitResult(
            feasibility_mask=empty_mask,
            draw_summary=pd.DataFrame(
                {
                    "draw_index": np.arange(n_draws, dtype=int),
                    "is_feasible": empty_mask,
                    "n_violations": np.zeros((n_draws,), dtype=int),
                    "max_violation_ratio": np.zeros((n_draws,), dtype=float),
                    "worst_channel": [None] * n_draws,
                    "worst_rule": [None] * n_draws,
                }
            ),
            channel_summary=pd.DataFrame(),
            effective_values=pd.DataFrame(),
            rules=[dict(rule) for rule in effective_parameter_limits],
            free_params=selected_free_params,
            param_names=param_names,
        )

    theta_template = np.asarray(
        [float(param_values[name]) for name in param_names], dtype=np.float64
    )
    param_index = {name: i for i, name in enumerate(param_names)}

    n_channels = len(matched_channels)
    effective_matrix = np.zeros((n_draws, n_channels), dtype=np.float64)
    lower_matrix = np.full((n_draws, n_channels), -np.inf, dtype=np.float64)
    upper_matrix = np.full((n_draws, n_channels), np.inf, dtype=np.float64)

    for draw_index in range(n_draws):
        theta = theta_template.copy()
        for name, values in posterior_values.items():
            if name in param_index:
                theta[param_index[name]] = float(values[draw_index])

        for channel_index, channel in enumerate(matched_channels):
            scale_expr = channel["scale_expr"]
            scale = _eval_scale_expr(scale_expr, param_names, theta) if scale_expr else 1.0
            base_value = float(theta[param_index[channel["param_name"]]])
            effective_matrix[draw_index, channel_index] = base_value * scale
            lower_matrix[draw_index, channel_index] = channel["lower"]
            upper_matrix[draw_index, channel_index] = channel["upper"]

    below_lower = effective_matrix < lower_matrix
    above_upper = effective_matrix > upper_matrix
    violated = below_lower | above_upper
    feasibility_mask = ~np.any(violated, axis=1)

    upper_ratio = np.divide(
        effective_matrix,
        upper_matrix,
        out=np.zeros_like(effective_matrix),
        where=np.isfinite(upper_matrix) & (upper_matrix != 0),
    )
    lower_ratio = np.divide(
        lower_matrix,
        effective_matrix,
        out=np.zeros_like(effective_matrix),
        where=np.isfinite(lower_matrix) & (effective_matrix != 0),
    )
    violation_ratio = np.maximum(
        np.where(above_upper, upper_ratio, 0.0),
        np.where(below_lower, lower_ratio, 0.0),
    )

    worst_channel_indices = np.argmax(violation_ratio, axis=1)
    max_violation_ratio = np.max(violation_ratio, axis=1)
    n_violations = np.sum(violated, axis=1)

    draw_summary = pd.DataFrame(
        {
            "draw_index": np.arange(n_draws, dtype=int),
            "is_feasible": feasibility_mask,
            "n_violations": n_violations,
            "max_violation_ratio": max_violation_ratio,
            "worst_channel": [
                matched_channels[idx]["channel_id"] if n_violations[row] else None
                for row, idx in enumerate(worst_channel_indices)
            ],
            "worst_rule": [
                matched_channels[idx]["rule_name"] if n_violations[row] else None
                for row, idx in enumerate(worst_channel_indices)
            ],
        }
    )

    channel_rows = []
    for channel_index, channel in enumerate(matched_channels):
        values = effective_matrix[:, channel_index]
        channel_violated = violated[:, channel_index]
        channel_rows.append(
            {
                **_channel_metadata(channel),
                "n_draws": n_draws,
                "n_violations": int(np.sum(channel_violated)),
                "violation_fraction": float(np.mean(channel_violated)),
                "min_effective": float(np.nanmin(values)),
                "median_effective": float(np.nanmedian(values)),
                "max_effective": float(np.nanmax(values)),
            }
        )
    channel_summary = pd.DataFrame(channel_rows)

    effective_values = _build_effective_values_frame(
        effective_matrix=effective_matrix,
        violated=violated,
        matched_channels=matched_channels,
        max_records=max_effective_value_records,
    )

    return EffectiveParameterLimitResult(
        feasibility_mask=feasibility_mask,
        draw_summary=draw_summary,
        channel_summary=channel_summary,
        effective_values=effective_values,
        rules=[dict(rule) for rule in effective_parameter_limits],
        free_params=selected_free_params,
        param_names=param_names,
    )


def _build_limited_channels(
    rxns: Sequence[ElementaryReaction],
    params: Sequence[str],
    scaling_params: Sequence[str],
    rules: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    channels = []
    for reaction in rxns:
        channels.append(
            {
                "rxn_name": reaction.rxn_name,
                "direction": "fwd",
                "param_name": reaction.rate_const_key,
                "base_value": float(reaction.rate_const_value),
                "scale_expr": reaction.scaling_group,
                "scale_param_names": _extract_scale_param_names(reaction.scaling_group) if reaction.scaling_group else [],
                "reactants": list(reaction.reactants.keys()),
                "products": list(reaction.products.keys()),
            }
        )
        if reaction.reversible and reaction.rvs_rate_const_key is not None:
            channels.append(
                {
                    "rxn_name": reaction.rxn_name,
                    "direction": "rvs",
                    "param_name": reaction.rvs_rate_const_key,
                    "base_value": float(reaction.rvs_rate_const_value),
                    "scale_expr": reaction.rvs_scaling_group,
                    "scale_param_names": _extract_scale_param_names(reaction.rvs_scaling_group) if reaction.rvs_scaling_group else [],
                    "reactants": list(reaction.products.keys()),
                    "products": list(reaction.reactants.keys()),
                }
            )

    matched_channels: list[dict[str, Any]] = []
    for channel in channels:
        for rule_index, rule in enumerate(rules):
            if not _rule_matches_channel(rule, channel):
                continue
            lower = float(rule.get("lower", -np.inf))
            upper = float(rule.get("upper", np.inf))
            rule_name = str(rule.get("name", f"rule_{rule_index}"))
            scale_expr = channel.get("scale_expr")
            matched_channels.append(
                {
                    **channel,
                    "channel_id": f"{channel['rxn_name']}::{channel['direction']}::{channel['param_name']}::{rule_name}",
                    "rule_index": rule_index,
                    "rule_name": rule_name,
                    "lower": lower,
                    "upper": upper,
                    "scale_param_names": channel.get("scale_param_names", []),
                }
            )
    return matched_channels


def _rule_matches_channel(rule: Mapping[str, Any], channel: Mapping[str, Any]) -> bool:
    direction = rule.get("direction")
    if direction not in (None, "both") and str(direction) != str(channel["direction"]):
        return False

    param_name = str(channel["param_name"])
    scale_param_names = set(channel.get("scale_param_names", []))
    param_names = rule.get("param_names")
    if param_names is not None and param_name not in set(param_names) and not (scale_param_names & set(param_names)):
        return False

    param_name_pattern = rule.get("param_name_pattern")
    if param_name_pattern and not _matches_text(param_name, str(param_name_pattern), rule):
        return False

    scaling_param_names = rule.get("scaling_param_names")
    if scaling_param_names is not None and not (scale_param_names & set(scaling_param_names)):
        return False

    scaling_param_name_pattern = rule.get("scaling_param_name_pattern")
    if scaling_param_name_pattern and not any(
        _matches_text(name, str(scaling_param_name_pattern), rule)
        for name in scale_param_names
    ):
        return False

    reaction_name_pattern = rule.get("reaction_name_pattern")
    if reaction_name_pattern and not _matches_text(str(channel["rxn_name"]), str(reaction_name_pattern), rule):
        return False

    reactant_patterns = rule.get("reactant_name_patterns", [])
    if reactant_patterns:
        match_count = _count_reactant_pattern_matches(channel["reactants"], reactant_patterns, rule)
        expected_count = rule.get("reactant_match_count")
        min_count = rule.get("min_reactant_match_count")
        max_count = rule.get("max_reactant_match_count")
        if expected_count is not None and match_count != int(expected_count):
            return False
        if min_count is not None and match_count < int(min_count):
            return False
        if max_count is not None and match_count > int(max_count):
            return False
        if expected_count is None and min_count is None and max_count is None and match_count == 0:
            return False

    return True


def _matches_text(value: str, pattern: str, rule: Mapping[str, Any]) -> bool:
    if rule.get("use_regex", True):
        return bool(re.search(pattern, value))
    return pattern in value


def _count_reactant_pattern_matches(
    reactants: Sequence[str],
    patterns: Sequence[str],
    rule: Mapping[str, Any],
) -> int:
    count = 0
    for reactant in reactants:
        if any(_matches_text(str(reactant), str(pattern), rule) for pattern in patterns):
            count += 1
    return count


def _channel_metadata(channel: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "rule_name": channel["rule_name"],
        "rxn_name": channel["rxn_name"],
        "direction": channel["direction"],
        "param_name": channel["param_name"],
        "reactants": " + ".join(channel["reactants"]),
        "scale_expr": channel.get("scale_expr"),
        "scale_param_names": ", ".join(channel.get("scale_param_names", [])),
        "lower": channel["lower"],
        "upper": channel["upper"],
    }


def _build_effective_values_frame(
    effective_matrix: np.ndarray,
    violated: np.ndarray,
    matched_channels: Sequence[Mapping[str, Any]],
    max_records: int,
) -> pd.DataFrame:
    n_draws, n_channels = effective_matrix.shape
    total_records = n_draws * n_channels
    if total_records > max_records:
        draw_indices = np.linspace(0, n_draws - 1, max_records // n_channels, dtype=int)
    else:
        draw_indices = np.arange(n_draws, dtype=int)

    records = []
    for draw_index in draw_indices:
        for channel_index, channel in enumerate(matched_channels):
            records.append(
                {
                    "draw_index": int(draw_index),
                    **_channel_metadata(channel),
                    "effective_value": float(effective_matrix[draw_index, channel_index]),
                    "violated": bool(violated[draw_index, channel_index]),
                }
            )
    return pd.DataFrame(records)
