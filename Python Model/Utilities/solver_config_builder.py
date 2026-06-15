from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping


def validate_required_fields(
    item: Mapping[str, Any],
    required_fields: list[str],
    section_name: str,
    item_index: int,
) -> None:
    """Raise a clear error if any required fields are missing from an item."""
    missing_fields = [field for field in required_fields if field not in item]
    if missing_fields:
        raise ValueError(
            f"{section_name}[{item_index}] is missing required fields: {missing_fields}"
        )


def build_free_kinetic_parameter_specs(
    free_parameter_prior_inputs: list[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Convert user-facing free parameter inputs into solver_params free_kinetic_params schema."""
    free_kinetic_params = []
    for index, parameter_input in enumerate(free_parameter_prior_inputs):
        validate_required_fields(
            item=parameter_input,
            required_fields=["param_name", "distribution", "lower", "upper"],
            section_name="free_parameter_prior_inputs",
            item_index=index,
        )

        free_kinetic_params.append(
            {
                "rxn_name": parameter_input.get("rxn_name"),
                "param_name": parameter_input["param_name"],
                "prior_dist_params": {
                    "distribution": parameter_input["distribution"],
                    "lower": float(parameter_input["lower"]),
                    "upper": float(parameter_input["upper"]),
                    "mass": float(parameter_input.get("mass", 0.95)),
                    "fixed_stat": parameter_input.get("fixed_stat", None),
                },
            }
        )

    return free_kinetic_params


def build_dataset_config_entries(
    dataset_inputs: list[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Convert user-facing dataset inputs into solver_params datasets schema."""
    dataset_entries = []
    for index, dataset_input in enumerate(dataset_inputs):
        validate_required_fields(
            item=dataset_input,
            required_fields=["name", "dataset_type", "data_file", "observables"],
            section_name="dataset_inputs",
            item_index=index,
        )

        observables_map = dataset_input["observables"]
        if not isinstance(observables_map, dict):
            raise TypeError(
                f"dataset_inputs[{index}] 'observables' must be a dictionary mapping calculated simulation outputs to CSV columns; got {type(observables_map)}."
            )
        
        dataset_entry = {
            "name": dataset_input["name"],
            "dataset_type": dataset_input["dataset_type"],
            "data_file": dataset_input["data_file"],
            "observables": dict(observables_map),
            "noise_model": dataset_input.get("noise_model", ),
            "noise_params": dataset_input.get("noise_params", {"frac": 0.05}),
        }

        if "enabled" in dataset_input:
            dataset_entry["enabled"] = dataset_input["enabled"]
        if "time_column" in dataset_input:
            dataset_entry["time_column"] = dataset_input["time_column"]
        if "time_values" in dataset_input:
            dataset_entry["time_values"] = dataset_input["time_values"]
        if "init_cond_columns" in dataset_input:
            dataset_entry["init_cond_columns"] = dataset_input["init_cond_columns"]
        if "init_cond_overrides" in dataset_input:
            dataset_entry["init_cond_overrides"] = dataset_input["init_cond_overrides"]

        dataset_entries.append(dataset_entry)

    return dataset_entries


def build_solver_params_config(
    free_parameter_prior_inputs: list[Mapping[str, Any]],
    dataset_inputs: list[Mapping[str, Any]],
    prior_sampling_settings: Mapping[str, Any],
    posterior_sampling_settings: Mapping[str, Any],
    ode_solver_settings: Mapping[str, Any],
    ode_stepsize_controller_settings: Mapping[str, Any],
    calculation_module_path: str,
    path_base: str | Path | None = None,
    output_paths: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the full solver_params.json payload from guided notebook inputs."""
    solver_params_config = {
        "free_kinetic_params": build_free_kinetic_parameter_specs(free_parameter_prior_inputs),
        "prior_sampling": dict(prior_sampling_settings),
        "posterior_sampling": dict(posterior_sampling_settings),
        "ODE_solver": dict(ode_solver_settings),
        "ODE_stepsize_controller": dict(ode_stepsize_controller_settings),
        "calculation_module": calculation_module_path,
        "datasets": build_dataset_config_entries(dataset_inputs),
    }

    if path_base is not None:
        solver_params_config["path_base"] = str(path_base)

    if output_paths is not None:
        solver_params_config["output_paths"] = dict(output_paths)

    return solver_params_config


def write_solver_params_json_file(
    solver_params_config: Mapping[str, Any],
    file_directory: str | Path,
    filename: str | Path,
) -> Path:
    """Write a solver params config dictionary to JSON and return resolved path."""
    resolved_output_path = (Path(file_directory).expanduser() / Path(filename)).resolve()
    resolved_output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(resolved_output_path, "w", encoding="utf-8") as output_file:
        json.dump(solver_params_config, output_file, indent=4)

    return resolved_output_path
