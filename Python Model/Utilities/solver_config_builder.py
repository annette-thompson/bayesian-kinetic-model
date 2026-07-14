from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence


def _resolve_path(base_dir: str | Path, path_value: "str | Path | Sequence[str | Path]") -> "Path | list[Path]":
    if isinstance(path_value, (list, tuple)):
        return [_resolve_path(base_dir, entry) for entry in path_value]  # type: ignore[return-value]
    path = Path(path_value).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (Path(base_dir).expanduser() / path).resolve()


def discover_observable_names(
    calculation_module_path: str | Path,
    reactions_source: "str | Path | Sequence[str | Path]",
    path_base: str | Path = ".",
) -> list[str]:
    """Return observable names exposed by a calculation module for a reaction network."""
    base_dir = Path(path_base).expanduser().resolve()
    utilities_dir = str(base_dir / "Utilities")
    if utilities_dir not in sys.path:
        sys.path.insert(0, utilities_dir)

    from experiment_framework import load_observable_definitions
    from reaction_model_builder import build_ode_system_from_reactions

    module_path = _resolve_path(base_dir, calculation_module_path)
    reactions_path = _resolve_path(base_dir, reactions_source)
    _, species_names, _, _, _ = build_ode_system_from_reactions(reactions_path)
    return list(load_observable_definitions(module_path, species_names=species_names).keys())


def build_observables_mapping(
    *,
    output_names: Sequence[str] | None = None,
    output_pattern: str | None = None,
    available_outputs: Sequence[str] | None = None,
    csv_columns: Mapping[str, str] | Sequence[str] | None = None,
) -> dict[str, str]:
    """Build calculated-output -> CSV-column mapping for a dataset."""
    if output_pattern is not None:
        if available_outputs is None:
            raise ValueError("output_pattern requires available_outputs.")
        pattern = re.compile(output_pattern)
        selected_outputs = [name for name in available_outputs if pattern.fullmatch(name)]
    elif output_names is not None:
        selected_outputs = list(output_names)
    elif csv_columns is not None:
        selected_outputs = list(csv_columns.keys() if isinstance(csv_columns, Mapping) else csv_columns)
    else:
        raise ValueError("Provide output_names, output_pattern, or csv_columns.")

    if not selected_outputs:
        raise ValueError("No dataset outputs were selected.")

    if available_outputs is not None:
        available_set = set(available_outputs)
        missing_outputs = [name for name in selected_outputs if name not in available_set]
        if missing_outputs:
            raise ValueError(f"Selected outputs are not available from the calculation module: {missing_outputs}")

    if csv_columns is None:
        return {name: name for name in selected_outputs}

    if isinstance(csv_columns, Mapping):
        missing_columns = [name for name in selected_outputs if name not in csv_columns]
        if missing_columns:
            raise ValueError(f"Missing CSV column mapping for outputs: {missing_columns}")
        return {name: csv_columns[name] for name in selected_outputs}

    if len(csv_columns) != len(selected_outputs):
        raise ValueError("csv_columns sequence must have the same length as selected outputs.")
    return dict(zip(selected_outputs, csv_columns))


def timeseries_dataset(
    *,
    name: str,
    data_file: str | Path,
    time_column: str,
    observables: Mapping[str, str] | None = None,
    output_names: Sequence[str] | None = None,
    output_pattern: str | None = None,
    available_outputs: Sequence[str] | None = None,
    csv_columns: Mapping[str, str] | Sequence[str] | None = None,
    init_cond_overrides: Mapping[str, Any] | None = None,
    init_cond_columns: Mapping[str, str] | None = None,
    noise_model: str | None = "relative_mean",
    noise_params: Mapping[str, Any] | None = None,
    enabled: bool = True,
) -> dict[str, Any]:
    dataset = {
        "name": name,
        "dataset_type": "timeseries",
        "data_file": str(data_file),
        "observables": dict(observables) if observables is not None else build_observables_mapping(
            output_names=output_names,
            output_pattern=output_pattern,
            available_outputs=available_outputs,
            csv_columns=csv_columns,
        ),
        "time_column": time_column,
        "noise_model": noise_model,
        "noise_params": dict(noise_params or {"frac": 0.05}),
        "enabled": enabled,
    }
    if init_cond_overrides is not None:
        dataset["init_cond_overrides"] = dict(init_cond_overrides)
    if init_cond_columns is not None:
        dataset["init_cond_columns"] = dict(init_cond_columns)
    return dataset


def endpoint_dataset(
    *,
    name: str,
    data_file: str | Path,
    time_values: Sequence[float] | float,
    observables: Mapping[str, str] | None = None,
    output_names: Sequence[str] | None = None,
    output_pattern: str | None = None,
    available_outputs: Sequence[str] | None = None,
    csv_columns: Mapping[str, str] | Sequence[str] | None = None,
    init_cond_columns: Mapping[str, str] | None = None,
    init_cond_overrides: Mapping[str, Any] | None = None,
    noise_model: str | None = "relative_mean",
    noise_params: Mapping[str, Any] | None = None,
    enabled: bool = True,
) -> dict[str, Any]:
    if isinstance(time_values, (int, float)):
        time_values = [float(time_values)]

    dataset = {
        "name": name,
        "dataset_type": "endpoint",
        "data_file": str(data_file),
        "observables": dict(observables) if observables is not None else build_observables_mapping(
            output_names=output_names,
            output_pattern=output_pattern,
            available_outputs=available_outputs,
            csv_columns=csv_columns,
        ),
        "time_values": list(time_values),
        "noise_model": noise_model,
        "noise_params": dict(noise_params or {"frac": 0.05}),
        "enabled": enabled,
    }
    if init_cond_columns is not None:
        dataset["init_cond_columns"] = dict(init_cond_columns)
    if init_cond_overrides is not None:
        dataset["init_cond_overrides"] = dict(init_cond_overrides)
    return dataset


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
