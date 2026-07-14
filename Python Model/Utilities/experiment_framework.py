from __future__ import annotations

from dataclasses import dataclass
from importlib.util import module_from_spec, spec_from_file_location
from inspect import signature
from pathlib import Path
from typing import Any, Callable, Mapping

import jax.numpy as jnp
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ObservableDefinition:
    name: str
    compute: Callable[..., Any]
    output_names: tuple[str, ...]
    required_species: tuple[str, ...] = ()
    description: str = ""

    def evaluate(
        self,
        times: jnp.ndarray,
        concentrations: jnp.ndarray,
        species_index: Mapping[str, int],
        dataset_config: Mapping[str, Any],
        experiment_config: Mapping[str, Any],
    ) -> dict[str, jnp.ndarray]:
        result = _invoke_user_function(
            self.compute,
            times=times,
            concentrations=concentrations,
            species_index=species_index,
            dataset_config=dataset_config,
            experiment_config=experiment_config,
        )

        if isinstance(result, Mapping):
            normalized = {name: jnp.asarray(result[name]) for name in self.output_names}
        else:
            if len(self.output_names) != 1:
                raise ValueError(
                    f"Observable '{self.name}' must return a mapping with outputs {self.output_names}."
                )
            normalized = {self.output_names[0]: jnp.asarray(result)}

        missing = [name for name in self.output_names if name not in normalized]
        if missing:
            raise ValueError(
                f"Observable '{self.name}' did not return required outputs: {missing}."
            )

        for output_name, values in normalized.items():
            if values.ndim != 1:
                raise ValueError(
                    f"Observable '{self.name}' output '{output_name}' must be a 1D array; got shape {values.shape}."
                )
            if values.shape[0] != times.shape[0]:
                raise ValueError(
                    f"Observable '{self.name}' output '{output_name}' must have length {times.shape[0]}; got {values.shape[0]}."
                )

        return normalized


@dataclass(frozen=True)
class NoiseModelDefinition:
    name: str
    compute: Callable[..., Any]
    description: str = ""

    def evaluate(
        self,
        values: np.ndarray,
        frame: pd.DataFrame,
        output_name: str,
        dataset_config: Mapping[str, Any],
        experiment_config: Mapping[str, Any],
    ) -> np.ndarray:
        result = _invoke_user_function(
            self.compute,
            values=values,
            frame=frame,
            output_name=output_name,
            dataset_config=dataset_config,
            experiment_config=experiment_config,
            noise_params=dataset_config.get("noise_params", {}),
        )
        sigma = np.asarray(result, dtype=np.float64)
        if sigma.ndim == 0:
            sigma = np.full(values.shape, float(sigma), dtype=np.float64)
        if sigma.shape != values.shape:
            raise ValueError(
                f"Noise model '{self.name}' for output '{output_name}' returned shape {sigma.shape}; expected {values.shape}."
            )
        if np.any(sigma <= 0):
            raise ValueError(
                f"Noise model '{self.name}' produced non-positive sigma values for output '{output_name}'."
            )
        return sigma


@dataclass(frozen=True)
class DatasetBundle:
    name: str
    dataset_type: str
    observable_names: tuple[str, ...]
    data_file: str
    time_column: str | None
    time_values: np.ndarray
    time_indices_np: np.ndarray
    time_indices_jax: jnp.ndarray
    condition_indices_np: np.ndarray
    condition_indices_jax: jnp.ndarray
    init_cond_columns: tuple[tuple[str, str], ...]
    observables_mapping: tuple[tuple[str, str], ...]
    noise_model: str | None
    noise_params: Mapping[str, Any]
    observed_values: np.ndarray
    observed_sigma: np.ndarray
    frame: pd.DataFrame


@dataclass(frozen=True)
class ExperimentBundle:
    calculation_module_path: str
    calculated_observables: dict[str, ObservableDefinition]
    datasets: tuple[DatasetBundle, ...]
    condition_matrix_np: np.ndarray
    condition_matrix_jax: jnp.ndarray
    simulation_times_np: np.ndarray
    simulation_times_jax: jnp.ndarray
    observed_values: np.ndarray
    observed_sigma: np.ndarray
    observed_labels: tuple[str, ...]
    dataset_configs_by_name: Mapping[str, Mapping[str, Any]]
    raw_config: Mapping[str, Any]


def load_experiment_bundle(
    solver_params: Mapping[str, Any],
    solver_params_file: str,
    species_names: list[str],
) -> ExperimentBundle:
    validate_experiment_config(
        solver_params=solver_params,
        solver_params_file=solver_params_file,
        species_names=species_names,
    )

    base_dir = _get_path_base_dir(solver_params, solver_params_file)
    calculation_module = solver_params.get("calculation_module")
    module_path = _resolve_path(base_dir, calculation_module)
    calculated_observables = load_observable_definitions(module_path, species_names=species_names)
    noise_models = load_noise_model_definitions()

    raw_dataset_configs = solver_params.get("datasets")
    dataset_configs = [
        _normalize_dataset_config(config, idx)
        for idx, config in enumerate(raw_dataset_configs)
    ]
    dataset_configs = [config for config in dataset_configs if config.get("enabled", True)]

    if not dataset_configs:
        raise ValueError("No enabled datasets found. Set at least one dataset with enabled=true.")

    dataset_configs_by_name = {
        config["name"]: config for config in dataset_configs
    }

    species_index = {name: idx for idx, name in enumerate(species_names)}
    y0_template = np.zeros((len(species_names),), dtype=np.float64)

    condition_lookup: dict[tuple[float, ...], int] = {}
    condition_matrix_rows: list[np.ndarray] = []

    def register_condition(y0_values: np.ndarray) -> int:
        key = tuple(float(value) for value in y0_values)
        if key not in condition_lookup:
            condition_lookup[key] = len(condition_matrix_rows)
            condition_matrix_rows.append(np.asarray(y0_values, dtype=np.float64))
        return condition_lookup[key]

    simulation_times_np = _build_simulation_times(dataset_configs, base_dir)
    simulation_times_jax = jnp.asarray(simulation_times_np)

    datasets: list[DatasetBundle] = []
    observed_values_parts: list[np.ndarray] = []
    observed_sigma_parts: list[np.ndarray] = []
    observed_labels: list[str] = []

    dummy_concentrations = jnp.ones(
        (simulation_times_jax.shape[0], len(species_names)), dtype=jnp.float64
    )

    for dataset_config in dataset_configs:
        observables_dict = dataset_config.get("observables", {})
        observable_names = tuple(observables_dict.keys())

        observables_for_dataset = []
        preview_outputs = {}
        for observable_name in observable_names:
            if observable_name not in calculated_observables:
                raise KeyError(
                    f"Dataset '{dataset_config.get('name', '<unnamed>')}' references unknown observable '{observable_name}'."
                )

            observable = calculated_observables[observable_name]
            missing_species = [
                species for species in observable.required_species if species not in species_index
            ]
            if missing_species:
                raise ValueError(
                    f"Observable '{observable_name}' requires species {missing_species}, but only {species_names} are present."
                )

            observables_for_dataset.append(observable)

            outputs = observable.evaluate(
                times=simulation_times_jax,
                concentrations=dummy_concentrations,
                species_index=species_index,
                dataset_config=dataset_config,
                experiment_config=solver_params,
            )

            preview_outputs.update(outputs)

        data_file = _resolve_path(base_dir, dataset_config["data_file"])
        frame = pd.read_csv(data_file)
        n_points = len(frame)
        time_column = dataset_config.get("time_column")

        column_mapping = tuple(observables_dict.items())
        missing_columns = [column for _, column in column_mapping if column not in frame.columns]
        if missing_columns:
            raise ValueError(
                f"Dataset '{dataset_config.get('name', data_file.name)}' is missing required data columns {missing_columns}."
            )

        time_values = _get_time_values_for_dataset(dataset_config, base_dir, frame)
        if time_values.ndim != 1 or time_values.size == 0:
            raise ValueError(
                f"Dataset '{dataset_config.get('name', data_file.name)}' must contain at least one time value."
            )

        if time_values.size != n_points:
            raise ValueError(
                f"Dataset '{dataset_config.get('name', data_file.name)}' has {n_points} rows, but resolved {time_values.size} time values."
            )

        time_indices_np = np.searchsorted(simulation_times_np, time_values)
        if not np.array_equal(simulation_times_np[time_indices_np], time_values):
            raise ValueError(
                f"Dataset '{dataset_config.get('name', data_file.name)}' contains time values that could not be matched to the simulation grid."
            )

        for obs_name in observable_names:
            for output_name in calculated_observables[obs_name].output_names:
                if output_name not in preview_outputs:
                    raise ValueError(
                        f"Observable '{obs_name}' did not produce expected output '{output_name}'."
                    )

        condition_indices_np = _build_condition_indices_for_dataset(
            dataset_config=dataset_config,
            frame=frame,
            species_index=species_index,
            base_y0=y0_template,
            register_condition=register_condition,
        )

        if condition_indices_np.size != n_points:
            raise ValueError(
                f"Dataset '{dataset_config.get('name', data_file.name)}' produced {condition_indices_np.size} condition indices for {n_points} rows."
            )

        value_parts: list[np.ndarray] = []
        sigma_parts: list[np.ndarray] = []
        for output_name, column_name in column_mapping:
            values = frame[column_name].to_numpy(dtype=np.float64)
            sigma = _build_sigma_array(
                values=values,
                frame=frame,
                output_name=output_name,
                dataset_config=dataset_config,
                experiment_config=solver_params,
                noise_models=noise_models,
            )
            value_parts.append(values)
            sigma_parts.append(sigma)
            observed_labels.append(f"{dataset_config.get('name', dataset_config['name'])}::{output_name}")

        dataset_bundle = DatasetBundle(
            name=dataset_config["name"],
            dataset_type=dataset_config["dataset_type"],
            observable_names=observable_names,
            data_file=str(data_file),
            time_column=time_column,
            time_values=time_values,
            time_indices_np=time_indices_np,
            time_indices_jax=jnp.asarray(time_indices_np),
            condition_indices_np=condition_indices_np,
            condition_indices_jax=jnp.asarray(condition_indices_np),
            init_cond_columns=tuple((k, v) for k, v in dataset_config.get("init_cond_columns", {}).items()),
            observables_mapping=column_mapping,
            noise_model=dataset_config["noise_model"],
            noise_params=dataset_config.get("noise_params", {}),
            observed_values=np.concatenate(value_parts),
            observed_sigma=np.concatenate(sigma_parts),
            frame=frame,
        )
        datasets.append(dataset_bundle)
        observed_values_parts.append(dataset_bundle.observed_values)
        observed_sigma_parts.append(dataset_bundle.observed_sigma)

    return ExperimentBundle(
        calculation_module_path=str(module_path),
        calculated_observables=calculated_observables,
        datasets=tuple(datasets),
        condition_matrix_np=np.asarray(condition_matrix_rows, dtype=np.float64),
        condition_matrix_jax=jnp.asarray(np.asarray(condition_matrix_rows, dtype=np.float64)),
        simulation_times_np=simulation_times_np,
        simulation_times_jax=simulation_times_jax,
        observed_values=np.concatenate(observed_values_parts).reshape(1, -1),
        observed_sigma=np.concatenate(observed_sigma_parts).reshape(1, -1),
        observed_labels=tuple(observed_labels),
        dataset_configs_by_name=dataset_configs_by_name,
        raw_config=solver_params,
    )


def validate_experiment_config(
    solver_params: Mapping[str, Any],
    solver_params_file: str,
    species_names: list[str],
) -> None:
    """Validate config and dataset wiring before expensive simulation/model build steps."""
    base_dir = _get_path_base_dir(solver_params, solver_params_file)

    calculation_module = solver_params.get("calculation_module")
    if not calculation_module:
        raise KeyError(
            "solver_params.json must define 'calculation_module' pointing to a Python file."
        )

    module_path = _resolve_path(base_dir, calculation_module)
    calculated_observables = load_observable_definitions(module_path, species_names=species_names)
    noise_models = load_noise_model_definitions()

    dataset_configs = solver_params.get("datasets")
    if not dataset_configs:
        raise KeyError("solver_params.json must define a non-empty 'datasets' list.")

    species_index = {name: idx for idx, name in enumerate(species_names)}
    normalized_configs = [
        _normalize_dataset_config(config, idx)
        for idx, config in enumerate(dataset_configs)
    ]
    normalized_configs = [config for config in normalized_configs if config.get("enabled", True)]

    if not normalized_configs:
        raise ValueError("No enabled datasets found. Set at least one dataset with enabled=true.")

    seen_names: set[str] = set()
    for dataset_config in normalized_configs:
        dataset_name = dataset_config["name"]
        if dataset_name in seen_names:
            raise ValueError(f"Duplicate dataset name '{dataset_name}' in datasets list.")
        seen_names.add(dataset_name)

        observables_dict = dataset_config.get("observables", {})
        if not isinstance(observables_dict, dict):
            raise TypeError(f"Dataset '{dataset_name}' 'observables' key must be a dictionary.")

        observable_names = tuple(observables_dict.keys())

        for observable_name in observable_names:
            if observable_name not in calculated_observables:
                raise KeyError(
                    f"Dataset '{dataset_name}' references unknown observable '{observable_name}'."
                )

            observable = calculated_observables[observable_name]
            missing_species = [
                species for species in observable.required_species if species not in species_index
            ]
            if missing_species:
                raise ValueError(
                    f"Observable '{observable_name}' requires species {missing_species}, but only {species_names} are present."
                )

        data_file = _resolve_path(base_dir, dataset_config["data_file"])
        if not data_file.exists():
            raise FileNotFoundError(f"Dataset file not found: {data_file}")

        frame = pd.read_csv(data_file)
        if frame.empty:
            raise ValueError(f"Dataset '{dataset_name}' is empty: {data_file}")

        column_mapping = tuple(observables_dict.items())
        missing_columns = [column for _, column in column_mapping if column not in frame.columns]
        if missing_columns:
            raise ValueError(
                f"Dataset '{dataset_name}' is missing required data columns {missing_columns}."
            )

        _get_time_values_for_dataset(dataset_config, base_dir, frame)

        if dataset_config["dataset_type"] not in SUPPORTED_DATASET_TYPES:
            raise ValueError(
                f"Dataset '{dataset_name}' has unsupported dataset_type '{dataset_config['dataset_type']}'. "
                f"Supported types: {sorted(SUPPORTED_DATASET_TYPES)}."
            )

        _build_condition_indices_for_dataset(
            dataset_config=dataset_config,
            frame=frame,
            species_index=species_index,
            base_y0=np.zeros((len(species_names),), dtype=np.float64),
            register_condition=lambda _y0_values: 0,
        )

        dataset_species = set(dataset_config.get("init_cond_columns", {}).keys()) | set(
            dataset_config.get("init_cond_overrides", {}).keys()
        )
        if not dataset_species:
            raise ValueError(
                f"Dataset '{dataset_name}' must define at least one initial condition source via "
                "'init_cond_columns' and/or 'init_cond_overrides'."
            )

        if dataset_config["noise_model"] not in noise_models and dataset_config["noise_model"] is not None:
            raise ValueError(
                f"Dataset '{dataset_name}' uses unsupported noise_model '{dataset_config['noise_model']}'. "
                f"Supported models: {sorted(noise_models.keys())} or use `None`."
            )

        for output_name, column_name in column_mapping:
            values = frame[column_name].to_numpy(dtype=np.float64)
            _build_sigma_array(
                values=values,
                frame=frame,
                output_name=output_name,
                dataset_config=dataset_config,
                experiment_config=solver_params,
                noise_models=noise_models,
            )


def build_nominal_parameter_tuple(
    param_values: Mapping[str, float],
    free_params: list[str],
) -> tuple[float, ...]:
    """Helper for notebook smoke tests and deterministic simulator checks."""
    return tuple(float(param_values[name]) for name in free_params)


def compute_observation_prediction(
    experiment: ExperimentBundle,
    concentrations: jnp.ndarray,
    species_names: list[str],
) -> jnp.ndarray:
    concentrations = jnp.asarray(concentrations)
    if concentrations.ndim == 2:
        concentrations = concentrations[jnp.newaxis, :, :]
    if concentrations.ndim != 3:
        raise ValueError(
            f"Expected concentrations with ndim 2 or 3, got shape {concentrations.shape}."
        )

    n_conditions = experiment.condition_matrix_jax.shape[0]
    if concentrations.shape[0] != n_conditions:
        raise ValueError(
            f"Expected concentrations for {n_conditions} initial-condition runs, got {concentrations.shape[0]}."
        )

    species_index = {name: idx for idx, name in enumerate(species_names)}
    prediction_parts = []

    for dataset in experiment.datasets:
        sampled_concentrations = concentrations[
            dataset.condition_indices_jax,
            dataset.time_indices_jax,
            :,
        ]
        sampled_times = experiment.simulation_times_jax[dataset.time_indices_jax]
        
        combined_outputs = {}
        for observable_name in dataset.observable_names:
            observable = experiment.calculated_observables[observable_name]
            outputs = observable.evaluate(
                times=sampled_times,
                concentrations=sampled_concentrations,
                species_index=species_index,
                dataset_config=experiment.dataset_configs_by_name[dataset.name],
                experiment_config=experiment.raw_config,
            )
            combined_outputs.update(outputs)

        for output_name, _ in dataset.observables_mapping:
            prediction_parts.append(combined_outputs[output_name])

    return jnp.concatenate(prediction_parts).reshape(1, -1)


def format_experiment_summary(experiment: ExperimentBundle) -> str:
    lines = [
        f"Calculation module: {experiment.calculation_module_path}",
        f"Datasets loaded: {len(experiment.datasets)}",
        f"Unique initial conditions: {len(experiment.condition_matrix_np)}",
        f"Unique simulation times: {len(experiment.simulation_times_np)}",
    ]
    for dataset in experiment.datasets:
        mapped_columns = ", ".join(
            f"{output}->{column}" for output, column in dataset.observables_mapping
        )
        dataset_config = experiment.dataset_configs_by_name.get(dataset.name, {})
        init_overrides = dataset_config.get("init_cond_overrides", {})
        init_sources = []
        if dataset.init_cond_columns:
            init_sources.extend(
                f"{species}<-{column}" for species, column in dataset.init_cond_columns
            )
        if init_overrides:
            init_sources.extend(
                f"{species}<-override" for species in init_overrides.keys()
            )
        if init_sources:
            init_mapping = ", ".join(init_sources)
        else:
            init_mapping = "(none)"
        lines.append(
            f"  - {dataset.name}: type='{dataset.dataset_type}', observables={list(dataset.observable_names)}, "
            f"noise='{dataset.noise_model}', points={len(dataset.time_values)}, init=[{init_mapping}], columns=[{mapped_columns}]"
        )
    return "\n".join(lines)


def load_noise_model_definitions() -> dict[str, NoiseModelDefinition]:
    return {
        "relative_mean": NoiseModelDefinition(
            name="relative_mean",
            compute=_noise_relative_mean,
            description="Constant sigma set to frac * mean(abs(values)).",
        ),
        "relative_pointwise": NoiseModelDefinition(
            name="relative_pointwise",
            compute=_noise_relative_pointwise,
            description="Pointwise sigma set to frac * abs(values).",
        ),
        "absolute": NoiseModelDefinition(
            name="absolute",
            compute=_noise_absolute,
            description="Constant sigma value.",
        ),
        "relative_plus_floor": NoiseModelDefinition(
            name="relative_plus_floor",
            compute=_noise_relative_plus_floor,
            description="sigma = floor + frac * abs(values).",
        ),
        "column": NoiseModelDefinition(
            name="column",
            compute=_noise_column,
            description="Per-point sigma from configured data columns.",
        ),
        "groupwise": NoiseModelDefinition(
            name="groupwise",
            compute=_noise_groupwise,
            description="Group-derived sigma repeated for each group row.",
        ),
    }


def load_observable_definitions(
    module_path: str | Path,
    species_names: list[str] | None = None,
) -> dict[str, ObservableDefinition]:
    module_path = Path(module_path).resolve()
    spec = spec_from_file_location(module_path.stem, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load calculation module from {module_path}.")

    module = module_from_spec(spec)
    spec.loader.exec_module(module)

    raw_observables = getattr(module, "OBSERVABLES", None)
    observable_factory = getattr(module, "build_observables", None)
    if callable(observable_factory):
        raw_observables = observable_factory(species_names or [])

    if not raw_observables:
        raise AttributeError(
            f"Calculation module '{module_path.name}' must define a non-empty OBSERVABLES mapping or build_observables(species_names)."
        )

    normalized: dict[str, ObservableDefinition] = {}
    for name, value in raw_observables.items():
        if isinstance(value, ObservableDefinition):
            observable = value
        elif isinstance(value, Mapping):
            observable = ObservableDefinition(
                name=name,
                compute=value["compute"],
                output_names=tuple(value["output_names"]),
                required_species=tuple(value.get("required_species", ())),
                description=value.get("description", ""),
            )
        else:
            raise TypeError(
                f"Observable '{name}' must be an ObservableDefinition or mapping; got {type(value)!r}."
            )

        if observable.name != name:
            observable = ObservableDefinition(
                name=name,
                compute=observable.compute,
                output_names=observable.output_names,
                required_species=observable.required_species,
                description=observable.description,
            )
        normalized[name] = observable

    return normalized


def _build_sigma_array(
    values: np.ndarray,
    frame: pd.DataFrame,
    output_name: str,
    dataset_config: Mapping[str, Any],
    experiment_config: Mapping[str, Any],
    noise_models: Mapping[str, NoiseModelDefinition],
) -> np.ndarray:
    noise_model_name = dataset_config.get("noise_model", None)
    if noise_model_name is None:
        return np.zeros(len(values)).astype(np.float64)
    if noise_model_name not in noise_models:
        raise ValueError(
            f"Unsupported noise_model '{noise_model_name}'. Available: {sorted(noise_models.keys())}."
        )
    sigma = noise_models[noise_model_name].evaluate(
        values=values,
        frame=frame,
        output_name=output_name,
        dataset_config=dataset_config,
        experiment_config=experiment_config,
    )
    return sigma.astype(np.float64)


def _get_time_values_for_dataset(
    dataset_config: Mapping[str, Any],
    base_dir: Path,
    frame: pd.DataFrame | None = None,
) -> np.ndarray:
    n_points = len(frame) if frame is not None else None

    if "time_values" in dataset_config:
        values = np.asarray(dataset_config["time_values"], dtype=np.float64)
        if values.ndim == 0:
            values = values.reshape(1)
        if n_points is not None and values.size == 1 and n_points > 1:
            values = np.full((n_points,), float(values[0]), dtype=np.float64)
        return values

    time_column = dataset_config.get("time_column", "time")
    if frame is None:
        data_file = _resolve_path(base_dir, dataset_config["data_file"])
        frame = pd.read_csv(data_file)
    if time_column not in frame.columns:
        raise ValueError(
            f"Dataset '{dataset_config.get('name', '<unnamed>')}' has no 'time_values' list in the config "
            f"and is missing column '{time_column}' in the data file. "
            f"Provide either 'time_values' in the dataset config or a 'time_column' that exists in the CSV."
        )
    return frame[time_column].to_numpy(dtype=np.float64)


def _build_simulation_times(
    dataset_configs: list[Mapping[str, Any]],
    base_dir: Path,
) -> np.ndarray:
    time_vectors = []
    for dataset_config in dataset_configs:
        if not dataset_config.get("enabled", True):
            continue
        data_file = _resolve_path(base_dir, dataset_config["data_file"])
        frame = pd.read_csv(data_file)
        time_vectors.append(_get_time_values_for_dataset(dataset_config, base_dir, frame))

    if not time_vectors:
        raise ValueError("No enabled datasets found while building simulation_times.")

    return np.unique(np.concatenate(time_vectors))


def _build_condition_indices_for_dataset(
    dataset_config: Mapping[str, Any],
    frame: pd.DataFrame,
    species_index: Mapping[str, int],
    base_y0: np.ndarray,
    register_condition: Callable[[np.ndarray], int],
) -> np.ndarray:
    init_cond_columns = dataset_config.get("init_cond_columns", {})
    init_cond_overrides = dataset_config.get("init_cond_overrides", {})
    n_points = len(frame)

    if not init_cond_columns and not init_cond_overrides:
        raise ValueError(
            f"Dataset '{dataset_config.get('name', '<unnamed>')}' must define initial conditions via "
            "'init_cond_columns' and/or 'init_cond_overrides'."
        )

    missing_species = [
        species
        for species in list(init_cond_columns.keys()) + list(init_cond_overrides.keys())
        if species not in species_index
    ]
    if missing_species:
        raise ValueError(
            f"Dataset '{dataset_config.get('name', '<unnamed>')}' references unknown species in init conditions: {missing_species}."
        )

    missing_columns = [
        column_name
        for column_name in init_cond_columns.values()
        if column_name not in frame.columns
    ]
    if missing_columns:
        raise ValueError(
            f"Dataset '{dataset_config.get('name', '<unnamed>')}' is missing init condition columns {missing_columns}."
        )

    condition_indices = []
    for row_i in range(n_points):
        y0_values = np.array(base_y0, dtype=np.float64)
        for species, value in init_cond_overrides.items():
            y0_values[species_index[species]] = float(value)
        for species, column_name in init_cond_columns.items():
            y0_values[species_index[species]] = float(frame.iloc[row_i][column_name])

        condition_indices.append(register_condition(y0_values))

    return np.asarray(condition_indices, dtype=np.int32)


def _invoke_user_function(func: Callable[..., Any], **available_kwargs: Any) -> Any:
    func_signature = signature(func)
    if any(parameter.kind.name == "VAR_KEYWORD" for parameter in func_signature.parameters.values()):
        return func(**available_kwargs)

    supported_kwargs = {
        name: value
        for name, value in available_kwargs.items()
        if name in func_signature.parameters
    }
    return func(**supported_kwargs)


def _resolve_path(base_dir: Path, path_value: str | Path) -> Path:
    path = Path(path_value).expanduser()
    if path.is_absolute():
        return path
    return (Path(base_dir) / path).resolve()


def _get_path_base_dir(solver_params: Mapping[str, Any], solver_params_file: str | Path) -> Path:
    solver_params_dir = Path(solver_params_file).resolve().parent
    return _resolve_path(solver_params_dir, solver_params.get("path_base", "."))


SUPPORTED_DATASET_TYPES = {
    "timeseries",
    "endpoint",
}


def _normalize_dataset_config(
    dataset_config: Mapping[str, Any],
    dataset_index: int,
) -> dict[str, Any]:
    config = dict(dataset_config)
    if "observables" not in config:
        raise KeyError(f"Dataset index {dataset_index} is missing required key 'observables'.")
    if "data_file" not in config:
        raise KeyError(f"Dataset index {dataset_index} is missing required key 'data_file'.")

    obs_dict = config["observables"]
    if isinstance(obs_dict, dict) and obs_dict:
        fallback_suffix = list(obs_dict.keys())[0]
    else:
        fallback_suffix = "dataset"

    config.setdefault("name", f"dataset_{dataset_index}_{fallback_suffix}")
    config.setdefault("dataset_type", _infer_dataset_type(config))
    config.setdefault("noise_model", "relative_mean")
    config.setdefault("noise_params", {})
    config.setdefault("init_cond_columns", {})
    config.setdefault("init_cond_overrides", {})
    config.setdefault("enabled", True)

    return config


def _infer_dataset_type(dataset_config: Mapping[str, Any]) -> str:
    if "time_values" in dataset_config:
        values = np.asarray(dataset_config["time_values"], dtype=np.float64)
        if values.size == 1:
            return "endpoint"
    return "timeseries"


def _noise_absolute(values: np.ndarray, noise_params: Mapping[str, Any], **_: Any) -> np.ndarray:
    value = float(noise_params.get("value", 0.05))
    return np.full(values.shape, value, dtype=np.float64)


def _noise_relative_mean(values: np.ndarray, noise_params: Mapping[str, Any], **_: Any) -> np.ndarray:
    frac = float(noise_params.get("frac", 0.05))
    return np.full(values.shape, frac * float(np.mean(np.abs(values))), dtype=np.float64)


def _noise_relative_pointwise(values: np.ndarray, noise_params: Mapping[str, Any], **_: Any) -> np.ndarray:
    frac = float(noise_params.get("frac", 0.05))
    floor = float(noise_params.get("floor", 0.0))
    return floor + frac * np.abs(values)


def _noise_relative_plus_floor(values: np.ndarray, noise_params: Mapping[str, Any], **_: Any) -> np.ndarray:
    frac = float(noise_params.get("frac", 0.05))
    floor = float(noise_params.get("floor", 1.0e-6))
    return floor + frac * np.abs(values)


def _noise_column(
    frame: pd.DataFrame,
    output_name: str,
    noise_params: Mapping[str, Any],
    **_: Any,
) -> np.ndarray:
    column_mapping = dict(noise_params.get("column_mapping", {}))
    sigma_column = column_mapping.get(output_name)
    if not sigma_column:
        raise ValueError(
            f"Noise model 'column' requires a valid column match for output '{output_name}' in noise_params.column_mapping."
        )
    if sigma_column not in frame.columns:
        raise ValueError(f"Noise model 'column' references missing column '{sigma_column}'.")
    return frame[sigma_column].to_numpy(dtype=np.float64)


def _noise_groupwise(
    values: np.ndarray,
    frame: pd.DataFrame,
    noise_params: Mapping[str, Any],
    **_: Any,
) -> np.ndarray:
    group_column = noise_params.get("group_column")
    if not group_column:
        raise ValueError("Noise model 'groupwise' requires noise_params.group_column.")
    if group_column not in frame.columns:
        raise ValueError(f"Noise model 'groupwise' missing group column '{group_column}'.")

    statistic = str(noise_params.get("statistic", "std"))
    if statistic == "std":
        mapped_sigmas = frame.groupby(group_column)[group_column].transform("std")
    elif statistic == "sem":
        mapped_sigmas = frame.groupby(group_column)[group_column].transform("sem")
    elif statistic == "mad":
        mapped_sigmas = frame.groupby(group_column)[group_column].transform(
            lambda x: np.median(np.abs(x - np.median(x)))
        )
    else:
        raise ValueError(f"Unknown groupwise statistic '{statistic}'. Options: std, sem, mad")

    return mapped_sigmas.to_numpy(dtype=np.float64)