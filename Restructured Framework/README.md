# Restructured Framework

Bayesian ODE parameter inference framework with configurable reaction models, observables, datasets, and noise models.

## Scope

Use this repository to:
- build an ODE model from reaction JSON,
- map model states to measured outputs,
- infer free kinetic parameters with PyMC,
- evaluate fit with summary statistics and predictive plots.

Supported dataset types:
- `timeseries`
- `replicate_timeseries`
- `endpoint`
- `rate`

Supported noise models:
- `relative_mean`
- `relative_pointwise`
- `absolute`
- `relative_plus_floor`
- `column`
- `groupwise`

## Quick Start

### Notebook workflow

1. Run `guided_solver_config_builder.ipynb` to write `solver_params.json`.
2. Run `guided_bayesian_inference.ipynb` to sample or load existing results.

### Python API workflow

```python
from inference_runner import run_bayesian_inference

result = run_bayesian_inference(
    solver_params_file="./Test_Model/solver_params_test.json",
    reactions_source="./Test_Model/Reactions",
    savedir="./Test_Model/Results",
)

print(result.summary)
```

## Configuration Reference

Configuration is stored in `solver_params.json`.

Top-level keys:
- `calculation_module`: Python file that defines `OBSERVABLES`.
- `datasets`: list of dataset blocks.
- `init_conds`: global default initial conditions by species.
- `free_kinetic_params`: priors for inferred kinetic parameters.
- `prior_sampling`: prior draw settings.
- `posterior_sampling`: posterior draw/tune/chain settings.
- `ODE_solver`: solver name and solve settings.
- `ODE_stepsize_controller`: tolerance/controller settings.

Per-dataset keys:
- `name`: unique dataset name.
- `enabled`: include/exclude dataset.
- `dataset_type`: `timeseries`, `replicate_timeseries`, `endpoint`, or `rate`.
- `data_file`: CSV path.
- `observable`: key from `OBSERVABLES`.
- `column_mapping`: observable output -> CSV column.
- `time_column` or `time_values`: how observation times are specified.
- `init_cond_columns`: species -> CSV column for row-wise initial conditions.
- `init_cond_overrides`: species -> fixed override value.
- `noise_model` and `noise_params`: uncertainty model for that dataset.
- `sigma_column_mapping`: required for `column` noise where applicable.

### Example: free parameter priors

```json
"free_kinetic_params": [
   {
      "rxn_name": "binding_forward",
      "param_name": "k_f",
      "prior_dist_params": {
         "distribution": "LogNormal",
         "lower": 0.001,
         "upper": 1.0,
         "mass": 0.95
      }
   }
]
```

### Example: dataset config (`timeseries`)

```json
{
   "name": "ts_rel_mean",
   "dataset_type": "timeseries",
   "data_file": "./Test_Model/Data/ts_relative_mean.csv",
   "observable": "product_fraction",
   "column_mapping": {"product_fraction": "product_fraction"},
   "time_column": "time",
   "noise_model": "relative_mean",
   "noise_params": {"frac": 0.05}
}
```

### Example: dataset config (`rate` + `column` noise)

```json
{
   "name": "rate_column",
   "dataset_type": "rate",
   "data_file": "./Test_Model/Data/rate_column_sigma.csv",
   "observable": "product_formation_rate",
   "column_mapping": {"product_formation_rate": "product_formation_rate"},
   "time_column": "time",
   "noise_model": "column",
   "sigma_column_mapping": {"product_formation_rate": "rate_sigma"}
}
```

### Example: dataset config (`replicate_timeseries` + `groupwise` noise)

```json
{
   "name": "replicate_groupwise",
   "dataset_type": "replicate_timeseries",
   "data_file": "./Test_Model/Data/replicate_groupwise.csv",
   "observable": "product_fraction",
   "column_mapping": {"product_fraction": "product_fraction"},
   "time_column": "time",
   "noise_model": "groupwise",
   "noise_params": {
      "group_column": "replicate_id",
      "statistic": "std",
      "min_sigma": 0.0001
   }
}
```

## Observable Module Contract

`calculation_module` must expose non-empty `OBSERVABLES`.

Each observable entry supports:
- `name`
- `compute`
- `output_names`
- `required_species`
- `description`

Minimal shape example:

```python
OBSERVABLES = [
      {
            "name": "product_fraction",
            "output_names": ["product_fraction"],
            "required_species": ["P", "S"],
            "compute": lambda times, concentrations, species_index, **kwargs: {
                  "product_fraction": concentrations[:, species_index["P"]] /
                  (concentrations[:, species_index["P"]] + concentrations[:, species_index["S"]] + 1e-12)
            },
      }
]
```

## Pipeline

1. **Reaction loading** (`reaction_model_builder.py`)
    - Build ODE RHS from reaction JSON.
2. **Experiment load + validation** (`experiment_framework.py`)
    - Validate schema/data and build observed vectors.
3. **Observable projection** (`calculation_module`)
    - Map model concentrations to measured outputs.
4. **Inference orchestration** (`inference_runner.py`)
    - Run prior/posterior sampling, compute summary/LOO/WAIC, save netCDF.
5. **Plotting** (`inference_plotting.py`)
    - Generate trace/marginal/prior/predictive diagnostics.

## Plotting Notes

Current predictive plotting behavior:
- observed data include noise error bars,
- `groupwise` datasets are aggregated to group means,
- `rate` datasets are plotted as bars,
- bar plots include a species-name table column.

Notebook-level style controls in `guided_bayesian_inference.ipynb`:
- `plt.style.use("default")` for Matplotlib default style,
- `posterior_plot_color` and `observed_plot_color` passed to `plot_predictive(...)`.

## Key Files

- `experiment_framework.py`: config validation, dataset bundling, noise handling.
- `reaction_model_builder.py`: ODE system construction from reaction files.
- `inference_runner.py`: end-to-end inference orchestration.
- `inference_plotting.py`: all plotting helpers.
- `solver_config_builder.py`: helper utilities for writing config JSON.
- `pytensor_ode_ops.py`: custom PyTensor Ops for ODE solve + VJP.
- `Test_Model/`: full fixture model for debugging and regression checks.

## Validation

`validate_experiment_config(...)` checks common failure points before sampling:
- missing files/modules/columns,
- invalid dataset types or species references,
- unresolved time inputs,
- invalid noise-model wiring.

## Source Attribution

Based on code from:
Systems modeling and uncertainty quantification of AMP-activated protein kinase signaling
Nathaniel Linden-Santangeli, Jin Zhang, Boris Kramer, Padmini Rangamani
bioRxiv 2025.06.02.657503; doi: https://doi.org/10.1101/2025.06.02.657503

Edited with support from GitHub Copilot (GPT-5.3-Codex).