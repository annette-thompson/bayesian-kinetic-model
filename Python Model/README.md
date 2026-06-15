# Bayesian Inference — Quick Start

A generic framework for fitting Bayesian kinetic parameters to enzyme reaction networks using PyMC, JAX-accelerated ODE solving, and MCMC sampling. Any reaction network defined as YAML files can be used — *E. coli* type II FAS is the worked example included here, solving for binding parameters of **FabD**.

---

## Getting Started

### 1. Create the environment
```bash
conda env create -f environment.yml
conda activate Bayesian
```

### 2. Run the model
Open and run **`ODE Runner/run_model.ipynb`**.

This notebook lets you run the model with the given reactions in `Reactions/EC_FAS_ME1/`. Includes demonstration of `reaction_sanity_checker.py` to help identify any mistakes in reactions and `query` property of the reaction class.

The bottom of this notebook also has some example functions to generate the data used for Bayesian inference.

### 3. Configure the model for Bayesian inference
Open and run **`Bayesian Inference/guided_solver_config_builder.ipynb`**.

This notebook walks through every setting step by step and writes `solver_params.json` — the configuration file read by the inference runner. Set both:

| Variable | Example | Purpose |
|---|---|---|
| `name` | `"FabD"` | Model label used in the solver params filename, e.g. `solver_params.json` |
| `folder_name` | `"FabD Inference"` | Folder used to organize reactions, data, calculation files, solver params, posterior samples, and plots |

The generated solver params JSON stores `path_base`, which points back to the Python Model home directory. All model paths inside the JSON are relative to that home directory, so the JSON can move with its results folder without requiring absolute paths.

### 4. Run inference and plot results
Open and run **`Bayesian Inference/guided_bayesian_inference.ipynb`**.

| Variable | Value | Effect |
|---|---|---|
| `use_existing_results` | `False` | Run full MCMC inference (~10 min with current (really low) sampling values) |
| `use_existing_results` | `True` | Load a saved posterior and go straight to plotting |
| `name` | `"FabD"` | Must match the `name` set in the config builder |
| `folder_name` | `"FabD Inference"` | Must match the `folder_name` set in the config builder |

Results (posterior samples and plots) are saved to `Results/<folder_name>/`.

---

## Folder Structure

| Path | Purpose |
|---|---|
| `Bayesian Inference/` | Config builder + main inference notebook — configure parameters, run MCMC, visualize results |
| `ODE Runner/` | Standalone deterministic ODE runner — simulate the full FAS network without Bayesian inference |
| `Reactions/EC_FAS_ME1/` | YAML files with reaction mechanisms and kinetic parameters |
| `Reactions/<folder_name>/` | Reaction YAML/JSON files for a Bayesian inference run, e.g. `Reactions/FabD Inference/` |
| `Data/<folder_name>/` | Experimental CSV files for a Bayesian inference run, e.g. `Data/FabD Inference/` |
| `Calculation Files/<folder_name>/` | Observable extraction functions for a Bayesian inference run, e.g. `Calculation Files/FabD Inference/` |
| `Results/<folder_name>/` | Solver params JSON, posterior/prior samples, and plots for a Bayesian inference run |
| `Utilities/` | Core library: ODE builder, reaction sanity checker, inference engine, plotting, experiment framework |

For the current FabD inference example, use `folder_name = "FabD Inference"`. The matching files live in:

| Path | Contents |
|---|---|
| `Reactions/FabD Inference/FabD.yaml` | FabD reaction mechanism |
| `Data/FabD Inference/` | FabD endpoint and time-series CSV files |
| `Calculation Files/FabD Inference/FabD_calculations.py` | FabD observable calculations |
| `Results/FabD Inference/solver_params.json` | Generated solver configuration |

---

## Output Files

After running inference, `Results/<folder_name>/` will contain:

| File | Contents |
|---|---|
| `solver_params.json` | Solver, dataset, path, prior, and sampler configuration |
| `prior_samples_pm.nc` | Prior samples (NetCDF / ArviZ format) |
| `posterior_samples_pm.nc` | Posterior samples (NetCDF / ArviZ format) |
| `trace_plot.png` | Posterior trace diagnostics |
| `trace_plot_hist.pdf` | Marginal posterior distributions |
| `trace_plot_prior_hist.pdf` | Prior distributions |
| `*_predictive_*.pdf` | Model predictions overlaid on observed data |

---

## View Existing Data and Config Files

Already-generated files are in:

- `Results/<folder_name>/solver_params.json` (config used for that run)
- `Results/<folder_name>/` (saved plots + `prior_samples_pm.nc` + `posterior_samples_pm.nc`)
- `Data/<folder_name>/` (CSV data files)

To replot existing results without rerunning MCMC, open `Bayesian Inference/guided_bayesian_inference.ipynb`, set `folder_name` to the run folder, set `use_existing_results = True`, and run the Configure and Run cell.
---

## Dependencies

All required packages are in `environment.yml`. Key ones:

| Package | Role |
|---|---|
| `pymc` | Probabilistic programming and MCMC sampling |
| `jax` + `diffrax` | Fast stiff ODE solving with automatic differentiation |
| `equinox` | JAX-compatible ODE system module |
| `preliz` | Prior specification via maximum entropy |
| `arviz` | Inference diagnostics (trace plots, LOO, WAIC) |
| `nutpie` | Fast NUTS sampler backend (compiled Rust) |

---


