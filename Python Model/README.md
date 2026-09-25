# Bayesian Inference — Quick Start

A framework for Bayesian inference of kinetic parameters in enzyme reaction networks: a PyMC model over a JAX/diffrax ODE solve, sampled with BlackJAX NUTS in resumable, checkpointed segments. Any network defined as YAML reaction files can be used. The worked example is *E. coli* type II fatty-acid synthesis (the ME1 model), fitted through its 18 scaling parameters on truncated chain-length systems (C4_NoFB through C20+unsat) against synthetic data with a known answer.

What the paper argues and which runs produce each figure: `Notes/paper_outline.md` and `Notes/tier1_experiment_plan.md`.

---

## Getting Started

### 1. Create the environment
```bash
conda env create -f environment.yml
conda activate Bayesian
```

### 2. Run the model
Open and run **`ODE Runner/run_model.ipynb`**. It loads one reaction set (`Reactions/EC_FAS_ME1/C8/` by default), checks it with `reaction_sanity_check.py`, shows the reaction class's `query` helper, then solves and plots the network.

Every build of the model states all scaling values explicitly: `build_ode_system_from_reactions(path, scaling_group={...})`. Multiplicative groups are 1 at nominal; `d`-prefixed groups enter inside an exponential and are **0** at nominal. Setting a `d` group to 1 does not fail, but it silently rescales TesA by ~4×10⁵ at C12. For a no-op build, `nominal_scaling_group_values(discover_scaling_groups(path))` gives the right values.

### 3. Make synthetic data
```bash
cd job_files/tier1
python -u make_tier1_rate_data.py C8
```
This writes `Data/Tier1_rates/Chain_C8/`: the noisy files the fit sees, a noise-free `clean/` copy, and `ground_truth.json`. Options cover noise level (`--noise_frac`), variant reaction sets (`--reactions`), off-nominal truths (`--set GROUP=VALUE`) and the output name (`--out_name`).

### 4. Build run configs
```bash
python build_tier1_configs.py --plan                      # every planned Tier-1 run
python build_tier1_configs.py --system C8 --params a1,c3  # one run
```
Each run gets `Results/Tier1/<run>/solver_params.json`, the one file the inference runner reads. Its `path_base` points back to the `Python Model` folder, and every path inside is relative to that folder, so a config can move with its results folder. It records the data's truth as `tier1_truth`.

### 5. Pre-flight, then run
```bash
python -u check_model_vs_data.py "Tier1 C8 - a1c3" --grad   # model reproduces its clean data; logp and gradient finite
```
Then run inference (see [Running Inference](#running-inference)), and score the result against the recorded truth:
```bash
python recovery_report.py --only "Tier1 C8 - a1c3"   # z, shrinkage, 50/90/95% coverage
```

---

## Folder Structure

| Path | Purpose |
|---|---|
| `Utilities/` | Core library: reaction-network builder, sanity checker, inference runner, resumable sampler, plotting, experiment framework, data generation |
| `Calculation Files/Full_FAS/FA_conc.py` | Observables: fatty-acid concentrations, C16 equivalents, initial rate (µM C16/min), mole fractions |
| `Reactions/EC_FAS_ME1/<system>/` | Reaction YAMLs per truncated system (C4 … C20+unsat), plus variants: `C14+unsat+c3split` (TesA's `c3` split in two) and `C20+unsat+FBinit` (matches the ME1 MATLAB model reaction for reaction) |
| `Reactions/Camelina_FAS_simple/` | A simplified Camelina FAS network, used by the model-error runs |
| `Data/Chain_<system>/` | Data for the single-parameter chain-ladder fits |
| `Data/Tier1_rates/` | Tier-1 data for the current runs: time series, chain-length profile and initial rates |
| `Data/Tier1/` | The earlier endpoint-condition Tier-1 design, which the chain-count test was fit to |
| `Data/Experimental/` | The ME1 Dataset S1 measurements and reference time course |
| `Results/` | One folder per run (see [Output Files](#output-files)): `Chain Scaling Tests/` (the ladder), `Chain Count Test/`, `Tier1/`, `Experimental Comparison/`, `Model Error/` |
| `job_files/` | Cluster job scripts and per-study tools: `gpu_submit.sh`, `tier1/`, `chain_scaling_tests/`, `multiparam_tests/`, `chain_system_sensitivity_analysis/` |
| `ODE Runner/` | Notebooks for deterministic solves, ODE-solver tuning and model-error configs |
| `Bayesian Inference/` | Interactive notebooks and the Alpine segment-chain scripts |
| `ME1_Refit/` | A Python port of the original ME1 scaling-parameter fit |
| `Notes/` | Paper outline, Tier-1 run plan, audit and decision records |
| `Sync/` | rsync to and from the clusters |
| `_staging_new_schema/` | Mock-up of a more general, less FAS-specific reaction schema (future work) |
| `*/Archive/` | Retired material, including the earlier one-enzyme-at-a-time stage (`Test_FabD*`) |

---

## Output Files

A finished run's folder, `Results/<...>/<run>/`, contains:

| File | Contents |
|---|---|
| `solver_params.json` | Solver, dataset, path, prior and sampler configuration |
| `prior_samples_pm.nc` | Prior samples (NetCDF / ArviZ) |
| `posterior_samples_pm.nc` | Posterior samples, log-likelihood and posterior predictive (NetCDF / ArviZ) |
| `timing.json` | Posterior-sampling time in seconds |
| `checkpoint/` | Resumable state and every draw: `draws.zarr`, `checkpoint.pkl`, `checkpoint_meta.json`, `progress_log.jsonl`, `status.json`. Not tracked by git. |
| `trace_plot.png` | Prior vs. posterior, zoomed posterior and per-chain trace, one column per free parameter |
| `convergence_diagnostics.png` | r-hat/ESS against cumulative draws (warm-up included) and cumulative divergences |
| `energy_plot.png` | Per-chain BFMI and marginal/transition energy |
| `rank_plot.png` | Per-chain rank-ECDF mixing check |
| `loo_diagnostics.png` | Pareto-k and LOO summary. It needs enough sampling draws for the Pareto tail fit; a few hundred is not always enough. |
| `predictive_plots_<system>.png` | Posterior-predictive check, one row per observable |

While a run finalizes it keeps `finalize_stage.nc`, so a job killed partway through doesn't redo the finished stages. The file is removed once the posterior file is written.

---

## Running Inference

Posterior sampling uses **BlackJAX**, and every draw, warm-up *and* sampling, is **checkpointed** under the run's `checkpoint/`. Runs are therefore resumable across cluster time limits, and a short local run just completes in one call.

### Locally (CPU)

```bash
python "Utilities/inference_runner.py" --solver_params_file "Results/Tier1/Tier1 C8 - a1c3/solver_params.json"
```

`--max_hours H` checkpoints and exits after H hours; running the same command again resumes. `--extra_draws N` raises the draw target, and `--no_resume` discards the checkpoint and starts over.

### On the cluster (GPU)

```bash
job_files/gpu_submit.sh --time 12:15:00 --name tier1_C8_a1c3 -- tier1/tier1.sbatch "Tier1 C8 - a1c3"
```

`gpu_submit.sh` queues the job on both Blanca and Alpine; whichever starts first runs it (`gpu_twin_claim.sh`) and cancels the other. The run samples until it converges, then finalizes and writes its figures. If it hits the time limit first it checkpoints, and resubmitting the same command resumes. Sync the `Python Model` folder first (`Sync/sync_to_cluster.sh`).

`Bayesian Inference/submit_inference_chain.sh` is the older route: a chain of dependent Alpine jobs (CPU, or `--gpu`) that resume one run segment by segment. It has `--segments N`, `--max-hours`, `--time` and `--extra-draws`.

### When a run stops

Convergence is checked every `rhat_check_every` draws (from `posterior_sampling` in `solver_params.json`):

- **Converged:** r-hat ≤ `rhat_threshold` and bulk ESS at or above the bar, on `convergence_consecutive_checks` consecutive checks. The ESS bar is `ess_per_split_chain × 2 × chains` (Vehtari et al. 2021: 400 at four chains, 800 at eight), or a flat `ess_threshold`.
- **Stranded chains:** a chain whose mean log-posterior sits more than `lp_exclusion_nats` (default 20) below the best chain is excluded from that decision and recorded in `checkpoint/status.json` as `stranded_chains`. At least `min_chains_for_convergence` chains must remain. The saved draws keep every chain, but the finalized netcdf, whether written live or by `finalize_window.py`, leaves the stranded ones out unless `--include_stranded` is passed.
- **Compute cap:** `max_total_hours` (default 24) caps the run's **total A100-equivalent compute** across all segments: wall time on each card × its measured speed relative to an A100 (`GPU_SPEED_VS_A100` in `resumable_sampler.py`). Time spent queued or idle doesn't count. Once the cap is crossed the run checkpoints and stops, as `total_time_budget`, and it does not resubmit. Raise it, or set it to `null`, before starting a run meant to go further. `max_sampling_hours` caps the sampling phase alone.
- **Dead runs:** if every chain accepts nothing for two consecutive checkpoints, the run stops as `not_sampling` and is never finalized.

### Re-picking the posterior window

All warm-up and sampling draws are kept, so the posterior window can be moved afterwards without resampling. `posterior_burn_in_draws` in `solver_params.json` adds burn-in beyond `tune`, both live and in the auto-written posterior. `finalize_window.py` re-derives the posterior netcdf from any window:

```bash
python "Utilities/finalize_window.py" --solver_params_file ".../solver_params.json" --burn_in 800 --end_draw 1400
```

`--burn_in` and `--end_draw` are indexes into the full per-chain timeline (warm-up then sampling). With neither given, it reproduces what the run already wrote.

### Plotting diagnostics

Three standalone scripts redraw the figures from the saved posterior netcdf, so no notebook or cluster run is needed:

```bash
python "Utilities/plot_convergence_trajectory.py" --solver_params_file ".../solver_params.json"  # r-hat/ESS, divergences, energy, rank-ECDF, LOO
python "Utilities/plot_trace_diagnostics.py"      --solver_params_file ".../solver_params.json"  # prior vs. posterior and traces
python "Utilities/plot_predictive_check.py"       --solver_params_file ".../solver_params.json"  # posterior-predictive checks
```

They are cheap enough to run locally after copying just `posterior_samples_pm.nc` down from the cluster. `Bayesian Inference/guided_bayesian_inference.ipynb` does the same interactively: set `folder_name` and `use_existing_results = True`.

---

## Dependencies

`environment.yml` (conda-forge) is the reference environment; `requirements.txt` is the pip equivalent. Key packages:

| Package | Role |
|---|---|
| `pymc` | Model definition (priors, transforms, likelihood) and netcdf/arviz plumbing |
| `jax` + `diffrax` | Stiff ODE solving (Kvaerno5, PID step control) with automatic differentiation |
| `blackjax` | The NUTS sampler, checkpointed and resumable, CPU or GPU |
| `equinox` | JAX-compatible ODE system module |
| `preliz` | Priors from bounds and mass. A LogNormal with a fixed median is solved exactly, and every fitted prior is checked against its requested mass. |
| `arviz` | Diagnostics: trace plots, r-hat/ESS, LOO. WAIC is not available in this arviz version. |
| `zarr` | On-disk store for checkpointed draws |
