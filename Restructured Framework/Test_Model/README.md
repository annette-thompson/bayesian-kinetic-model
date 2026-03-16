# Test_Model

This folder is a full-coverage fixture package for debugging the generalized Bayesian ODE inference framework.

## Coverage goals

- Dataset types covered: `timeseries`, `replicate_timeseries`, `endpoint`, `rate`
- Noise models covered: `relative_mean`, `relative_pointwise`, `relative_plus_floor`, `absolute`, `column`, `groupwise`
- Condition handling covered: global `init_conds`, `init_cond_columns`, and grouped replicate behavior
- Reaction loading covered: directory of multiple reaction JSON files

## Folder layout

- `Data/`: CSV fixtures for each dataset/noise case
- `Reactions/`: split reaction JSON files used together as one model
- `Results/`: output target folder for netCDF and plots
- `test_calculations.py`: OBSERVABLES used by all fixture datasets
- `solver_params_test.json`: pre-wired config to run inference immediately

## Quick run

Use `inference_runner.run_bayesian_inference(...)` with:

- `solver_params_file="./Test_Model/solver_params_test.json"`
- `reactions_source="./Test_Model/Reactions"`
- `savedir="./Test_Model/Results"`

Example parameter file mapping is summarized in `Data/fixture_matrix.csv`.

## Notes

- Data values are synthetic and designed for schema and workflow validation.
- This package is intended for fast debugging and regression checks, not biological realism.
