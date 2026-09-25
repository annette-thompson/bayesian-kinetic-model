# ME1 refit

A Python replacement for the ME1 scaling-parameter fit, originally
`Combined_Pathway_Optimizer.m` + `Combined_Pathway_Handler.m` + `param_func.m`
(`Git Repositories/Matlab Current Projects/Previous Models/ME1`).

**Only the optimization was translated.** The model is not re-implemented:
`Reactions/EC_FAS_ME1/C20+unsat+FBinit` was verified reaction-for-reaction against the
MATLAB ODEs — 588 directed steps, every rate constant equal — and its YAML already declares
all 18 scaling parameters, including the composite expressions `(1/b3)`, `(b1/b2)`, `(f*c2)`
and the TesA free-energy terms `1/exp(n*d1+d2)`. Setting 18 numbers makes the builder
recompute every rate constant, which is what `param_func.m` did in 463 lines of branching.

## Files

| file | what it is |
|---|---|
| `ME1_refit.ipynb` | the notebook: verify the published fit, inspect the objective, refit |
| `me1_config.py` | the 18 parameters, bounds, conditions, solver settings — every literal the MATLAB had inline |
| `me1_model.py` | solve at a given p_vec; batched over conditions |
| `me1_objective.py` | obj1 / obj2 / obj3 and the guard rails, from `Combined_Pathway_Handler.m` |
| `data/initial_rates.csv` | the 7 initial-rate conditions (was `rate_exp`, a bare literal) |
| `data/timecourse.csv` | reference time course (was `Experimental_Dataset.csv`) |
| `data/profile_fractions.csv` | target chain-length distribution (was `fit_dist`, a literal) |
| `data/parameter_crosswalk.csv` | every MATLAB parameter name + index ↔ YAML rate key, reaction, scaling group |

## Quick start

```python
import me1_config as cfg, me1_model as mm, me1_objective as mo
model = mm.ME1Model()                       # ~3 s
res = mo.evaluate(model, cfg.PUBLISHED)     # ~45 s first call, ~23 s after
print(res.obj1, res.obj2, res.obj3, res.total)
```

## What changed, and why

- **Parameters are named.** `p_vec(4) > 6.29E4` became `UPPER_BOUNDS["b1"]`, with a line
  saying what `b1` does.
- **Targets are data files**, not literals buried in the objective.
- **Conditions are declarative.** Adding one means appending to `RATE_CONDITIONS` and a CSV
  row; the MATLAB needed a new `enz_conc_*` vector, a new solver call and a new literal.
- **The 7 conditions solve in one batched call** instead of a loop.
- **Fitting is in log10 space** (except `d1`/`d2`, which are additive). Published values span
  0.0054 to 142,474 and Nelder-Mead steps in absolute units, so a simplex that can move `c3`
  cannot move `a1`. The MATLAB did not do this.
- **The `1e8` sentinel is documented as a constraint**, not error handling: `fminsearch` is
  unconstrained, so that value is the only thing keeping the search physical.

## Substrate concentrations are fixed on purpose

Every condition is solved at 0.5 mM malonyl-CoA / 0.5 mM acetyl-CoA / 1 mM NADPH / 1 mM
NADH, regardless of what the individual assay used. The model is answerable to both in
vitro data (initial rates, time course) and in vivo data (the chain-length profile, from a
TesA-overexpressing strain), and this composition is the one reported to best represent the
in vivo cytosol. In vivo the only lever is expression level, so enzyme concentrations vary
between conditions and substrates do not. Matching each in vitro buffer individually would
break the shared basis that lets one parameter set serve both.

## Two properties of the objective

Worth knowing before reading a refit; each affects where the optimizer goes.

1. **obj3 constrains shape, not titre.** It compares the profile against
   `total_FA × fractions` using the model's *own* total.

2. **The three terms are multiplied.** Relative weighting is whatever the terms happen to be
   in their own units. `combine="sum"` / `"log_sum"` exist but do not reproduce the paper.

## Cost

One objective evaluation is ~23 s: eight stiff solves of a 318-species system. Nelder-Mead
on five parameters typically wants a few hundred, so a real refit is a 1–3 hour background
job. `MAX_EVALS` in the notebook is set low so it runs end to end; raise it for a real fit.

Batching the seven conditions is roughly cost-neutral on CPU — `vmap` runs the stiff solves
in lockstep, so the batch moves at the pace of its slowest member. It is kept because it is
clearer and would win on GPU.

## Not carried over

The MATLAB's ACC block (`k1_*`, 7 species) is inert there (`enzyme_conc(1) = 0`, marked "not
used") and absent from the reaction files. The FabF acetyl-CoA branch (`k8_4`, `k8_5`) is
present in the MATLAB with all four constants at zero and simply omitted from the YAML.
Neither carries flux in either model.
