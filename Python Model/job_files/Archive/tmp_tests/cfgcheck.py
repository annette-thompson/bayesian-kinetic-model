"""Build the chain-test model once and evaluate its log density: catches a bad dataset
mapping or sigma column before five GPU jobs queue for it."""
import sys, json
sys.path.insert(0, "/projects/anth4580/Bayesian/Utilities")
import numpy as np, inference_runner as ir
from experiment_framework import format_experiment_summary
cfg = "/projects/anth4580/Bayesian/Results/Chain Count Test/Chain C12 - a1c3_8chains/solver_params.json"
imported = ir.import_solver_params(cfg)
bundle = ir._build_model_bundle(imported)
print(format_experiment_summary(bundle.experiment) if hasattr(bundle, "experiment") else "(no experiment summary attr)")
m = bundle.pm_model
print("free RVs:", [r.name for r in m.free_RVs], "| value vars:", [v.name for v in m.value_vars])
pt = {v.name: 0.0 for v in m.value_vars}
print("logp at nominal (a1=c3=1):", float(m.compile_logp()(pt)))
d = bundle.experiment if hasattr(bundle, "experiment") else None
if d is not None:
    print("observed points:", d.observed_values.size, "| sigma range:", float(np.min(d.observed_sigma)), float(np.max(d.observed_sigma)))
