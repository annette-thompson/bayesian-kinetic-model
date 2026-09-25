"""Is the high-a1 region a flat plateau? Profile the log posterior over the prior range for
the C10 [0.05,20] config, and report the gradient a chain there would feel."""
import json, sys
sys.path.insert(0, "/projects/anth4580/Bayesian/Utilities")
import numpy as np, inference_runner as ir
CFG = "/projects/anth4580/Bayesian/Results/Chain Scaling Tests/Chain C10 - a1_0.05-20_no_floor/solver_params.json"
b = ir._build_model_bundle(ir.import_solver_params(CFG))
m = b.pm_model
logp, dlogp = m.compile_logp(), m.compile_dlogp()
print("%10s %14s %16s %s" % ("a1", "log posterior", "d logp/d log a1", "note"))
prev = None
for a1 in (0.05, 0.1, 0.3, 0.43, 0.7, 1.0, 1.5, 3.0, 5.0, 9.4, 10.7, 15.0, 20.0):
    pt = {"a1_log__": float(np.log(a1))}
    lp = float(logp(pt)); g = float(np.asarray(dlogp(pt)).ravel()[0])
    note = ""
    if abs(g) < 1.0: note = "flat: gradient cannot pull a chain out"
    if prev is not None and lp > prev: note += " (rising)"
    print("%10.2f %14.2f %16.3f %s" % (a1, lp, g, note))
    prev = lp
