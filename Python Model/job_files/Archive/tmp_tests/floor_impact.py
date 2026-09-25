"""How much did finalizing no-floor runs with the FLOOR model change their log-likelihood?
For each run: the model's log density at 10 posterior draws, built once with the clamp
and once without. Priors are identical, so the difference is purely likelihood."""
import sys, json
sys.path.insert(0, "/projects/anth4580/Bayesian/Utilities")
import numpy as np, arviz as az, jax.numpy as jnp
import reaction_model_builder as rmb
import inference_runner as ir

def make_call(floor):
    def call(self, t, y, args):
        if self.param_idx_arr.shape[0] == 0:
            return jnp.zeros_like(y)
        theta = jnp.asarray(args)
        conc = y[self.reactant_idx_arr]
        if floor:
            conc = jnp.maximum(conc, 0.0)
        powers = jnp.where(self.reactant_mask_arr, conc ** self.reactant_stoich_arr, 1.0)
        mass = jnp.prod(powers, axis=1)
        scale = jnp.stack([fn(theta) if fn is not None else jnp.ones(()) for fn in self.scale_fns])
        return self.stoich_matrix @ (theta[self.param_idx_arr] * scale * mass)
    return call

BASE = "/projects/anth4580/Bayesian/Results/Chain Scaling Tests"
for run in sys.argv[1:]:
    sp = f"{BASE}/{run}/solver_params.json"
    post = az.from_netcdf(f"{BASE}/{run}/posterior_samples_pm.nc").posterior
    nch, ndr = post.sizes["chain"], post.sizes["draw"]
    picks = [(c % nch, int(d)) for c, d in zip(range(10), np.linspace(0, ndr - 1, 10))]
    lps = {}
    for floor in (True, False):
        rmb.ReactionNetwork.__call__ = make_call(floor)
        model = ir._build_model_bundle(ir.import_solver_params(sp)).pm_model
        f = model.compile_logp()
        vals = []
        for c, d in picks:
            pt = {}
            for vv in model.value_vars:
                n = vv.name
                if n.endswith("_log__"):
                    pt[n] = float(np.log(post[n[:-6]].values[c, d]))
                elif "_x" in n and n.endswith("__"):
                    base, fac = n[:-2].rsplit("_x", 1)
                    pt[n] = float(post[base].values[c, d]) * float(fac)
                else:
                    pt[n] = float(post[n].values[c, d])
            vals.append(float(f(pt)))
        lps[floor] = np.array(vals)
    diff = lps[False] - lps[True]
    print(f"{run}: logp (floor) {lps[True].min():.3f}..{lps[True].max():.3f}; "
          f"no-floor minus floor: max |diff| {np.abs(diff).max():.3e} nats, mean {diff.mean():.3e}", flush=True)
print("DONE")
