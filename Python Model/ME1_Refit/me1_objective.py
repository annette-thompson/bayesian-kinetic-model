"""The ME1 fitting objective, translated from Combined_Pathway_Handler.m.

Three terms, combined as a PRODUCT (the MATLAB's `total_obj = obj1*obj23(1)*obj23(2)`):

  obj1  seven initial rates (C16 equivalents over the 150 s window, uM/min), SSE
  obj2  the reference time course to 720 s, sum of squared residuals
  obj3  the fatty-acid chain-length profile at 720 s, sum of squared residuals

Two properties of the objective are worth knowing before reading a refit, since each
affects where the optimizer goes.

1. obj3 CANNOT SEE TITRE. It compares the model profile against `total_FA * fractions`,
   where `total_FA` is the model's OWN total. It constrains the shape of the distribution;
   total production is constrained by obj1 and obj2 only.

2. THE THREE TERMS ARE MULTIPLIED (the MATLAB's `total_obj = obj1*obj23(1)*obj23(2)`), so
   the relative weighting is whatever the terms happen to be in their own units.
   `combine="product"` is the original; `"sum"` and `"log_sum"` are offered because they are
   better behaved, not because they reproduce the paper.

The 1e8 sentinel returned for out-of-range parameters is a real constraint, not error
handling: fminsearch is unconstrained, so that value is the only thing keeping the search
inside the physical region.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

import me1_config as cfg

@dataclass
class Objective:
    total: float
    obj1: float
    obj2: float
    obj3: float
    rejected: str | None = None
    model_rates: np.ndarray = field(default_factory=lambda: np.array([]))
    model_timecourse: np.ndarray = field(default_factory=lambda: np.array([]))
    model_profile: np.ndarray = field(default_factory=lambda: np.array([]))


def load_data(data_dir=None):
    d = data_dir or cfg.DATA
    rates = pd.read_csv(d / "initial_rates.csv")
    tc = pd.read_csv(d / "timecourse.csv")
    prof = pd.read_csv(d / "profile_fractions.csv")
    return rates, tc, prof


def check_parameters(p_abs: dict, model=None) -> str | None:
    """Return a reason string if this p_vec is out of range, else None."""
    probe = {k: (abs(v) if k in cfg.ADDITIVE else v) for k, v in p_abs.items()}
    for name, value in probe.items():
        if value <= 0:
            return f"{name} <= 0"
    for name, hi in cfg.UPPER_BOUNDS.items():
        if probe[name] > hi:
            return f"{name} > {hi:g}"
    for name, lo in cfg.LOWER_BOUNDS.items():
        if probe[name] < lo:
            return f"{name} < {lo:g}"
    if model is not None:
        # Association rates must stay under the diffusion limit, and no Kd may be tighter
        # than 0.01 uM. The MATLAB checked named parameters; here the same limits are
        # applied through the crosswalk to the equivalent reaction constants.
        rc = model.rate_constants(p_abs)
        for key, vals in rc.items():
            if not key.startswith("kon") and "_kon_" not in key:
                continue
            hi = 1650.0 if any(t in key for t in ("_D_", "_H_", "_G_NADPH", "_I_NADH")) else 629.0
            if max(vals) > hi:
                return f"{key} = {max(vals):.4g} exceeds the {hi:g} association limit"
    return None


def evaluate(model, p_abs: dict, data=None, combine="product",
             enforce_bounds=True, check_rate_constants=False,
             rate_conditions=None, rate_measured=None) -> Objective:
    """Objective at one p_vec. `model` is an me1_model.ME1Model.

    `rate_conditions` / `rate_measured` swap out which initial-rate dataset obj1 scores
    against, leaving obj2 and obj3 alone. Passing the held-out set answers "what would
    these parameters be if they had been fitted to D instead of A" without building a
    second objective function.
    """
    rates, tc, prof = data if data is not None else load_data()

    if enforce_bounds:
        why = check_parameters(p_abs, model if check_rate_constants else None)
        if why:
            return Objective(cfg.REJECT_OBJECTIVE, np.nan, np.nan, np.nan, rejected=why)

    # ---- obj1: seven initial rates -------------------------------------------------
    conditions = rate_conditions if rate_conditions is not None else cfg.RATE_CONDITIONS
    model_rates = model.initial_rates_c16(p_abs, conditions)
    if not np.all(np.isfinite(model_rates)):
        return Objective(cfg.REJECT_OBJECTIVE, np.nan, np.nan, np.nan,
                         rejected="a rate condition failed to solve")
    # The model gives C16 equivalents accumulated over the assay window; the measurements
    # are per minute, so divide by the window in minutes.
    predicted = model_rates / (cfg.RATE_WINDOW_S / 60.0)
    measured = (np.asarray(rate_measured) if rate_measured is not None
                else rates["measured_rate_uM_C16_per_min"].to_numpy())
    obj1 = float(np.sum((measured - predicted) ** 2))

    # ---- obj2 + obj3: one reference solve to 720 s ----------------------------------
    ref = cfg.RATE_CONDITIONS[0]
    t_data = tc["time_min"].to_numpy() * 60.0
    # Solved at the measurement times directly. The MATLAB splined the adaptive solver
    # output instead; evaluating the solution where the data actually is removes that
    # interpolation error rather than reproducing it.
    times = np.unique(np.concatenate([t_data, [cfg.ENDPOINT_S]]))
    sol = model.solve(p_abs, model.condition_y0(ref), times)
    if not sol.ok:
        return Objective(cfg.REJECT_OBJECTIVE, obj1, np.nan, np.nan,
                         rejected="the reference solve failed")
    at = {float(t): i for i, t in enumerate(times)}
    pred_tc = np.array([sol.c16_equivalents[at[float(t)]] for t in t_data])
    obj2 = float(np.sum((pred_tc - tc["c16_equivalents_uM"].to_numpy()) ** 2))

    end = sol.profile[at[float(cfg.ENDPOINT_S)]]
    target = end.sum() * prof["mole_fraction"].to_numpy()
    obj3 = float(np.sum((end - target) ** 2))

    if combine == "product":
        total = obj1 * obj2 * obj3
    elif combine == "sum":
        total = obj1 + obj2 + obj3
    elif combine == "log_sum":
        total = float(np.sum(np.log([max(o, 1e-300) for o in (obj1, obj2, obj3)])))
    else:
        raise ValueError(f"unknown combine={combine!r}")
    return Objective(total, obj1, obj2, obj3, None, model_rates, pred_tc, end)


def make_scalar_objective(model, fitted, p_start, data=None, log_space=True, **kw):
    """A f(x) -> float for scipy.optimize, varying only `fitted`.

    log_space fits log10 of each multiplicative parameter, which the MATLAB did not do.
    The published values span eight orders of magnitude (0.0054 to 142,474), and Nelder-Mead
    takes steps in absolute units, so a simplex that moves c3 usefully cannot move a1 at all.
    Additive parameters (d1, d2) are always fitted directly.
    """
    data = data if data is not None else load_data()
    history = []

    def unpack(x):
        p = dict(p_start)
        for name, value in zip(fitted, np.atleast_1d(x)):
            if log_space and name not in cfg.ADDITIVE:
                p[name] = float(10.0 ** value)
            else:
                p[name] = float(value)
        return p

    def f(x):
        p = unpack(x)
        res = evaluate(model, p, data=data, **kw)
        history.append({"total": res.total, "obj1": res.obj1, "obj2": res.obj2,
                        "obj3": res.obj3, "rejected": res.rejected,
                        **{n: p[n] for n in fitted}})
        return res.total

    def pack(p):
        return np.array([np.log10(p[n]) if (log_space and n not in cfg.ADDITIVE) else p[n]
                         for n in fitted], dtype=float)

    f.unpack, f.pack, f.history = unpack, pack, history
    return f
