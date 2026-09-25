"""Everything about the ME1 refit that is a choice rather than a computation.

The MATLAB original spread these values across four files, mostly as unlabelled numeric
literals inside the objective function (`p_vec(4) > 6.29E4`, `enz_conc_HB = [0 1 0 1 1 1 10
1 1 0]`, a bare `rate_exp = [...]`). They are gathered here, named, and given the units and
provenance that made them hard to read there.

The model itself is NOT re-implemented. `Reactions/EC_FAS_ME1/C20+unsat+FBinit` was verified
reaction-for-reaction against the MATLAB ODEs (588 directed steps, all rate constants equal),
and its YAML already declares every scaling parameter -- including the composite expressions
`(1/b3)`, `(b1/b2)`, `(f*c2)` and the TesA free-energy terms `1/exp(n*d1+d2)` -- so the whole
of `param_func.m` is reproduced by setting 18 numbers and letting the builder recompute the
rate constants.
"""
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
REACTIONS = PROJECT / "Reactions" / "EC_FAS_ME1" / "C20+unsat+FBinit"
DATA = HERE / "data"

# ----------------------------------------------------------------------------------
# The 18 scaling parameters
# ----------------------------------------------------------------------------------
# Order is the MATLAB p_vec order, kept so published values can be pasted in directly.
PARAM_NAMES = ["a1", "a2", "a3", "b1", "c2", "c3", "c1", "b2",
               "b3", "d1", "d2", "e", "f", "c4", "x1", "x2", "x3", "x4"]

PUBLISHED = {
    "a1": 142473.7238,     "a2": 7597.676912,   "a3": 4.276689943,
    "b1": 40213.92919,     "c2": 88.88525384,   "c3": 0.005388274,
    "c1": 4.645634978,     "b2": 0.006677519,   "b3": 0.284982219,
    "d1": -0.285700283,    "d2": 3.348915642,   "e": 2.886607673,
    "f": 132.8499358,      "c4": 2180.050007,   "x1": 0.539756276,
    "x2": 0.053673263,     "x3": 34.49718991,   "x4": 11.15058888,
}

WHAT_EACH_DOES = {
    "a1": "initiation binding (FabD, FabH)",
    "a2": "elongation binding (FabF, FabB, and the shared acyl-ACP steps)",
    "a3": "termination binding (TesA)",
    "b1": "Kd fit, used as (b1/b2) and (b1/b3)",
    "b2": "Kd fit, appears only as (b1/b2)",
    "b3": "Kd fit, appears as (1/b3) and (b1/b3)",
    "c1": "initiation kcat (FabD, FabH)",
    "c2": "elongation kcat (FabG, FabI, FabF, FabA, FabB)",
    "c3": "termination kcat (TesA)",
    "c4": "FabZ / FabA dehydratase kcat",
    "d1": "TesA linear free-energy SLOPE, additive inside exp(n*d1 + d2)",
    "d2": "TesA linear free-energy INTERCEPT, additive inside exp(n*d1 + d2)",
    "e":  "acyl-ACP / holo-ACP binding strength (enzyme sequestration by ACP)",
    "f":  "FabA specificity for the unsaturated branch",
    "x1": "FabF decarboxylation kcat (malonyl-ACP -> acetyl-ACP)",
    "x2": "FabB decarboxylation kcat",
    "x3": "FabF acetyl-CoA binding in the FabH-like route",
    "x4": "FabB acetyl-CoA binding in the FabH-like route",
}

# d1 and d2 enter additively inside an exponential, so their no-op offset is 0 and a refit
# value is published + offset. Every other parameter is multiplicative: no-op 1, refit value
# published x multiplier. Getting this backwards silently rescales TesA by ~e^3.3.
ADDITIVE = {"d1", "d2"}

# Which parameters Combined_Pathway_Optimizer.m actually varied. Everything else was pinned
# at the published value inside the anonymous fitfunc, which is why that line was 300
# characters of literals.
MATLAB_FITTED = ["f", "x1", "x2", "x3", "x4"]

# ----------------------------------------------------------------------------------
# Guard rails, transcribed from Combined_Pathway_Handler.m
# ----------------------------------------------------------------------------------
# The original returned a sentinel objective of 1e8 when a proposal left these ranges. Since
# fminsearch is unconstrained, that sentinel IS the constraint -- it is not error handling,
# and removing it changes the fit. d1/d2 are tested on their absolute value because their
# sign is meaningful.
REJECT_OBJECTIVE = 1e8
UPPER_BOUNDS = {"b1": 6.29e4, "c2": 240.0, "c3": 1.0, "c1": 15.0}
LOWER_BOUNDS = {"a1": 1e-3, "a2": 1e-3, "a3": 1e-3, "c2": 1e-2,
                "c3": 7.41e-5, "c1": 1e-3, "b2": 2.46e-7, "b3": 2.46e-7}

# Association-rate ceilings (uM^-1 s^-1): a kon above the diffusion limit is unphysical.
# The MATLAB applied these to named parameters with two different limits; kept as-is.
KON_CEILING = {"k2_1f": 1650.0, "k2_3f": 1650.0, "k3_1f": 1650.0,
               "k4_1f": 1650.0, "k6_1f": 1650.0,
               "k3_3f": 629.0, "k4_2f": 629.0, "k5_1f": 629.0,
               "k6_2f": 629.0, "k8_1f": 629.0, "k8_3f": 629.0, "k7_1f": 629.0}
# ...and the matching koff/kon floor, i.e. no Kd tighter than 0.01 uM.
MIN_KOFF_OVER_KON = 0.01

# ----------------------------------------------------------------------------------
# Experimental conditions
# ----------------------------------------------------------------------------------
# uM. Held at one substrate composition for every dataset, deliberately.
#
# The model is fitted to a mix of in vitro and in vivo data: the initial rates (A) and the
# time course (C) are in vitro, while the chain-length profile (B) comes from an E. coli
# strain overexpressing TesA (Grisewood et al. 2017) -- in vivo. 0.5 mM malonyl-CoA /
# 0.5 mM acetyl-CoA / 1 mM NADPH / 1 mM NADH is the composition reported to best represent
# the in vivo cytosol, so it is used throughout rather than matching each in vitro assay's
# own buffer. In vivo the only lever is expression level, so ENZYME concentrations are what
# vary between conditions and substrates stay fixed; holding them constant is what makes a
# single parameter set answerable to both kinds of data.
#
# The Figure S1 caption records both compositions: modelled A-D at the values above, and
# the in vitro reconstitutions in A and D at 1.3 mM NADPH / 0.5 mM malonyl-CoA /
# 0.1 mM acetyl-CoA. Changing these to match an individual assay would break the shared
# basis, not improve the fit.
BASE_SUBSTRATES = {"C2_AcCoA": 500.0, "ACP": 10.0, "NADPH": 1000.0,
                   "NADH": 1000.0, "C3_MalCoA": 500.0}
BASE_ENZYMES = {"FabD": 1.0, "FabH": 1.0, "FabG": 1.0, "FabZ": 1.0, "FabI": 1.0,
                "TesA": 10.0, "FabF": 1.0, "FabA": 1.0, "FabB": 1.0}

RATE_WINDOW_S = 150.0      # the initial-rate assay window: the caption's "2.5 minutes"
ENDPOINT_S = 720.0         # product profile and time course: the caption's "12 minutes"

# The seven initial-rate conditions, in the order rate_exp was written in the MATLAB.
# "drop" lists enzymes set to zero. Acetyl-CoA is inherited from BASE_SUBSTRATES unless the
# condition sets it explicitly -- only the three "no AcCoA" rows do, at 0. That keeps
# BASE_SUBSTRATES["C2_AcCoA"] the single place to change the acetyl-CoA level for every
# condition at once; repeating the literal here would silently leave the fitted panel
# behind when you changed the base.
RATE_CONDITIONS = [
    {"label": "reference",                     "drop": []},
    {"label": "no FabH",                       "drop": ["FabH"]},
    {"label": "no FabH, no FabF",              "drop": ["FabH", "FabF"]},
    {"label": "no FabH, no FabB",              "drop": ["FabH", "FabB"]},
    {"label": "no FabH, no AcCoA",             "drop": ["FabH"],          "acetyl_coa": 0.0},
    {"label": "no FabH, no FabF, no AcCoA",    "drop": ["FabH", "FabF"],  "acetyl_coa": 0.0},
    {"label": "no FabH, no FabB, no AcCoA",    "drop": ["FabH", "FabB"],  "acetyl_coa": 0.0},
]

# Held-out validation: initial rates the model was NOT fitted on (Figure S1, "New Initial
# Rates"). Same buffer as the reference; each entry overrides enzyme concentrations rather
# than only zeroing them, so "30 uM FabH" and "no FabZ" are expressed the same way.
HELDOUT_CONDITIONS = [
    {"label": "reference",            "enzymes": {}},
    {"label": "30 uM FabH",           "enzymes": {"FabH": 30.0}},
    {"label": "10 uM FabF",           "enzymes": {"FabF": 10.0}},
    {"label": "10 uM FabI",           "enzymes": {"FabI": 10.0}},
    {"label": "no FabZ",              "enzymes": {"FabZ": 0.0}},
    {"label": "10 uM FabA + no FabZ", "enzymes": {"FabA": 10.0, "FabZ": 0.0}},
]

# A second held-out set (2025). These are the rates that motivated removing the FabH-
# independent initiation reactions: with FabH absent the measured rate is 0.43 +/- 0.43,
# i.e. indistinguishable from zero, where the older kinetics set had 2.42. A model that
# initiates without FabH cannot produce that.
#
# The experiment ran at 100 uM acetyl-CoA; the model is still solved at the standard
# 500 uM basis, so the comparison is model-as-parameterised against the new measurements.
HELDOUT_2025_CONDITIONS = [
    {"label": "reference",         "drop": []},
    {"label": "no FabH",           "drop": ["FabH"]},
    {"label": "no AcCoA",          "drop": [],        "acetyl_coa": 0.0},
    {"label": "no AcCoA, no FabH", "drop": ["FabH"],  "acetyl_coa": 0.0},
]

# Chain lengths in the order the MATLAB's F_raw / fit_dist vectors use: saturated and
# unsaturated interleaved, not grouped.
PROFILE_ORDER = [(4, False), (6, False), (8, False), (10, False),
                 (12, False), (12, True), (14, False), (14, True),
                 (16, False), (16, True), (18, False), (18, True),
                 (20, False), (20, True)]

# ----------------------------------------------------------------------------------
# Solver
# ----------------------------------------------------------------------------------
# The MATLAB used ode15s at RelTol 1e-6 (AbsTol at its 1e-6 default). These are the
# tolerances the rest of this project settled on;
RTOL, ATOL = 1e-5, 1e-7
MAX_STEPS = 100_000
