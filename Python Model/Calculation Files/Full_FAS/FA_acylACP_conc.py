"""Fatty-acid observables plus the measurable acyl-ACP pathway intermediates.

Everything FA_conc.py provides (per-species fatty acid, C16 equivalents, initial rate,
mole fractions), plus the ACP-bound intermediates of each elongation cycle, per chain length
n and saturation. These are the data type for Fig 9's "individual species + intermediates"
cells. A config names one calculation module, so the fatty-acid observables come along rather
than living in a second module. Which intermediates are observables is set by MEASURED_FORMS
below; edit it to change the set.

  "C<n>_KetoacylACP (uM)"      beta-ketoacyl-ACP (FabH, FabB, FabF product), free plus FabG-NADPH-bound
  "C<n>_HydroxyacylACP (uM)"   beta-hydroxyacyl-ACP (FabG product), free plus FabA- and FabZ-bound
  "C<n>_EnoylACP (uM)"         enoyl-ACP (FabA, FabZ product), free plus FabA-, FabZ- and FabI-NADH-bound
  "C<n>_AcylACP (uM)"          acyl-ACP (FabI product), n >= 4, free plus bound to FabB, FabF, TesA,
                               and the acyl-ACP half of acetyl-FabH complexes (C<n+2>_FabH_Act_AcACP)
  ... and "_unsat" versions of each on the unsaturated branch.

cis-3-decenoyl-ACP, the branch point into unsaturated chains, is C10_EnoylACP_unsat. By mass it
is indistinguishable from trans-2-decenoyl-ACP (C10_EnoylACP), so an assay that cannot separate
them measures the sum of the two. Acetyl-ACP (C2) comes from initiation, not FabI, and is not an
AcylACP observable.

Enzyme-bound forms are included because they are non-covalent complexes a quenched
measurement releases as free acyl-ACP.

classify_acp_species assigns every species that contains ACP to what its ACP carries, including
the forms not exposed here (beta-ketoacyl-, acyl- and malonyl-ACP, and empty ACP). An ACP
species of an unknown form raises, so a new complex type cannot drop out of the totals silently.
"""
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import re

import jax.numpy as jnp

from experiment_framework import ObservableDefinition

# The fatty-acid observables, loaded from the sibling module by path: calculation modules are
# loaded by file, not imported as a package.
_spec = spec_from_file_location("FA_conc", Path(__file__).with_name("FA_conc.py"))
fa_conc = module_from_spec(_spec)
_spec.loader.exec_module(fa_conc)

# The intermediates exposed as observables: form code in the species name -> observable stem.
# Edit this to change the measured set; a form left out is still classified, just not exposed.
MEASURED_FORMS = {
    "BKeAc": "KetoacylACP",
    "BHyAc": "HydroxyacylACP",
    "EnAc": "EnoylACP",
    "cis3EnAc": "EnoylACP",
    "Ac": "AcylACP",
}
# Shortest chain exposed per form: acetyl-ACP (C2) is an initiation species, not a FabI product.
MIN_CHAIN = {"Ac": 4}

_ENZ = r"(?:Fab[A-Z]|TesA)"
# A chain on the ACP itself, free or enzyme-bound (with or without a cofactor).
_ACYL = re.compile(rf"^C(\d+)_(?:{_ENZ}(?:_NADPH|_NADH)?_)?(BKeAc|BHyAc|EnAc|cis3EnAc|Ac)ACP(_unsat)?$")
# Acetyl-FabH holding an acyl-ACP: C<n+2>_FabH_Act_AcACP carries a C<n> acyl-ACP.
_FABH_ACT_ACYL = re.compile(r"^C(\d+)_FabH_Act_AcACP(_unsat)?$")
# Malonyl on the ACP: free or bound, or opposite an acyl-enzyme (C<m>_FabX_Act_MalACP[_unsat|_cis3]).
_MALONYL = re.compile(rf"^C3_(?:{_ENZ}_)?MalACP$|^C\d+_{_ENZ}_Act_MalACP(?:_unsat|_cis3)?$")
# ACP with nothing on it.
_EMPTY = re.compile(rf"^ACP$|^{_ENZ}_ACP$|^C\d+_{_ENZ}_Act_ACP$")


def classify_acp_species(species_name):
    """What an ACP-containing species carries on its ACP.

    Returns ("acyl", n, unsaturated, form) with form one of BKeAc/BHyAc/EnAc/cis3EnAc/Ac,
    ("malonyl",) or ("empty",); None for species without ACP. Raises on an ACP species of a
    form this module does not know.
    """
    if "ACP" not in species_name:
        return None
    match = _ACYL.fullmatch(species_name)
    if match:
        n, form, unsat = match.groups()
        return ("acyl", int(n), bool(unsat) or form == "cis3EnAc", form)
    match = _FABH_ACT_ACYL.fullmatch(species_name)
    if match:
        return ("acyl", int(match.group(1)) - 2, bool(match.group(2)), "Ac")
    if _MALONYL.fullmatch(species_name):
        return ("malonyl",)
    if _EMPTY.fullmatch(species_name):
        return ("empty",)
    raise ValueError(f"{species_name!r} contains ACP in a form FA_acylACP_conc.py does not classify.")


def intermediate_groups(species_names):
    """{observable name: member species} for every intermediate form in MEASURED_FORMS."""
    groups = {}
    for name in species_names:
        kind = classify_acp_species(name)
        if kind is None or kind[0] != "acyl" or kind[3] not in MEASURED_FORMS:
            continue
        _, n, unsat, form = kind
        if n < MIN_CHAIN.get(form, 0):
            continue
        label = f"C{n}_{MEASURED_FORMS[form]}{'_unsat' if unsat else ''} (uM)"
        groups.setdefault(label, []).append(name)
    return groups


def make_sum_observable(output_name, members):
    members = tuple(members)

    def compute(times, concentrations, species_index, **_):
        idx = jnp.asarray([species_index[name] for name in members])
        return {output_name: concentrations[:, idx].sum(axis=1)}

    return ObservableDefinition(
        name=output_name,
        compute=compute,
        output_names=(output_name,),
        required_species=members,
        description=f"Sum of {len(members)} ACP-bound species: {', '.join(members)}.",
    )


def build_observables(species_names):
    observables = fa_conc.build_observables(species_names)
    observables.update({
        label: make_sum_observable(label, members)
        for label, members in intermediate_groups(species_names).items()
    })
    return observables


OBSERVABLES = {}
