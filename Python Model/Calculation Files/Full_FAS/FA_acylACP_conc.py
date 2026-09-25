"""Fatty-acid observables plus acyl-ACP pathway intermediates.

Everything FA_conc.py provides (per-species fatty acid, C16 equivalents, initial rate,
mole fractions), plus the ACP-bound intermediates a quenched acyl-ACP measurement would
see. That is the data type Fig 9's "individual species + intermediates" cells need. A config
names one calculation module, so the fatty-acid observables come along rather than living
in a second module.

What an acyl-ACP measurement counts: every ACP thioester, whether free or held by an enzyme,
because quenching releases the complexes. So each observable sums the free intermediate and
every complex that carries the same thioester on its ACP:

  "C<n>_AcylACP (uM)", "C<n>_AcylACP_unsat (uM)"
      all ACP-bound chains of length n: beta-ketoacyl-, beta-hydroxyacyl-, enoyl- and acyl-ACP
      (and cis-3-decenoyl-ACP, counted as unsaturated C10), free, in enzyme complexes
      (C<n>_FabA_EnAcACP, C<n>_FabG_NADPH_BKeAcACP, ...), and as the acyl-ACP half of
      C<n+2>_FabH_Act_AcACP (acetyl-FabH holding a C<n> acyl-ACP)
  "Malonyl-ACP (uM)"
      free and enzyme-bound malonyl-ACP, including C<m>_Fab{B,F,H}_Act_MalACP, whose ACP
      carries malonyl while the acyl chain sits on the enzyme

ACP that carries nothing (ACP, <Enzyme>_ACP, C3_FabD_Act_ACP) is not an intermediate. Any
other species containing "ACP" raises, so a new complex type cannot be dropped silently.
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

MALONYL_ACP_NAME = "Malonyl-ACP (uM)"

_ENZ = r"(?:Fab[A-Z]|TesA)"
# An acyl chain on the ACP itself, free or enzyme-bound (with or without a cofactor).
_ACYL = re.compile(rf"^C(\d+)_(?:{_ENZ}(?:_NADPH|_NADH)?_)?(BKeAc|BHyAc|EnAc|cis3EnAc|Ac)ACP(_unsat)?$")
# Acetyl-FabH holding an acyl-ACP: C<n+2>_FabH_Act_AcACP carries a C<n> acyl-ACP.
_FABH_ACT_ACYL = re.compile(r"^C(\d+)_FabH_Act_AcACP(_unsat)?$")
# Malonyl on the ACP: free or bound, or opposite an acyl-enzyme (C<m>_FabX_Act_MalACP[_unsat|_cis3]).
_MALONYL = re.compile(rf"^C3_(?:{_ENZ}_)?MalACP$|^C\d+_{_ENZ}_Act_MalACP(?:_unsat|_cis3)?$")
# ACP with nothing on it.
_EMPTY = re.compile(rf"^ACP$|^{_ENZ}_ACP$|^C\d+_{_ENZ}_Act_ACP$")


def classify_acp_species(species_name):
    """What an ACP-containing species carries on its ACP.

    Returns ("acyl", n, unsaturated), ("malonyl",) or ("empty",); None for species without
    ACP. Raises on an ACP species of a form this module does not know.
    """
    if "ACP" not in species_name:
        return None
    match = _ACYL.fullmatch(species_name)
    if match:
        n, form, unsat = match.groups()
        return ("acyl", int(n), bool(unsat) or form == "cis3EnAc")
    match = _FABH_ACT_ACYL.fullmatch(species_name)
    if match:
        return ("acyl", int(match.group(1)) - 2, bool(match.group(2)))
    if _MALONYL.fullmatch(species_name):
        return ("malonyl",)
    if _EMPTY.fullmatch(species_name):
        return ("empty",)
    raise ValueError(f"{species_name!r} contains ACP in a form FA_acylACP_conc.py does not classify.")


def acyl_acp_groups(species_names):
    """{observable name: member species} for every acyl-ACP chain length and malonyl-ACP."""
    groups = {}
    for name in species_names:
        kind = classify_acp_species(name)
        if kind is None or kind[0] == "empty":
            continue
        if kind[0] == "malonyl":
            label = MALONYL_ACP_NAME
        else:
            _, n, unsat = kind
            label = f"C{n}_AcylACP{'_unsat' if unsat else ''} (uM)"
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
        for label, members in acyl_acp_groups(species_names).items()
    })
    return observables


OBSERVABLES = {}
