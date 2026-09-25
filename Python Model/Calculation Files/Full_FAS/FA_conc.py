from experiment_framework import ObservableDefinition
import re

import jax.numpy as jnp


SELECTED_SPECIES = ()

SPECIES_PATTERN = r"^C(\d+)_FA(_unsat)?$"

C16_EQUIV_NAME = "C16 Equivalents (uM)"
INITIAL_RATE_NAME = "Initial Rate (uM C16 Equivalents/min)"
SEC_PER_MIN = 60.0
MOLE_FRACTION_SUFFIX = " (MF)"
# Mole fractions and rates both divide by a quantity that is exactly zero at t=0, before any
# fatty acid exists. Clamping the denominator to a tiny floor is NOT enough: it leaves the
# forward value correct but makes d(x/y)/dx = 1/y astronomically large and y**2 underflow in
# the reverse pass, so the gradient comes back non-finite (verified directly). The safe
# pattern is two wheres -- one choosing a harmless denominator so the division never sees a
# zero, one selecting the result -- which keeps both the value and its gradient finite.
_DIV_EPS = 1e-12


def _safe_divide(numerator, denominator):
    """numerator / denominator, defined as 0 where the denominator vanishes.

    Both branches must be finite for reverse-mode AD: the division is evaluated with a
    substituted denominator of 1 wherever the real one is ~0, so no NaN is ever created and
    then masked. A masked NaN still poisons the backward pass.
    """
    ok = denominator > _DIV_EPS
    safe = jnp.where(ok, denominator, 1.0)
    return jnp.where(ok, numerator / safe, 0.0)


def c16_equiv_weight(species_name):
    """Carbon weight of one fatty acid in C16 equivalents: n/16 for a C{n} chain,
    saturated or unsaturated alike (the double bond does not change the carbon count)."""
    match = re.fullmatch(SPECIES_PATTERN, species_name)
    if match is None:
        raise ValueError(f"{species_name!r} is not a fatty-acid species ({SPECIES_PATTERN}).")
    return int(match.group(1)) / 16.0


def make_observable(species_name):
    output_name = f"{species_name} (uM)"

    def compute(times, concentrations, species_index, **_):
        return {output_name: concentrations[:, species_index[species_name]]}

    return ObservableDefinition(
        name=output_name,
        compute=compute,
        output_names=(output_name,),
        required_species=(species_name,),
        description=f"Concentration of {species_name}.",
    )


def make_c16_equiv_observable(fa_species):
    """Total fatty acid in C16 equivalents: sum over every FA species of (n/16)*[C{n}_FA].

    Not the plain sum: one C8 carries half the carbon of one C16, so a total-FA
    measurement calibrated against a C16 standard reports it as half a C16.
    """
    fa_species = tuple(fa_species)
    weights = jnp.asarray([c16_equiv_weight(name) for name in fa_species])

    def compute(times, concentrations, species_index, **_):
        idx = jnp.asarray([species_index[name] for name in fa_species])
        return {C16_EQUIV_NAME: concentrations[:, idx] @ weights}

    return ObservableDefinition(
        name=C16_EQUIV_NAME,
        compute=compute,
        output_names=(C16_EQUIV_NAME,),
        required_species=fa_species,
        description="Total fatty acid in C16 equivalents, sum of (n/16)*[C{n}_FA].",
    )


def make_mole_fraction_observable(species_name, fa_species):
    """One fatty acid as a fraction of total fatty acid, uM species / uM total FA.

    This is what the experimental profile reports: the assay measures the distribution
    across chain lengths, not absolute titre, so the data are dimensionless fractions
    summing to 1 across species. Note the denominator is the plain sum of FA
    concentrations, NOT the C16-equivalent sum -- a mole fraction counts molecules, so a
    C8 and a C16 each contribute one, while C16 equivalents weight them by carbon.
    """
    fa_species = tuple(fa_species)
    output_name = f"{species_name}{MOLE_FRACTION_SUFFIX}"

    def compute(times, concentrations, species_index, **_):
        idx = jnp.asarray([species_index[name] for name in fa_species])
        block = concentrations[:, idx]
        total = block.sum(axis=1)
        return {output_name: _safe_divide(concentrations[:, species_index[species_name]], total)}

    return ObservableDefinition(
        name=output_name,
        compute=compute,
        output_names=(output_name,),
        required_species=fa_species,
        description=f"Mole fraction of {species_name} among all fatty acids.",
    )


def make_initial_rate_observable(fa_species):
    """Average rate of fatty-acid production since t=0, in uM C16 equivalents per minute.

    Per minute because that is how the ME1 kinetics data report it (uM C16/min); model time
    is in seconds, so the secant is converted here rather than in every dataset.

    The experimental "initial rate" is not an instantaneous derivative: it is the C16
    equivalents accumulated by a fixed early time divided by that time (150 s in the
    current dataset). Since there is no fatty acid at t=0, that secant is exactly
    C16Equiv(t)/t, so evaluating this observable at the assay's own time reproduces the
    measured quantity with no window constant to keep in sync -- ask for it at t=150 and
    it is the 150 s initial rate, at t=300 the 300 s one.
    """
    fa_species = tuple(fa_species)
    weights = jnp.asarray([c16_equiv_weight(name) for name in fa_species])

    def compute(times, concentrations, species_index, **_):
        idx = jnp.asarray([species_index[name] for name in fa_species])
        c16_equiv = concentrations[:, idx] @ weights
        return {INITIAL_RATE_NAME: SEC_PER_MIN * _safe_divide(c16_equiv, jnp.asarray(times))}

    return ObservableDefinition(
        name=INITIAL_RATE_NAME,
        compute=compute,
        output_names=(INITIAL_RATE_NAME,),
        required_species=fa_species,
        description="C16 equivalents accumulated since t=0, divided by t, per minute.",
    )


def choose_species(species_names):
    if SPECIES_PATTERN:
        pattern = re.compile(SPECIES_PATTERN)
        return [species_name for species_name in species_names if pattern.fullmatch(species_name)]

    missing_species = [species_name for species_name in SELECTED_SPECIES if species_name not in species_names]
    if missing_species:
        raise ValueError(f"Requested species not found in reaction network: {missing_species}")

    return list(SELECTED_SPECIES)


def build_observables(species_names):
    chosen = choose_species(species_names)
    observables = {
        f"{species_name} (uM)": make_observable(species_name)
        for species_name in chosen
    }
    if chosen:
        observables[C16_EQUIV_NAME] = make_c16_equiv_observable(chosen)
        observables[INITIAL_RATE_NAME] = make_initial_rate_observable(chosen)
        # The experimental profile is reported as mole fractions, one column per species.
        observables.update({
            f"{species_name}{MOLE_FRACTION_SUFFIX}": make_mole_fraction_observable(species_name, chosen)
            for species_name in chosen
        })
    return observables


OBSERVABLES = {}
