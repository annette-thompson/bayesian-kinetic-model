from experiment_framework import ObservableDefinition
import re

SELECTED_SPECIES = ()

# Full saturated fatty-acid product profile: every C<n>_FA species this
# network can produce (C4_FA and C6_FA at minimum, since FabF unlocks the
# second elongation cycle). No unsaturated branch yet -- FabA isn't loaded
# in this network, so no C<n>_FA_unsat species even exist to match.
SPECIES_PATTERN = r"^C(\d+)_FA$"


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


def choose_species(species_names):
    if SPECIES_PATTERN:
        pattern = re.compile(SPECIES_PATTERN)
        return [species_name for species_name in species_names if pattern.fullmatch(species_name)]

    missing_species = [species_name for species_name in SELECTED_SPECIES if species_name not in species_names]
    if missing_species:
        raise ValueError(f"Requested species not found in reaction network: {missing_species}")

    return list(SELECTED_SPECIES)


def build_observables(species_names):
    return {
        f"{species_name} (uM)": make_observable(species_name)
        for species_name in choose_species(species_names)
    }


OBSERVABLES = {}
