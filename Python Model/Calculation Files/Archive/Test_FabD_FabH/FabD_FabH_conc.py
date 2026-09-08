from experiment_framework import ObservableDefinition
import re

SELECTED_SPECIES = (
    "C4_BKeAcACP",
)

SPECIES_PATTERN = None


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
