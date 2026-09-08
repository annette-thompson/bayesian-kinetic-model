from experiment_framework import ObservableDefinition

def final_MalACP(times, concentrations, species_index, **_):
    return {"MalACP (uM)": concentrations[:, species_index["C3_MalACP"]]}

OBSERVABLES = {
    "MalACP (uM)": ObservableDefinition(
        name="MalACP (uM)",
        compute=final_MalACP,
        output_names=("MalACP (uM)",),
        required_species=("C3_MalACP",),
        description="Final Malonyl-ACP concentration.",
    )
}
