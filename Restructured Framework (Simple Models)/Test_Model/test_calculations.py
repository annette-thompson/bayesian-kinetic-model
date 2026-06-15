import jax.numpy as jnp

from experiment_framework import ObservableDefinition


def product_fraction(times, concentrations, species_index, **_):
    product = concentrations[:, species_index["P"]]
    substrate = concentrations[:, species_index["S"]]
    enzyme_substrate = concentrations[:, species_index["ES"]]
    total_pool = jnp.clip(product + substrate + enzyme_substrate, 1.0e-12)
    return {"product_fraction": product / total_pool}


def product_concentration(times, concentrations, species_index, **_):
    return {"product_concentration": concentrations[:, species_index["P"]]}


def total_substrate_pool(times, concentrations, species_index, **_):
    product = concentrations[:, species_index["P"]]
    substrate = concentrations[:, species_index["S"]]
    enzyme_substrate = concentrations[:, species_index["ES"]]
    return {"total_substrate_pool": product + substrate + enzyme_substrate}


def total_product_pool(times, concentrations, species_index, **_):
    product = concentrations[:, species_index["P"]]
    side_product = concentrations[:, species_index["Q"]]
    return {"total_product_pool": product + side_product}


def final_product_concentration(times, concentrations, species_index, **_):
    return {"final_product_concentration": concentrations[:, species_index["P"]]}


def product_formation_rate(times, concentrations, species_index, **_):
    product = concentrations[:, species_index["P"]]
    # Use index-based gradient to stay stable when fixtures include repeated time values.
    rate = jnp.gradient(product)
    return {"product_formation_rate": rate}


OBSERVABLES = {
    "product_fraction": ObservableDefinition(
        name="product_fraction",
        compute=product_fraction,
        output_names=("product_fraction",),
        required_species=("P", "S", "ES"),
        description="Fractional product conversion P / (S + ES + P).",
    ),
    "product_concentration": ObservableDefinition(
        name="product_concentration",
        compute=product_concentration,
        output_names=("product_concentration",),
        required_species=("P",),
        description="Direct product concentration trajectory.",
    ),
    "total_substrate_pool": ObservableDefinition(
        name="total_substrate_pool",
        compute=total_substrate_pool,
        output_names=("total_substrate_pool",),
        required_species=("P", "S", "ES"),
        description="Conserved substrate-containing pool S + ES + P.",
    ),
    "total_product_pool": ObservableDefinition(
        name="total_product_pool",
        compute=total_product_pool,
        output_names=("total_product_pool",),
        required_species=("P", "Q"),
        description="Combined product pool P + Q for side-reaction testing.",
    ),
    "final_product_concentration": ObservableDefinition(
        name="final_product_concentration",
        compute=final_product_concentration,
        output_names=("final_product_concentration",),
        required_species=("P",),
        description="Endpoint product concentration.",
    ),
    "product_formation_rate": ObservableDefinition(
        name="product_formation_rate",
        compute=product_formation_rate,
        output_names=("product_formation_rate",),
        required_species=("P",),
        description="Numerical dP/dt for rate-dataset testing.",
    ),
}
