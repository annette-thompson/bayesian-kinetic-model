import jax.numpy as jnp

from experiment_framework import ObservableDefinition


def product_fraction(times, concentrations, species_index):
    product = concentrations[:, species_index["P"]]
    substrate = concentrations[:, species_index["S"]]
    enzyme_substrate = concentrations[:, species_index["ES"]]
    total_substrate_pool = product + substrate + enzyme_substrate
    return {"product_fraction": product / total_substrate_pool}


def product_concentration(times, concentrations, species_index):
    return {"product_concentration": concentrations[:, species_index["P"]]}


def total_substrate_pool(times, concentrations, species_index):
    product = concentrations[:, species_index["P"]]
    substrate = concentrations[:, species_index["S"]]
    enzyme_substrate = concentrations[:, species_index["ES"]]
    return {"total_substrate_pool": product + substrate + enzyme_substrate}


def final_product_concentration(times, concentrations, species_index):
    return {"final_product_concentration": concentrations[:, species_index["P"]]}


OBSERVABLES = {
    "product_fraction": ObservableDefinition(
        name="product_fraction",
        compute=product_fraction,
        output_names=("product_fraction",),
        required_species=("P", "S", "ES"),
        description="Fraction of total substrate pool converted to product: P / (S + ES + P).",
    ),
    "product_concentration": ObservableDefinition(
        name="product_concentration",
        compute=product_concentration,
        output_names=("product_concentration",),
        required_species=("P",),
        description="Direct product concentration trace.",
    ),
    "total_substrate_pool": ObservableDefinition(
        name="total_substrate_pool",
        compute=total_substrate_pool,
        output_names=("total_substrate_pool",),
        required_species=("P", "S", "ES"),
        description="Total substrate-containing pool S + ES + P.",
    ),
    "final_product_concentration": ObservableDefinition(
        name="final_product_concentration",
        compute=final_product_concentration,
        output_names=("final_product_concentration",),
        required_species=("P",),
        description="Product concentration evaluated at per-row requested times.",
    ),
}