'''Utilities for compiling reaction network parameters from elementary reactions'''

__author__ = 'Timotej Bernat'

import logging
LOGGER = logging.getLogger(__name__)

from typing import Sequence
from collections import defaultdict

import numpy as np

from .containers import ElementaryReaction, StoichBalanceTerms


def compile_reaction_network(rxns : Sequence[ElementaryReaction]) -> tuple[dict[str, float], dict[str, StoichBalanceTerms], dict[int, str]]:
    '''
    Collect unique reactants and rate constants among a collection of reactions
    Returns a dict of rate const key-value pairs, a dict of stoichiometric contributions keyed by species,
    and a dict of unique indices for each species
    '''
    rate_consts : dict[str, float] = {}
    contributing_terms  = defaultdict(StoichBalanceTerms)
    
    SPECIES_BY_CONTRIB : dict[str, str] = {
        'generation' :  'products',  # species is considered "generated" if appearing in the products...
        'consumption' : 'reactants', # ...and "consumed if it appears as one of the products"
    }
    for rxn in rxns:
        # register rate constant
        if rxn.rate_const_key in rate_consts:
            raise KeyError(f'Duplicate rate constant "{rxn.rate_const_key}={rxn.rate_const_value}" defined in reaction: {str(rxn)}')
        rate_consts[rxn.rate_const_key] = rxn.rate_const_value # TODO: scale this by scaling group, if requested?
        
        # register participating species
        for balance_type, species_list_type in SPECIES_BY_CONTRIB.items():
            for species in getattr(rxn, species_list_type):
                getattr(contributing_terms[species], balance_type).add(rxn)
    contributing_terms = dict(contributing_terms) # convert to vanilla dict for simplicity after collecting all reactants

    return rate_consts, contributing_terms

def compute_rate_const_tensors(contributing_terms : dict[str, StoichBalanceTerms], idxs_by_species : dict[str, int], scaling_groups : dict[int, float]) -> dict[int, np.ndarray[float]]:
    '''Generate tensors of '''
    n_species = len(idxs_by_species)
    rate_const_tensors_by_order = {
        1 : np.zeros((n_species, n_species), dtype=float),
        2 : np.zeros((n_species, n_species, n_species), dtype=float), # NOTE: for now, do not support any reactions beyond 1st and 2nd order
    }

    for species, sbt in contributing_terms.items():
        LOGGER.info(f'{species} : {sbt.rate_expression}')
        curr_spec_idx = idxs_by_species[species]

        for sign, rxn in sbt.signed_rxns:
            order = rxn.order
            reactant_idxs = [idxs_by_species[s] for s in rxn.reactants] # index of each species (corresponds to the rate of change of conc of this species)
            if (rate_const_tensor := rate_const_tensors_by_order.get(order)) is None:
                LOGGER.warn(f'Reactions of {order=} are currently unsupported, will be skipped when building system of rate equations')
            
            scale_factor = scaling_groups.get(rxn.scaling_group_id, 1.0) # default to scale factor of 1.0 (i.e. no scaling) if no scale factor group is assigned
            rate_const_tensor[curr_spec_idx, *reactant_idxs] = sign * scale_factor * rxn.rate_const_value

    return rate_const_tensors_by_order