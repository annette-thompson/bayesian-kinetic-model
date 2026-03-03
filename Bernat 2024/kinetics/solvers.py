'''Methods for solving the systems of ODEs resulting from a reaction network'''

__author__ = 'Timotej Bernat'

from typing import TypeAlias, TypeVar
Shape : TypeAlias = tuple
N = TypeVar('N')

import numpy as np
from scipy.integrate import solve_ivp
from scipy.integrate._ivp.ivp import OdeResult


def integrate_reaction_network(init_nonzero_concs : dict[str, float], rate_const_tensors : dict[int, np.ndarray], idxs_by_species : dict[str, int], t0 : float=0.0, tf : float=10.0, **options) -> OdeResult:
    '''Solve system of ODEs for processed reaction network. Returns the SciPy ODEResult object containing all solutions'''
    n_species = len(idxs_by_species)
    init_concs = np.zeros(n_species, dtype=float) # initialize all concentrations to 0 M
    for species, c0 in init_nonzero_concs.items():
        i = idxs_by_species[species]
        init_concs[i] = c0 # set non-zero concs by passed values

    # Define ODE time step function
    def law_of_mass_action(t : float, C : np.ndarray[Shape[N], float], rate_const_tensors : dict[int, np.ndarray]) -> np.ndarray[Shape[N], float]:
        K  = rate_const_tensors[1]
        K2 = rate_const_tensors[2]
        C2 = np.outer(C, C) # captures every possible pairwise product of cocentrations

        return np.dot(K, C) + np.tensordot(K2, C2) # sum contributions from first and second-order rxns, respectively

    return solve_ivp(law_of_mass_action, t_span=[t0, tf], y0=init_concs, args=(rate_const_tensors,), **options)