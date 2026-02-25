"""
Generic reaction network model builder from JSON specifications.
Annette Thompson (CU Boulder) - 2025
"""
import jax.numpy as jnp
import equinox as eqx
import json


def build_ode_system_from_reactions(reactions_json):
    """
    Build a generic ODE system from a JSON specification of reactions.
    Automatically infers species and parameters from what's in the JSON.
    
    Forward reaction rate: k_forward * product(reactants)
    Reverse reaction rate: k_reverse * product(products)
    
    Parameters:
    -----------
    reactions_json : str
        Path to JSON file with reaction specifications
        
    Returns:
    --------
    ReactionNetwork : class
        An equinox module that computes ODEs for the reaction network
    state_names : list
        Names of the state variables in order (inferred from reactions)
    param_names : list
        Names of the parameters in order (inferred from reactions)
    """
    
    with open(reactions_json, 'r') as f:
        reaction_spec = json.load(f)
    
    reactions = reaction_spec['reactions']
    
    # Automatically infer species and parameters from reactions
    species_set = set()
    params_set = set()
    
    for reaction in reactions:
        # Collect species from reactants and products
        species_set.update(reaction['reactants'].keys())
        species_set.update(reaction['products'].keys())
        
        # Collect parameters from rate constants
        params_set.add(reaction['rate_constant'])
        if reaction.get('reversible', False):
            params_set.add(reaction['reverse_rate'])
    
    # Convert sets to sorted lists for consistent ordering
    species = sorted(list(species_set))
    params = sorted(list(params_set))
    
    # Create mappings
    species_idx = {state: i for i, state in enumerate(species)}
    param_idx = {param: i for i, param in enumerate(params)}
    
    # Build the ODE system class dynamically
    class ReactionNetwork(eqx.Module):
        """Auto-generated reaction network ODE system"""
        
        def __call__(self, t, y, args):
            # Initialize derivatives
            dydt = jnp.zeros(len(species))
            
            # Process each reaction
            for reaction in reactions:
                reactants = reaction['reactants']
                products = reaction['products']
                k_name = reaction['rate_constant']
                k = args[param_idx[k_name]]
                
                # Compute forward reaction rate: k * product(reactants)
                rate_fwd = k
                for reactant, stoich in reactants.items():
                    rate_fwd *= y[species_idx[reactant]] ** stoich
                
                # Update derivatives for reactants (consumed)
                for reactant, stoich in reactants.items():
                    dydt = dydt.at[species_idx[reactant]].add(-stoich * rate_fwd)
                
                # Update derivatives for products (produced)
                for product, stoich in products.items():
                    dydt = dydt.at[species_idx[product]].add(stoich * rate_fwd)
                
                # Handle reversible reactions
                if reaction.get('reversible', False):
                    k_rev_name = reaction['reverse_rate']
                    k_rev = args[param_idx[k_rev_name]]
                    
                    # Compute reverse reaction rate: k_rev * product(products)
                    rate_rev = k_rev
                    for product, stoich in products.items():
                        rate_rev *= y[species_idx[product]] ** stoich
                    
                    # Update derivatives for reverse reaction
                    for product, stoich in products.items():
                        dydt = dydt.at[species_idx[product]].add(-stoich * rate_rev)
                    
                    for reactant, stoich in reactants.items():
                        dydt = dydt.at[species_idx[reactant]].add(stoich * rate_rev)
            
            return dydt
    
    return ReactionNetwork(), species, params

