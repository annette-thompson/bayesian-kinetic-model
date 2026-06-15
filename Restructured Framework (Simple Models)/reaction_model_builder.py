"""
Generic reaction network model builder from JSON specifications.
Annette Thompson (CU Boulder) - 2025
"""
import jax.numpy as jnp
import equinox as eqx
import json
import numpy as np
from pathlib import Path


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
    param_values : dict
        Dictionary of nominal parameter values
    """
    
    reactions = _load_reactions(reactions_json)
    
    # Infer species/parameters in encounter order to preserve JSON-defined semantics.
    species = []
    params = []
    param_values = {}

    def append_unique(target, item):
        if item not in target:
            target.append(item)

    for reaction in reactions:
        for state in reaction['reactants'].keys():
            append_unique(species, state)
        for state in reaction['products'].keys():
            append_unique(species, state)

        k_fwd = reaction['rate_const_key']
        append_unique(params, k_fwd)
        param_values[k_fwd] = reaction['rate_const_value']

        if reaction.get('reversible', False):
            k_rev = reaction['rvs_rate_const_key']
            append_unique(params, k_rev)
            param_values[k_rev] = reaction['rvs_rate_const_value']

    # Create mappings
    species_idx = {state: i for i, state in enumerate(species)}
    param_idx = {param: i for i, param in enumerate(params)}

    # Compile reactions into irreversible channels (forward + reverse when reversible)
    channels = []

    def add_channel(param_name, reactants, products):
        reactant_idxs = [species_idx[state] for state in reactants.keys()]
        reactant_stoich = [float(stoich) for stoich in reactants.values()]

        delta = np.zeros(len(species), dtype=np.float64)
        for state, stoich in reactants.items():
            delta[species_idx[state]] -= float(stoich)
        for state, stoich in products.items():
            delta[species_idx[state]] += float(stoich)

        channels.append(
            {
                'param_i': param_idx[param_name],
                'reactant_idxs': reactant_idxs,
                'reactant_stoich': reactant_stoich,
                'delta': delta,
            }
        )

    for reaction in reactions:
        add_channel(
            reaction['rate_const_key'],
            reaction['reactants'],
            reaction['products'],
        )

        if reaction.get('reversible', False):
            add_channel(
                reaction['rvs_rate_const_key'],
                reaction['products'],
                reaction['reactants'],
            )

    n_channels = len(channels)
    max_reactants = max(len(ch['reactant_idxs']) for ch in channels) if n_channels > 0 else 0

    # Dense channel tensors for JAX-friendly runtime computation
    param_idx_arr = np.zeros((n_channels,), dtype=np.int32)
    reactant_idx_arr = np.zeros((n_channels, max_reactants), dtype=np.int32)
    reactant_stoich_arr = np.zeros((n_channels, max_reactants), dtype=np.float64)
    reactant_mask_arr = np.zeros((n_channels, max_reactants), dtype=bool)
    stoich_matrix = np.zeros((len(species), n_channels), dtype=np.float64)

    for j, ch in enumerate(channels):
        param_idx_arr[j] = ch['param_i']
        stoich_matrix[:, j] = ch['delta']
        r_len = len(ch['reactant_idxs'])
        if r_len > 0:
            reactant_idx_arr[j, :r_len] = np.asarray(ch['reactant_idxs'], dtype=np.int32)
            reactant_stoich_arr[j, :r_len] = np.asarray(ch['reactant_stoich'], dtype=np.float64)
            reactant_mask_arr[j, :r_len] = True

    param_idx_arr = jnp.asarray(param_idx_arr)
    reactant_idx_arr = jnp.asarray(reactant_idx_arr)
    reactant_stoich_arr = jnp.asarray(reactant_stoich_arr)
    reactant_mask_arr = jnp.asarray(reactant_mask_arr)
    stoich_matrix = jnp.asarray(stoich_matrix)

    # Build the ODE system class dynamically
    class ReactionNetwork(eqx.Module):
        """Auto-generated reaction network ODE system"""

        param_idx_arr: jnp.ndarray
        reactant_idx_arr: jnp.ndarray
        reactant_stoich_arr: jnp.ndarray
        reactant_mask_arr: jnp.ndarray
        stoich_matrix: jnp.ndarray

        def __init__(self):
            self.param_idx_arr = param_idx_arr
            self.reactant_idx_arr = reactant_idx_arr
            self.reactant_stoich_arr = reactant_stoich_arr
            self.reactant_mask_arr = reactant_mask_arr
            self.stoich_matrix = stoich_matrix
        
        def __call__(self, t, y, args):
            if self.param_idx_arr.shape[0] == 0:
                return jnp.zeros_like(y)

            theta = jnp.asarray(args)

            # For each channel, gather reactant concentrations and compute
            # mass-action term as product_i y[idx_i]**stoich_i
            reactant_conc = y[self.reactant_idx_arr]
            reactant_powers = jnp.where(
                self.reactant_mask_arr,
                reactant_conc ** self.reactant_stoich_arr,
                1.0,
            )
            mass_action_terms = jnp.prod(reactant_powers, axis=1)

            # rates[j] = theta[param_idx_arr[j]] * mass_action_terms[j]
            rates = theta[self.param_idx_arr] * mass_action_terms

            # dydt = S @ rates
            return self.stoich_matrix @ rates
    
    return ReactionNetwork(), species, params, param_values


def _load_reactions(reactions_json):
    """Load reactions from a single JSON file or merge all JSON files in a directory."""
    source_path = Path(reactions_json).expanduser().resolve()
    if not source_path.exists():
        raise FileNotFoundError(f"Reaction source not found: {source_path}")

    file_reaction_specs = []
    if source_path.is_dir():
        json_files = sorted(source_path.glob("*.json"))
        if not json_files:
            raise ValueError(f"No JSON reaction files found in directory: {source_path}")
        for file_path in json_files:
            with open(file_path, "r") as f:
                spec = json.load(f)
            file_reaction_specs.append((str(file_path), spec))
    else:
        with open(source_path, "r") as f:
            spec = json.load(f)
        file_reaction_specs.append((str(source_path), spec))

    reactions = []
    seen_param_values = {}
    required_keys = {
        "rxn_name",
        "reactants",
        "products",
        "rate_const_key",
        "rate_const_value",
        "reversible",
    }

    for source_file, reaction_spec in file_reaction_specs:
        if "reactions" not in reaction_spec:
            raise KeyError(f"Missing 'reactions' key in {source_file}")
        if not isinstance(reaction_spec["reactions"], list):
            raise TypeError(f"'reactions' must be a list in {source_file}")

        for idx, reaction in enumerate(reaction_spec["reactions"]):
            if not isinstance(reaction, dict):
                raise TypeError(
                    f"Reaction index {idx} in {source_file} must be a mapping/dict."
                )

            missing_keys = sorted(required_keys.difference(reaction.keys()))
            if missing_keys:
                raise KeyError(
                    f"Reaction index {idx} in {source_file} missing keys: {missing_keys}"
                )

            if not isinstance(reaction["reactants"], dict) or not isinstance(reaction["products"], dict):
                raise TypeError(
                    f"Reaction '{reaction.get('rxn_name', idx)}' in {source_file} must use dict stoichiometry for reactants/products."
                )

            _validate_param_conflict(
                seen_param_values=seen_param_values,
                param_key=reaction["rate_const_key"],
                param_value=float(reaction["rate_const_value"]),
                source_file=source_file,
                reaction_name=reaction.get("rxn_name", f"reaction_{idx}"),
            )

            if reaction.get("reversible", False):
                if "rvs_rate_const_key" not in reaction or "rvs_rate_const_value" not in reaction:
                    raise KeyError(
                        f"Reversible reaction '{reaction.get('rxn_name', idx)}' in {source_file} is missing reverse-rate keys."
                    )
                _validate_param_conflict(
                    seen_param_values=seen_param_values,
                    param_key=reaction["rvs_rate_const_key"],
                    param_value=float(reaction["rvs_rate_const_value"]),
                    source_file=source_file,
                    reaction_name=reaction.get("rxn_name", f"reaction_{idx}"),
                )

            reactions.append(reaction)

    if not reactions:
        raise ValueError(f"No reactions loaded from source: {source_path}")
    return reactions


def _validate_param_conflict(seen_param_values, param_key, param_value, source_file, reaction_name):
    """Reject conflicting nominal values when the same kinetic key appears more than once."""
    if param_key not in seen_param_values:
        seen_param_values[param_key] = (param_value, source_file, reaction_name)
        return

    previous_value, previous_file, previous_reaction = seen_param_values[param_key]
    if not np.isclose(previous_value, param_value):
        raise ValueError(
            "Conflicting nominal values for parameter "
            f"'{param_key}': {previous_value} ({previous_reaction} in {previous_file}) vs "
            f"{param_value} ({reaction_name} in {source_file})."
        )

