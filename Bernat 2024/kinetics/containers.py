'''Representation classes for raection-related functionality'''

__author__ = 'Timotej Bernat'

from typing import Optional, Union
from dataclasses import dataclass, field

import json
from pathlib import Path


@dataclass
class ElementaryReaction:
    '''For representing a single reactant -> product change in a human-readable format'''
    reactants : list[str]
    products  : list[str]
    rate_const_value : float
    rate_const_key : str = 'k'

    name : str = ''
    scaling_group_id : Optional[int] = None

    def create_reverse_reaction(self, k_rev_value : float, k_rev_key : Optional[str]=None, rev_name : Optional[str]=None, default_suffix : str='rev') -> 'ElementaryReaction': 
        '''Generates the corresponding reverse reaction given a reverse rate constant'''
        if k_rev_key is None: # TOSELF: worth making reverse rate constant VALUE default to that of forward as well? (might encourage redundant/lazy definitions)
            k_rev_key = f'{self.rate_const_key}{"_" if self.rate_const_key else ""}{default_suffix}'

        if rev_name is None:
            rev_name = f'{self.name}{"_" if self.name else ""}{default_suffix}'
        
        return self.__class__(
            reactants=self.products,
            products=self.reactants,
            rate_const_value=k_rev_value,
            rate_const_key=k_rev_key,
            name=rev_name,
        )
    reversed = create_rev_rxn = create_reverse_reaction # aliases for convenience

    # representation and expression strings
    @property
    def order(self) -> int:
        return len(self.reactants)

    @property
    def rate_expression(self) -> str:
        '''Generate algebraic rate equation for the current reaction step'''
        return "*".join([self.rate_const_key] + self.reactants)

    def reaction_expression(self, spacing_width : int=1, species_sep : str='+', arrow_stem : str='=', arrow_head : str='>', arrow_seg_len : int=2) -> str:
        '''Generate symbolic representation of the current reaction'''
        assert(arrow_seg_len > 0)
        assert(spacing_width > 0)

        space = ' '*spacing_width
        species_sep_spaced = f'{space}{species_sep}{space}'
        reactant_str = species_sep_spaced.join(self.reactants)
        product_str  = species_sep_spaced.join(self.products)
        arrow = f'{arrow_stem*arrow_seg_len}[{self.rate_const_key}]{arrow_stem*arrow_seg_len}{arrow_head}'

        return f'{reactant_str}{space}{arrow}{space}{product_str}'
    
    def __str__(self) -> str:
        return self.reaction_expression()
    
    def __hash__(self) -> int:
        return hash(self.reaction_expression())
    
    # file I/O
    def to_file(self, save_path : Union[Path, str], indent : int=4) -> None:
        '''Save the current reaction to a file on disc'''
        if isinstance(save_path, str):
            save_path = Path(save_path)
        assert(save_path.suffix) == '.json' # only allow saving to JSON files for now

        with save_path.open('w') as file:
            json.dump(self.__dict__, file, indent=indent)

    @classmethod
    def from_file(cls, load_path : Union[Path, str]) -> 'ElementaryReaction':
        '''Load a reaction from a saved reaction file on disc'''
        if isinstance(load_path, str):
            load_path = Path(load_path)

        assert(load_path.exists())
        with load_path.open('r') as file:
            return cls(**json.load(file))

@dataclass
class StoichBalanceTerms:
    '''For encapsulating info about which material balance terms a transformation occurs in'''
    generation  : set[ElementaryReaction] = field(default_factory=set)
    consumption : set[ElementaryReaction] = field(default_factory=set)
    # flow_in  : set = field(default_factory=set) # may be worth including if flow/species removal terms are needed
    # flow_out : set = field(default_factory=set)

    @property
    def signed_rxns(self) -> list[tuple[int, ElementaryReaction]]:
        '''Returns all contributing reactions and their sign when inserted into a rate expression'''
        return [(1, rxn) for rxn in self.generation] + [(-1, rxn) for rxn in self.consumption]

    @staticmethod
    def _int_to_sign_str(sign_int : int) -> str:
        if sign_int not in (1, -1):
            raise ValueError
        return '-' if (sign_int == -1) else ''

    @property
    def rate_expression(self) -> str:
        '''Generate symbolic rate equation describing the species balance'''
        return ' + '.join(self._int_to_sign_str(sign_int)+rxn.rate_expression for sign_int, rxn in self.signed_rxns)

    @property
    def expressions(self) -> list[tuple[float, str]]:
        return [(1, desc) for desc in self.generation_expressions]