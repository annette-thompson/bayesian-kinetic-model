"""
    Nathaniel Linden (UCSD MAE)
    Created: November 26th, 2024
"""
import jax.numpy as jnp
import equinox as eqx

class MA(eqx.Module):
    def __call__(self, t, y, args):
        
        # unpack parameters
        k_f     = args[0]
        k_r     = args[1]
        k_cat   = args[2] 

        # unpack states
        S = y[0]
        E = y[1]
        ES = y[2]
        P = y[3]
    
        # FLUXES
        # single AXP complexing
        J1 = k_f*S*E - k_r*ES
        J2 = k_cat*ES

        # now return the odes for each state variable
        d_S = -J1
        d_E = -J1 + J2
        d_ES = J1 - J2
        d_P = J2

        return jnp.array([d_S, d_E, d_ES, d_P])