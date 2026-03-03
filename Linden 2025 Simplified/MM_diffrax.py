"""
    Nathaniel Linden (UCSD MAE)
    Created: November 26th, 2024
"""
import jax.numpy as jnp
import equinox as eqx

class MM(eqx.Module):

    def __call__(self, t, y, args):
        
        # unpack parameters
        V_max   = args[0]
        K_m     = args[1]
 
        # unpack states
        S = y[0]
        P = y[1]
    
        # FLUXES
        # single AXP complexing
        J1 = V_max*S/(K_m + S)
        
        # now return the odes for each state variable
        d_S = -J1
        d_P = J1

        return jnp.array([d_S, d_P])