"""
    Nathaniel Linden (UCSD MAE)
    Created: November 26th, 2024
"""
import jax
import jax.numpy as jnp
import equinox as eqx

jax.config.update("jax_enable_x64", True)

class Hill(eqx.Module):

    def __call__(self, t, y, args):

        y = eqx.error_if(y, jnp.any(jnp.isnan(y) | jnp.isinf(y)), "y is NaN or Inf")

        # unpack parameters
        V_max   = args[0]
        K_m     = args[1]
        n      = args[2]
 
        # unpack states
        S = y[0]
        P = y[1]

        # clip S if its below 0!
        # For some reason this generates negative values when the time step is not ridiculously small
        # so we need to clip it to 0 to avoid numerical trouble when computing gradients
        S = jnp.clip(S, min=0, max=None)
        # P = jnp.clip(P, min=0, max=None)
    
        # FLUXES
        # single AXP complexing
        J1 = V_max*(S**n)/(K_m + (S**n))

        # clip if its too big
        # J1 = jnp.clip(J1, max=1e6, min=None)
        
        # now return the odes for each state variable
        d_S = -J1
        d_P = J1

        dydt = jnp.array([d_S, d_P])

        # jax.debug.print(
        #     "t: {t}, x: {x}, dxdt: {dxdt}, args: {args}", t=t, x=y, dxdt=dydt, args=args, ordered=True
        # )

        dydt = eqx.error_if(dydt, jnp.any(jnp.isnan(dydt) | jnp.isinf(dydt)), "dydt is NaN or Inf")

        return dydt