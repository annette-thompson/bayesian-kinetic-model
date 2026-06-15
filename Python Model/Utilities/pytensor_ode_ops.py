import numpy as np
import pytensor.tensor as pt
from pytensor.graph import Apply, Op


class SolOp(Op):
    """Pytensor Op for ODE simulation output with an attached gradient Op."""

    def __init__(self, sol_op_jax_jitted, vjp_sol_op):
        self.sol_op_jax_jitted = sol_op_jax_jitted
        self.vjp_sol_op = vjp_sol_op

    def make_node(self, *inputs):
        # Keep all values in float64 to avoid mixed-precision instability in
        # JAX callbacks and VJP computation.
        inputs = [pt.cast(pt.as_tensor_variable(inp), "float64") for inp in inputs]
        outputs = [pt.matrix(dtype="float64")]
        return Apply(self, inputs, outputs)

    def perform(self, node, inputs, outputs):
        result = self.sol_op_jax_jitted(*inputs)
        result_arr = np.asarray(result, dtype="float64")
        if not np.isfinite(result_arr).all():
            raise FloatingPointError(
                "Non-finite values returned by ODE forward solve in SolOp.perform."
            )
        outputs[0][0] = result_arr

    def grad(self, inputs, output_grads):
        (gz,) = output_grads
        return self.vjp_sol_op(*inputs, gz)


class VJPSolOp(Op):
    """Pytensor Op for vector-Jacobian products of the ODE simulation Op."""

    def __init__(self, vjp_sol_op_jax_jitted):
        self.vjp_sol_op_jax_jitted = vjp_sol_op_jax_jitted

    def make_node(self, *inputs):
        if len(inputs) < 2:
            raise ValueError("VJPSolOp expects parameter inputs followed by output gradient gz.")

        tensor_inputs = [pt.cast(pt.as_tensor_variable(inp), "float64") for inp in inputs]
        *params, gz = tensor_inputs
        outputs = [pt.tensor(dtype="float64", shape=param.type.shape) for param in params]
        return Apply(self, [*params, gz], outputs)

    def perform(self, node, inputs, outputs):
        *params, gz = inputs
        if any(not np.isfinite(np.asarray(param)).all() for param in params):
            raise FloatingPointError(
                "Non-finite parameter values reached VJPSolOp.perform before JAX VJP evaluation."
            )
        if not np.isfinite(np.asarray(gz)).all():
            raise FloatingPointError(
                "Non-finite output gradient reached VJPSolOp.perform before JAX VJP evaluation."
            )

        result = self.vjp_sol_op_jax_jitted(gz, *params)
        for i, res in enumerate(result):
            res_arr = np.asarray(res, dtype="float64")
            if not np.isfinite(res_arr).all():
                raise FloatingPointError(
                    "Non-finite values returned by ODE VJP solve in VJPSolOp.perform."
                )
            outputs[i][0] = res_arr
