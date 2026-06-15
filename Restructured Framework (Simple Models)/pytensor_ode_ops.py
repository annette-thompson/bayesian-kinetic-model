import numpy as np
import pytensor.tensor as pt
from pytensor.graph import Apply, Op


class SolOp(Op):
    """Pytensor Op for ODE simulation output with an attached gradient Op."""

    def __init__(self, sol_op_jax_jitted, vjp_sol_op):
        self.sol_op_jax_jitted = sol_op_jax_jitted
        self.vjp_sol_op = vjp_sol_op

    def make_node(self, *inputs):
        inputs = [pt.as_tensor_variable(inp) for inp in inputs]
        outputs = [pt.matrix()]
        return Apply(self, inputs, outputs)

    def perform(self, node, inputs, outputs):
        result = self.sol_op_jax_jitted(*inputs)
        outputs[0][0] = np.asarray(result, dtype="float64")

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

        tensor_inputs = [pt.as_tensor_variable(inp) for inp in inputs]
        *params, gz = tensor_inputs
        outputs = [param.type() for param in params]
        return Apply(self, [*params, gz], outputs)

    def perform(self, node, inputs, outputs):
        *params, gz = inputs
        result = self.vjp_sol_op_jax_jitted(gz, *params)
        for i, res in enumerate(result):
            outputs[i][0] = np.asarray(res, dtype="float64")
