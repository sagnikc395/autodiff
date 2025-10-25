from var import Var
from utils import get_topological_order 
from typing import List 
import numpy as np 


def backpropagation(output: Var, inputs: List[Var]) -> List[np.ndarray]:
    """Compute gradients of output wrt given input nodes."""
    order = get_topological_order(output)

    # Reset all gradients
    for node in order:
        node.grad = np.zeros_like(node.value)
    output.grad = np.ones_like(output.value)

    # Traverse backward
    for node in reversed(order):
        if node.op is None:
            continue
        parent_values = [p.value for p in node.parents]
        grads_to_parents = node.op.backward_fn(*parent_values, node.grad)
        for parent, g in zip(node.parents, grads_to_parents):
            parent.grad += g

    return [inp.grad for inp in inputs]


def grad(func):
    def grad_f(*inputs):
        vars_ = [Var.from_value(v) if not isinstance(v, Var) else v for v in inputs]
        out = func(*vars_)
        return backpropagation(out, vars_)

    return grad_f