from var import Var
from utils import get_topological_order 


def backprop(output: Var, args: list[Var]):
    topo_order = get_topological_order(output)

    grad_map = {output: 1.0}

    for node in reversed(topo_order):
        current_grad = grad_map[node]

        parent_values = [p.value for p in node.parents]
        local_grads = node.op.backward_fn(current_grad, *parent_values)

        for i, parent_var in enumerate(node.parents):
            grad_map[parent_var] = grad_map.get(parent_var, 0.0) + local_grads[i]

    result_grads = tuple(grad_map.get(arg, 0.0) for arg in args)

    return result_grads
