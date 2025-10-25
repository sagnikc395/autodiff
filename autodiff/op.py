# op is the overloaded operator that operates on our data
# to add various methods

from var import Var

class Op:
    def __init__(self, name, num_inputs, forward_fn, backward_fn) -> None:
        self.name = name
        self.num_inputs = num_inputs
        self.forward_fn = forward_fn
        self.backward_fn = backward_fn

    def __call__(self, *inputs):
        # Ensure inputs are Var instances
        if not all(isinstance(i, Var) for i in inputs):
            raise TypeError(f"All inputs to '{self.name}' must be Var instances.")
        return Var(self, inputs)