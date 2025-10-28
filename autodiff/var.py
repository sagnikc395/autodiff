import numpy as np

from typing import Tuple

from op import Op

class Var:
    def __init__(self, op: Op, parents: Tuple["Var", ...]):
        # Non-leaf node initialization
        if len(parents) != op.num_inputs:
            raise ValueError(
                f"Op '{op.name}' expects {op.num_inputs} inputs, got {len(parents)}."
            )

        self.op = op
        self.parents = parents
        self.value = op.forward_fn(*[p.value for p in parents])
        self.grad = np.zeros_like(self.value)

    @classmethod
    def from_value(cls, value):
        """Create a constant (leaf) variable."""
        obj = cls.__new__(cls)
        obj.op = None
        obj.parents = ()
        obj.value = np.array(value, dtype=float)
        obj.grad = np.zeros_like(obj.value)
        return obj

    def __repr__(self):
        name = self.op.name if self.op else "leaf"
        return f"Var(value={self.value}, op='{name}')"
