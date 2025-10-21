# op is the overloaded operator that operates on our data
# to add various methods

import math
from collections import deque
from typing import Callable, Tuple
from dataclasses import dataclass , field
import numpy as np

@dataclass(frozen=True)
class Op:
    name: str
    num_inputs: int
    forward_fn: Callable
    backward_fn: Callable

    def __call__(self,*inputs):
        if len(inputs) != self.num_inputs:
            raise ValueError(f"Op '{self.name}' expects {self.num_inputs} inputs, but got {len(inputs)}.")

        parents = [p if isinstance(p,Var) else constant(p) for p in inputs]

        return Var(self,parents)


