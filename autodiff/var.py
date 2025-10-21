from op import Op
from typing import Tuple 


class Var:
    def __init__(self,op: Op, parents: Tuple['Var',...]):
        if len(parents) != op.num_inputs:
            raise ValueError(f"Number of parents {len(parents)} does not matc op. '{op.name}' num_inputs ({op.num_inputs})")

        self._op = op
        self._parents = tuple(parents)

        input_vals = [p.value for p in self._parents]
        self._value = self._op.forward_fn(*input_vals)


    @property
    def value(self):
        return self._value

    @property
    def op(self):
        return self._op

    @property
    def parents(self):
        return self._parents

    def __repr__(self):
        return f"Var(value={self.value:.4f}), op='{self.op.name}'"

    def __add__(self,other):
        return add(self,other)

    def __radd__(self,other):
        return add(other,self)

    def __sub__(self,other):
        return sub(self,other)

    def __rsub__(self,other):
        return sub(other,self)

    def __mul__(self,other):
        return mul(self,other)

    def __truediv__(self,other):
        return div(self,other)

    def __rtruediv__(self,other):
        return div(other,self)

    def __neg__(self):
        return neg(self)

    def __pow__(self,other):
        return power(self,other)

    def __rpow__(self,other):
        return power(other,self)


## global functions to create constant Var
def constant(value: float):
    # forward_fn and backward_fn are closures over the 'value'
    constant_op_instance = Op(
        name=f"constant({value})",
        num_inputs=0,
        forward_fn= lambda: value,
        backward_fn=lambda grad_output: (),
    )

    return Var(constant_op_instance,())

def exp_backwards(grad_output,x_val):
    return (grad_output * exp(x_val))

import math


exp = Op(
    "exp",1,
    lambda x: math.exp(x),
    lambda grad_output, x_val: (grad_output * math.exp(x_val))
)

log = Op(
    "log",1,
    lambda x:math.log(x),
    lambda grad_output,x_val: (grad_output * (1.0/x_val),)
)

sin = Op(
    "sin",1,
    lambda x: math.sin(x),
    lambda grad_output,x_val : (grad_output * math.cos(x_val))
)

cos = Op(
    "cos",1,
    lambda x: math.cos(x),
    lambda grad_output,x_val : (grad_output * (-math.sin(x_val)),)
)

add = Op(
    "add", 2,
    lambda x, y: x + y,
    lambda grad_output, x_val, y_val: (grad_output * 1.0, grad_output * 1.0)
)

sub = Op(
    "sub", 2,
    lambda x, y: x - y,
    lambda grad_output, x_val, y_val: (grad_output * 1.0, grad_output * -1.0)
)

mul = Op(
    "mul", 2,
    lambda x, y: x * y,
    lambda grad_output, x_val, y_val: (grad_output * y_val, grad_output * x_val)
)

div = Op(
    "div", 2,
    lambda x, y: x / y,
    lambda grad_output, x_val, y_val: (grad_output * (1.0 / y_val), grad_output * (-x_val / (y_val ** 2)))
)

power = Op(
    "power", 2,
    lambda x, y: x ** y,
    lambda grad_output, x_val, y_val: (
        grad_output * (y_val * (x_val ** (y_val - 1))),
        grad_output * (x_val ** y_val * math.log(x_val))
    )
)
