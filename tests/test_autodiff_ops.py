import sys
from pathlib import Path

import numpy as np

# Ensure the autodiff modules are importable despite non-package style imports.
MODULE_ROOT = Path(__file__).resolve().parent.parent / "autodiff"
if str(MODULE_ROOT) not in sys.path:
    sys.path.insert(0, str(MODULE_ROOT))

from autodiff.backprop import backpropagation, grad  
from autodiff.operators import add, exp, inner, log, logsumexp, matmul, mul, solve 
from autodiff.var import Var  


def test_addition_backward_scalar():
    x = Var.from_value(2.0)
    y = Var.from_value(-3.0)
    z = add(x, y)

    grads = backpropagation(z, [x, y])

    np.testing.assert_allclose(z.value, np.array(-1.0))
    np.testing.assert_allclose(grads[0], np.array(1.0))
    np.testing.assert_allclose(grads[1], np.array(1.0))


def test_chain_rule_accumulates_gradient():
    x = Var.from_value(3.0)
    y = Var.from_value(0.5)
    prod = mul(x, y)
    expr = add(prod, x)

    grad_x, grad_y = backpropagation(expr, [x, y])

    np.testing.assert_allclose(expr.value, np.array(2.5))
    np.testing.assert_allclose(grad_x, np.array(1.5))
    np.testing.assert_allclose(grad_y, np.array(3.0))


def test_matmul_and_inner_gradients():
    A = Var.from_value(np.array([[3.0, 2.0], [1.0, 4.0]]))
    x = Var.from_value(np.array([1.0, -1.0]))
    weight = Var.from_value(np.array([0.5, -2.0]))

    Ax = matmul(A, x)
    loss = inner(Ax, weight)

    grad_A, grad_x, grad_weight = backpropagation(loss, [A, x, weight])

    expected_grad_A = np.outer(weight.value, x.value)
    expected_grad_x = A.value.T @ weight.value
    expected_grad_weight = Ax.value

    np.testing.assert_allclose(grad_A, expected_grad_A)
    np.testing.assert_allclose(grad_x, expected_grad_x)
    np.testing.assert_allclose(grad_weight, expected_grad_weight)


def test_logsumexp_gradient_matches_softmax():
    x = Var.from_value(np.array([2.0, -1.0, 0.5]))
    output = logsumexp(x)

    (grad_x,) = backpropagation(output, [x])

    ex = np.exp(x.value - np.max(x.value))
    expected = ex / np.sum(ex)

    np.testing.assert_allclose(output.value, np.log(np.sum(np.exp(x.value))))
    np.testing.assert_allclose(grad_x, expected)


def test_log_and_exp_are_inverse_in_gradients():
    x = Var.from_value(np.array([1.5, 2.5]))
    exp_x = exp(x)
    composed = log(exp_x)

    (grad_x,) = backpropagation(composed, [x])

    np.testing.assert_allclose(composed.value, x.value)
    np.testing.assert_allclose(grad_x, np.ones_like(x.value))


def test_grad_decorator_matches_backprop():
    def function(a, b):
        prod = mul(a, b)
        shifted = add(prod, exp(a))
        return shifted

    grad_f = grad(function)

    a = Var.from_value(1.2)
    b = Var.from_value(-0.7)

    backprop_grads = backpropagation(function(a, b), [a, b])
    grad_grads = grad_f(1.2, -0.7)

    np.testing.assert_allclose(backprop_grads[0], grad_grads[0])
    np.testing.assert_allclose(backprop_grads[1], grad_grads[1])


def test_solve_backward_against_manual_derivative():
    A = Var.from_value(np.array([[4.0, 1.0], [2.0, 3.0]]))
    b = Var.from_value(np.array([1.0, -2.0]))
    solution = solve(A, b)
    loss = inner(solution, Var.from_value(np.array([1.0, 0.5])))

    grad_A, grad_b = backpropagation(loss, [A, b])

    expected_solution = np.linalg.solve(A.value, b.value)
    expected_grad_b = np.linalg.solve(A.value.T, np.array([1.0, 0.5]))

    # Manual derivative for A using implicit differentiation: -A^{-T} * (uv^T)
    v = np.array([1.0, 0.5])
    expected_grad_A = -np.linalg.solve(A.value.T, np.outer(v, expected_solution))

    np.testing.assert_allclose(solution.value, expected_solution)
    np.testing.assert_allclose(grad_b, expected_grad_b)
    np.testing.assert_allclose(grad_A, expected_grad_A)
