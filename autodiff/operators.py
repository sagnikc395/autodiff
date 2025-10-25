from op import Op
import numpy as np 

add = Op("add", 2, lambda x, y: x + y, lambda x, y, u: (u, u))
sub = Op("sub", 2, lambda x, y: x - y, lambda x, y, u: (u, -u))
mul = Op("mul", 2, lambda x, y: x * y, lambda x, y, u: (u * y, u * x))
matmul = Op(
    "matmul",
    2,
    lambda A, x: A @ x,
    lambda A, x, u: (u @ x.T if u.ndim > 1 else np.outer(u, x), A.T @ u),
)
inner = Op("inner", 2, lambda x, y: np.sum(x * y), lambda x, y, u: (u * y, u * x))
solve = Op(
    "solve",
    2,
    lambda A, x: np.linalg.solve(A, x),
    lambda A, x, u: (
        -np.linalg.inv(A).T @ np.outer(u, np.linalg.solve(A, x)),
        np.linalg.inv(A).T @ u,
    ),
)
logdet = Op(
    "logdet",
    1,
    lambda A: np.log(np.linalg.det(A)),
    lambda A, u: (u * np.linalg.inv(A).T,),
)

def logsumexp_forward(x):
    c = np.max(x)
    return c + np.log(np.sum(np.exp(x - c)))


def logsumexp_backward(x, u):
    ex = np.exp(x - np.max(x))
    softmax = ex / np.sum(ex)
    return (u * softmax,)

logsumexp = Op(
    "logsumexp",
    1,
    logsumexp_forward,
    logsumexp_backward,
)

exp = Op(
    name="exp",
    num_inputs=1,
    forward_fn=lambda x: np.exp(x),
    backward_fn=lambda x, u: (u * np.exp(x),),
)

log = Op(
    name="log",
    num_inputs=1,
    forward_fn=lambda x: np.log(x),
    backward_fn=lambda x, u: (u / x,),
)