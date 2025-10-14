from typing import Callable


class Op:
    def __init__(self,name: str,num_inputs: int, forward_fn: Callable, backward_fn: Callable):
        self.name = name
        self.num_inputs = num_inputs
        self.forward_fn = forward_fn
        self.backward_fn = backward_fn


    def add(self,x,y):
        # elementwise addition of two arrays z = x + y 
        pass

    def sub(self,x,y):
        # elementwise subtraction of two arrays z = x -y
        pass

    def mul(self,x,y):
        # elementwise multiplication of 2 arrays z = x mul y
        pass

    def matmul(self,x,y):
        # matrix multiplication bw two arrays z = XY
        pass

    def inner(self,x,y):
        # inner (dot) product bw two vectors or matrices
        pass

    def sum(self,x,axis=0):
        # summation over all elements or along a given axis
        pass

    def solve(self,eqn):
        # solves the linear system Ax = b
        pass

    def logdet(self,A):
        # computes the logarithm of the determinant of a matrix
        pass

    def logsumexp(self,x):
        # computes the logarithm of the determinant of a matrix log(det(A))
        pass

    def exp(self,x):
        # elementwise exponential of an array
        pass

    def log(self,x):
        # elementwise natural logarithm of an array
        pass

    
