## autodiff — A Minimal Automatic Differentiation Engine in Python

This is an educational reimplementation of core automatic differentiation (autograd) concepts from first principles — built as part of CS689: Machine Learning at UMass Amherst (Fall 2025).

### Overview

autodiff is a lightweight Python library that implements reverse-mode automatic differentiation (the algorithm behind PyTorch’s autograd and JAX’s grad), written completely from scratch using NumPy only.

It constructs a computational graph dynamically during forward computation and supports efficient gradient backpropagation through complex compositions of operations — making it a transparent, extensible foundation for understanding how modern deep learning frameworks work under the hood.

### Key Features
- Computation Graph Construction — every operation dynamically creates nodes tracking dependencies between variables.
- Reverse-Mode Backpropagation — gradients are computed efficiently for scalar outputs with respect to all intermediate variables.
- Extensible Op Class System — easily define custom operations with forward and backward functions.
- Matrix-Compatible — supports vector and matrix operations (add, matmul, logdet, solve, etc.).
- Minimal Yet Modular Design — no external ML frameworks; pure NumPy core for educational clarity.
- Unit-Tested — verified gradients against PyTorch/JAX to ensure correctness.

