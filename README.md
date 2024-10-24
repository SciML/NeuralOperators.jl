# NeuralOperators.jl

[![Join the chat at https://julialang.zulipchat.com #sciml-bridged](https://img.shields.io/static/v1?label=Zulip&message=chat&color=9558b2&labelColor=389826)](https://julialang.zulipchat.com/#narrow/stream/279055-sciml-bridged)
[![Global Docs](https://img.shields.io/badge/docs-SciML-blue.svg)](https://sciml.github.io/NeuralOperators.jl/)

[![codecov](https://codecov.io/gh/SciML/NeuralOperators.jl/branch/main/graph/badge.svg?token=wTS4cxrvB1)](https://codecov.io/gh/SciML/NeuralOperators.jl)
[![Build Status](https://github.com/SciML/NeuralOperators.jl/workflows/CI/badge.svg)](https://github.com/SciML/NeuralOperators.jl/actions?query=workflow%3ACI)
[![Build status](https://badge.buildkite.com/dd09599b08f61de1b5c7960aacd5390554c53e3b54f1407ba1.svg?branch=main)](https://buildkite.com/julialang/neuraloperators-dot-jl)

[![ColPrac: Contributor's Guide on Collaborative Practices for Community Packages](https://img.shields.io/badge/ColPrac-Contributor%27s%20Guide-blueviolet)](https://github.com/SciML/ColPrac)
[![SciML Code Style](https://img.shields.io/static/v1?label=code%20style&message=SciML&color=9558b2&labelColor=389826)](https://github.com/SciML/SciMLStyle)

NeuralOperators.jl is a package written in Julia to provide the architectures for learning
mapping between function spaces, and learning grid invariant solution of PDEs. Checkout the
[documentation](https://sciml.github.io/NeuralOperators.jl/) for tutorials and API
reference.

## Installation

On Julia 1.10+, you can install `NeuralOperators.jl` by running

```julia
import Pkg
Pkg.add("NeuralOperators")
```

## Citation

If you found this library to be useful in academic work, then please cite:

```bibtex
@software{pal2023lux,
  author    = {Pal, Avik},
  title     = {{Lux: Explicit Parameterization of Deep Neural Networks in Julia}},
  month     = apr,
  year      = 2023,
  note      = {If you use this software, please cite it as below.},
  publisher = {Zenodo},
  version   = {v0.5.0},
  doi       = {10.5281/zenodo.7808904},
  url       = {https://doi.org/10.5281/zenodo.7808904}
}

@thesis{pal2023efficient,
  title     = {{On Efficient Training \& Inference of Neural Differential Equations}},
  author    = {Pal, Avik},
  year      = {2023},
  school    = {Massachusetts Institute of Technology}
}
```
