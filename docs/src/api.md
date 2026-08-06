# Public API

The exported API consists of ready-to-use operator-learning models, reusable Lux layers,
and the Fourier transform used by spectral layers.

## Pre-Built Architectures

```@docs
NOMAD
DeepONet
FourierNeuralOperator
```

## Building blocks

```@docs
OperatorConv
SpectralConv
OperatorKernel
SpectralKernel
GridEmbedding
ComplexDecomposedLayer
SoftGating
```

## Transform

```@docs
NeuralOperators.FourierTransform
```

## Developer Extension API

!!! warning
    This interface is for package developers implementing transforms for
    [`OperatorConv`](@ref). Application code should use [`FourierTransform`](@ref),
    [`SpectralConv`](@ref), or [`SpectralKernel`](@ref) instead. The developer interface is
    versioned, but it is not exported and must be imported or qualified explicitly.

```@docs
NeuralOperators.AbstractTransform
NeuralOperators.transform
NeuralOperators.truncate_modes
NeuralOperators.inverse
```
