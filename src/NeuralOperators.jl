module NeuralOperators

using AbstractFFTs: fft, rfft, ifft, irfft, fftshift
using ConcreteStructs: @concrete
using Random: Random, AbstractRNG

using Lux: Lux, Chain, Dense, Conv, Parallel, NoOpLayer, WrappedFunction, Scale
using LuxCore: LuxCore, AbstractLuxLayer, AbstractLuxWrapperLayer
using LuxLib: fast_activation!!
using NNlib: NNlib, batched_mul, gelu
using WeightInitializers: glorot_uniform

include("utils.jl")

include("transform.jl")
include("layers.jl")

include("models/fno.jl")
include("models/deeponet.jl")
include("models/nomad.jl")

export FourierTransform
export SpectralConv, OperatorConv, SpectralKernel, OperatorKernel
export GridEmbedding, ComplexDecomposedLayer, SoftGating

export FourierNeuralOperator
export DeepONet
export NOMAD

end
