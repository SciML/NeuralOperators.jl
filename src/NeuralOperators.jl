module NeuralOperators

using AbstractFFTs: fft, rfft, ifft, irfft, fftshift
using Base.Broadcast: BroadcastFunction
using ConcreteStructs: @concrete
import FFTW
using Random: Random, AbstractRNG

using Lux: Lux, Chain, Dense, Conv, Parallel, NoOpLayer, WrappedFunction, Scale,
    recursive_eltype
using LuxCore: LuxCore, AbstractLuxLayer, AbstractLuxWrapperLayer
using LuxLib: fast_activation!!
using NNlib: batched_mul, gelu, pad_constant, sigmoid, sigmoid_fast, tanh_fast
using PrecompileTools: @compile_workload, @setup_workload
using SciMLPublic: @public
using WeightInitializers: glorot_uniform

include("utils.jl")

include("transform.jl")
include("layers.jl")

include("models/fno.jl")
include("models/deeponet.jl")
include("models/nomad.jl")
include("precompilation.jl")

export FourierTransform
export SpectralConv, OperatorConv, SpectralKernel, OperatorKernel
export GridEmbedding, ComplexDecomposedLayer, SoftGating

export FourierNeuralOperator
export DeepONet
export NOMAD

@public AbstractTransform, transform, truncate_modes, inverse

end
