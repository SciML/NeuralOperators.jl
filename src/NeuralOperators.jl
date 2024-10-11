module NeuralOperators

using ArgCheck: @argcheck
using ChainRulesCore: ChainRulesCore, NoTangent, @non_differentiable
using ConcreteStructs: @concrete
using FFTW: FFTW, irfft, rfft
using Random: Random, AbstractRNG
using Static: StaticBool, False, True, known, static

using Lux
using LuxCore: LuxCore, AbstractLuxLayer, AbstractLuxContainerLayer, AbstractLuxWrapperLayer
using LuxLib: batched_matmul
using MLDataDevices: AbstractDevice, AbstractGPUDevice
using NNlib: NNlib

const BoolLike = Union{Bool, StaticBool, Val{true}, Val{false}}
const CRC = ChainRulesCore

include("utils.jl")

include("transform.jl")
include("layers.jl")

include("models/fno.jl")
include("models/deeponet.jl")
include("models/nomad.jl")

export FourierTransform
export SpectralConv, OperatorConv, SpectralKernel, OperatorKernel
export FourierNeuralOperator
export DeepONet
export NOMAD

end
