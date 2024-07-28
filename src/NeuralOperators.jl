module NeuralOperators

using ArgCheck: @argcheck
using ChainRulesCore: ChainRulesCore, NoTangent
using ConcreteStructs: @concrete
using FFTW: FFTW, irfft, rfft
using Lux
using LuxCore: LuxCore, AbstractExplicitLayer
using LuxDeviceUtils: get_device, LuxAMDGPUDevice
using NNlib: NNlib, ⊠, batched_adjoint
using Random: Random, AbstractRNG
using Reexport: @reexport

const CRC = ChainRulesCore

@reexport using Lux

abstract type NeuralOperator end

@concrete struct DeepONet{L} <: NeuralOperator
    ch::L
end

@concrete struct FourierNeuralOperator{L} <: NeuralOperator
    ch::L
end

for f in (:DeepONet, :FourierNeuralOperator)
    @eval Lux.setup(rng::RNG, l::L) where {RNG <: AbstractRNG, L <: $f} = Lux.setup(
        rng, l.ch)
    @eval (model::($f))(x::T, ps::NamedTuple, st::NamedTuple) where {T} = model.ch(
        x, ps, st)
    @eval Lux.Experimental.TrainState(model::L, args...; kwargs...) where {L <: $f} = Lux.Experimental.TrainState(
        model.ch, args...; kwargs...)
end

include("utils.jl")
include("transform.jl")

include("functional.jl")
include("layers.jl")

include("fno.jl")
include("deeponet.jl")

export FourierTransform
export SpectralConv, OperatorConv, SpectralKernel, OperatorKernel
export FourierNeuralOperator
export DeepONet

end
