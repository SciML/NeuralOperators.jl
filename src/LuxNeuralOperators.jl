module LuxNeuralOperators

using PrecompileTools: @recompile_invalidations

@recompile_invalidations begin
    using ArgCheck: @argcheck
    using ChainRulesCore: ChainRulesCore, NoTangent
    using ConcreteStructs: @concrete
    using FFTW: FFTW, irfft, rfft
    using Lux: _print_wrapper_model
    using LuxCore: LuxCore, AbstractExplicitLayer
    using NNlib: NNlib, ⊠, batched_adjoint
    using Random: Random, AbstractRNG
    using Reexport: @reexport
    import Base: show
end

const CRC = ChainRulesCore

@reexport using Lux

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
