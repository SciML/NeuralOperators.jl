module NeuralOperators

using AbstractFFTs: rfft, irfft
using ConcreteStructs: @concrete
using Random: Random, AbstractRNG

using Lux
using LuxCore: LuxCore, AbstractLuxLayer, AbstractLuxContainerLayer, AbstractLuxWrapperLayer
using WeightInitializers: glorot_uniform

include("utils.jl")

include("transform.jl")
include("layers.jl")

# include("models/fno.jl")
# include("models/deeponet.jl")
# include("models/nomad.jl")

export FourierTransform
export SpectralConv, OperatorConv, SpectralKernel, OperatorKernel
# export FourierNeuralOperator
# export DeepONet
# export NOMAD

end
