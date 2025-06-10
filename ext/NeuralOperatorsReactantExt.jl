module NeuralOperatorsReactantExt

using NeuralOperators: NeuralOperators
using Reactant: Reactant, TracedRNumber

NeuralOperators.unwrapped_eltype(x::TracedRNumber) = Reactant.unwrapped_eltype(x)

end
