using ReTestItems, Test, NeuralOperators, Reactant

@testset "NeuralOperators.jl Tests" begin
    ReTestItems.runtests(NeuralOperators; logs = :issues)
end
