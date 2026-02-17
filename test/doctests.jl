using NeuralOperators, Test, Documenter, FastTransforms

@testset "Doctests: Quality Assurance" begin
    DocMeta.setdocmeta!(
        NeuralOperators,
        :DocTestSetup,
        :(using Lux, NeuralOperators, Random);
        recursive = true,
    )
    doctest(NeuralOperators; manual = false)
end
