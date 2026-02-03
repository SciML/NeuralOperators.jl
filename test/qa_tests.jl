using NeuralOperators, Test, ExplicitImports, Aqua

@testset "Aqua: Quality Assurance" begin
    Aqua.test_all(NeuralOperators; ambiguities = false)
    Aqua.test_ambiguities(NeuralOperators; recursive = false)
end

@testset "Explicit Imports: Quality Assurance" begin
    test_explicit_imports(NeuralOperators; all_qualified_accesses_are_public = false)
end
