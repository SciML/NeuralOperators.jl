@testitem "doctests: Quality Assurance" tags = [:qa] begin
    using Documenter, NeuralOperators

    DocMeta.setdocmeta!(
        NeuralOperators,
        :DocTestSetup,
        :(using Lux, NeuralOperators, Random);
        recursive = true,
    )
    doctest(NeuralOperators; manual = false)
end

@testitem "Aqua: Quality Assurance" tags = [:qa] begin
    using Aqua

    Aqua.test_all(NeuralOperators; ambiguities = false)
    Aqua.test_ambiguities(NeuralOperators; recursive = false)
end

@testitem "Explicit Imports: Quality Assurance" tags = [:qa] begin
    using ExplicitImports
    test_explicit_imports(NeuralOperators; all_qualified_accesses_are_public=false)
end
