include("layers_testsetup.jl")

@testset "SpectralConv" begin
    run_op_tests(SpectralConv, LAYERS_SETUPS)
end
