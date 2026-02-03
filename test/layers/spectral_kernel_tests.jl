include("layers_testsetup.jl")

@testset "SpectralKernel" begin
    run_op_tests(SpectralKernel, LAYERS_SETUPS)
end
