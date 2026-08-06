include("layers_testsetup.jl")
using NNlib: sigmoid, sigmoid_fast, tanh_fast

@testset "SpectralKernel" begin
    run_op_tests(SpectralKernel, LAYERS_SETUPS)

    @testset "fast activations" begin
        x = reshape(Float32[-2, -1, 0, 1], 4, 1, 1)
        init_weight = (rng, T, dims...) -> zeros(T, dims...)

        for (activation, fast_activation) in ((tanh, tanh_fast), (sigmoid, sigmoid_fast))
            layer = SpectralKernel(
                1 => 1, (2,), activation; fno_skip = :none, init_weight
            )
            parameters, states = Lux.setup(StableRNG(12345), layer)
            output, _ = layer(x, parameters, states)

            @test output == fast_activation.(x)
        end
    end
end
