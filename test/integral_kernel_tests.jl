@testitem "IntegralKernelOperator" setup=[SharedTestSetup] begin
    @testset "BACKEND: $(mode)" for (mode, aType, dev, ongpu) in MODES
        rng = StableRNG(12345)

        @testset "Kernel test" begin
            u = rand(Float64, 1, 16) |> aType
            domain_ = reshape([0.0, 1.0], 1, 2) |> aType

            integral_kernel_ = IntegralKernel(
                Chain(Dense(1 => 16), Dense(16 => 16), Dense(16 => 1)),
                Chain(Dense(2 => 16), Dense(16 => 16), Dense(16 => 1)))

            ps, st = Lux.setup(rng, integral_kernel_) |> dev

            # @inferred integral_kernel_((u, domain_), ps, st)
            # @jet integral_kernel_((u, domain_), ps, st)

            pred = first(integral_kernel_((u, domain_), ps, st))
            @test size(pred) == size(u)
        end

        @testset "Operator test" begin
            u = rand(Float64, 1, 16) |> aType
            domain_ = reshape([0.0, 1.0], 1, 2) |> aType

            kernels = [IntegralKernel(
                           Chain(Dense(1 => 16), Dense(16 => 16), Dense(16 => 1)),
                           Chain(Dense(2 => 16), Dense(16 => 16), Dense(16 => 1)))
                       for _ in 1:3]

            model = IntegralKernelOperator(
                Chain(Dense(1 => 16), Dense(16 => 16), Dense(16 => 1)), kernels,
                Chain(Dense(1 => 16), Dense(16 => 16), Dense(16 => 1)), domain_)

            ps, st = Lux.setup(rng, model) |> dev

            # @inferred model(u, ps, st)
            # @jet model(u, ps, st)

            pred = first(model(u, ps, st))
            @test size(pred) == size(u)
        end
    end
end
