@testitem "Performance: Type Stability" setup = [SharedTestSetup] begin
    using InteractiveUtils
    using NeuralOperators
    using NeuralOperators: pad_zeros_spatial, expand_pad_dims, meshgrid

    # Test that key functions are type-stable (return concrete types, not Any)
    @testset "pad_zeros_spatial type stability" begin
        # 1D case
        x1d = rand(ComplexF32, 16, 5, 5)
        result_type = Base.return_types(pad_zeros_spatial, (typeof(x1d), Tuple{Int}))[1]
        @test result_type <: AbstractArray{ComplexF32, 3}

        # 2D case
        x2d = rand(ComplexF32, 16, 16, 5, 5)
        result_type = Base.return_types(pad_zeros_spatial, (typeof(x2d), Tuple{Int, Int}))[1]
        @test result_type <: AbstractArray{ComplexF32, 4}
    end

    @testset "meshgrid type stability" begin
        r1 = range(0.0f0, 1.0f0; length = 32)
        r2 = range(0.0f0, 1.0f0; length = 32)

        result_type = Base.return_types(meshgrid, (typeof(r1), typeof(r2)))[1]
        @test result_type <: AbstractArray{Float32, 3}
    end

    @testset "OperatorConv type stability" begin
        rng = StableRNG(12345)

        # 1D case
        sc1d = SpectralConv(2 => 5, (16,))
        ps, st = Lux.setup(rng, sc1d)
        x = rand(Float32, 1024, 2, 5)

        result_type = Base.return_types((sc1d, x, ps, st) -> sc1d(x, ps, st),
                                        (typeof(sc1d), typeof(x), typeof(ps), typeof(st)))[1]
        @test result_type <: Tuple{AbstractArray{Float32, 3}, Any}

        # 2D case
        sc2d = SpectralConv(2 => 5, (16, 16))
        ps2d, st2d = Lux.setup(rng, sc2d)
        x2d = rand(Float32, 32, 32, 2, 5)

        result_type = Base.return_types((sc2d, x2d, ps2d, st2d) -> sc2d(x2d, ps2d, st2d),
                                        (typeof(sc2d), typeof(x2d), typeof(ps2d), typeof(st2d)))[1]
        @test result_type <: Tuple{AbstractArray{Float32, 4}, Any}
    end
end

@testitem "Performance: Allocation Bounds" setup = [SharedTestSetup] begin
    using BenchmarkTools
    using NeuralOperators

    @testset "SpectralConv allocation bounds" begin
        rng = StableRNG(12345)

        # 1D case - establish allocation bounds
        sc1d = SpectralConv(2 => 5, (16,))
        ps1d, st1d = Lux.setup(rng, sc1d)
        x1d = rand(Float32, 1024, 2, 5)

        # Warmup
        sc1d(x1d, ps1d, st1d)

        # Check allocations are within expected bounds
        # Current baseline: ~256KB, allow 20% margin
        allocs = @allocated sc1d(x1d, ps1d, st1d)
        @test allocs < 310_000  # 256KB + 20% margin

        # 2D case
        sc2d = SpectralConv(2 => 5, (16, 16))
        ps2d, st2d = Lux.setup(rng, sc2d)
        x2d = rand(Float32, 32, 32, 2, 5)

        sc2d(x2d, ps2d, st2d)

        # Current baseline: ~598KB, allow 20% margin
        allocs = @allocated sc2d(x2d, ps2d, st2d)
        @test allocs < 720_000  # 600KB + 20% margin
    end

    @testset "FourierNeuralOperator allocation bounds" begin
        rng = StableRNG(12345)

        fno = FourierNeuralOperator(; chs = (2, 32, 32, 32, 32, 64, 1), modes = (16,))
        ps, st = Lux.setup(rng, fno)
        x = rand(Float32, 256, 2, 4)

        # Warmup
        fno(x, ps, st)

        # Current baseline: ~3.1MB, allow 50% margin for CI variance
        allocs = @allocated fno(x, ps, st)
        @test allocs < 5_000_000  # ~4.8MB max (generous bound for CI)
    end

    @testset "DeepONet allocation bounds" begin
        rng = StableRNG(12345)

        deeponet = DeepONet(; branch = (64, 32, 32, 16), trunk = (1, 8, 8, 16))
        ps, st = Lux.setup(rng, deeponet)
        u = rand(Float32, 64, 5)
        y = rand(Float32, 1, 10)

        # Warmup
        deeponet((u, y), ps, st)

        # Current baseline: ~3.7KB, allow 20% margin
        allocs = @allocated deeponet((u, y), ps, st)
        @test allocs < 5_000

        # Allocation count
        b = @benchmark $deeponet(($u, $y), $ps, $st) samples = 3 evals = 1
        @test b.allocs < 10
    end

    @testset "NOMAD allocation bounds" begin
        rng = StableRNG(12345)

        nomad = NOMAD(; approximator = (8, 32, 32, 16), decoder = (18, 16, 8, 8))
        ps, st = Lux.setup(rng, nomad)
        u = rand(Float32, 8, 5)
        y = rand(Float32, 2, 5)

        # Warmup
        nomad((u, y), ps, st)

        # Current baseline: ~3.2KB, allow 20% margin
        allocs = @allocated nomad((u, y), ps, st)
        @test allocs < 5_000

        # Allocation count
        b = @benchmark $nomad(($u, $y), $ps, $st) samples = 3 evals = 1
        @test b.allocs < 10
    end
end
