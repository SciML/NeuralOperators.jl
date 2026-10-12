using NeuralOperators, Test

include("../shared_testsetup.jl")

@testset "Convolutional Neural Operator" begin
    rng = StableRNG(12345)

    setups = [
        (
            modes = (4,),
            in_channels = 1,
            out_channels = 1,
            hidden_channels = 8,
            num_layers = 2,
            x_size = (16, 1, 4),
            y_size = (16, 1, 4),
        ),
        (
            modes = (4, 4),
            in_channels = 2,
            out_channels = 1,
            hidden_channels = 8,
            num_layers = 2,
            x_size = (16, 16, 2, 4),
            y_size = (16, 16, 1, 4),
        ),
    ]

    xdev = reactant_device(; force = true)

    @testset "$(length(setup.modes))D" for setup in setups
        cno = ConvolutionalNeuralOperator(
            setup.modes, setup.in_channels, setup.out_channels, setup.hidden_channels;
            num_layers = setup.num_layers,
        )
        display(cno)
        ps, st = Lux.setup(rng, cno)

        x = rand(rng, Float32, setup.x_size...)
        y = rand(rng, Float32, setup.y_size...)

        @test size(first(cno(x, ps, st))) == setup.y_size

        ps_ra, st_ra = (ps, st) |> xdev
        x_ra, y_ra = (x, y) |> xdev

        res = first(cno(x, ps, st))
        res_ra, _ = @jit cno(x_ra, ps_ra, st_ra)
        @test res_ra ≈ res atol = 1.0f-2 rtol = 1.0f-2

        @testset "check gradients" begin
            ∂x_fd, ∂ps_fd = ∇sumabs2_finite_difference(cno, x, ps, st)
            ∂x_ra, ∂ps_ra = ∇sumabs2_reactant(cno, x_ra, ps_ra, st_ra)
            ∂x_ra, ∂ps_ra = (∂x_ra, ∂ps_ra) |> cpu_device()

            @test ∂x_fd ≈ ∂x_ra atol = 1.0f-1 rtol = 1.0f-1
            @test check_approx(∂ps_fd, ∂ps_ra; atol = 1.0f-1, rtol = 1.0f-1)
        end
    end

    @testset "activation does not alias" begin
        block = NeuralOperators.CNOBlock(1, 1, (4,), abs2)
        ps, st = Lux.setup(rng, block)
        ps.layer_1.weight .= 0
        ps.layer_1.weight[2, 1, 1] = 1
        ps.layer_1.bias .= 0

        # cos² of frequency 12 is DC plus frequency 24, which must be dropped, not folded
        # back to frequency 8, where the grid cannot represent it
        for n in (32, 64, 128)
            x = reshape(Float32[cos(2π * 12 * j / n) for j in 0:(n - 1)], n, 1, 1)
            y = vec(first(block(x, ps, st)))
            @test sum(y) / n ≈ 0.5 atol = 1.0f-5
            @test 2abs(sum(y .* cis.(-2π * 8 * (0:(n - 1)) / n))) / n < 1.0f-5
        end
    end

    @testset "spectral resampling" begin
        f(x, y) = cos(2π * (3x + 5y)) + sin(2π * (-4x + 2y))
        samples(n) = reshape([f(i / n[1], j / n[2]) for i in 0:(n[1] - 1), j in 0:(n[2] - 1)], n..., 1, 1)

        for (n, m) in (((16, 12), (32, 36)), ((15, 11), (30, 22)), ((16, 12), (13, 17)))
            y = NeuralOperators.spectral_resample(samples(n), m)
            @test y ≈ samples(m)
            @test NeuralOperators.spectral_resample(y, n) ≈ samples(n)
        end
    end
end
