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
            ∂x_fd, ∂ps_fd = ∇sumabs2_reactant_fd(cno, x_ra, ps_ra, st_ra)
            ∂x_ra, ∂ps_ra = ∇sumabs2_reactant(cno, x_ra, ps_ra, st_ra)

            @test ∂x_fd ≈ ∂x_ra atol = 1.0f-1 rtol = 1.0f-1
            @test check_approx(∂ps_fd, ∂ps_ra; atol = 1.0f-1, rtol = 1.0f-1)
        end
    end
end
