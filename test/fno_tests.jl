@testitem "Fourier Neural Operator" setup = [SharedTestSetup] begin
    rng = StableRNG(12345)

    setups = [
        (
            modes = (4,),
            chs = (2, 4, 4, 4, 1),
            x_size = (8, 2, 2),
            y_size = (8, 1, 2),
            shift = false,
        ),
        (
            modes = (4, 4),
            chs = (2, 4, 4, 4, 4),
            x_size = (8, 8, 2, 2),
            y_size = (8, 8, 4, 2),
            shift = false,
        ),
        (
            modes = (4, 4),
            chs = (2, 4, 4, 4, 4),
            x_size = (8, 8, 2, 2),
            y_size = (8, 8, 4, 2),
            shift = true,
        ),
    ]

    xdev = reactant_device(; force = true)

    @testset "$(length(setup.modes))D | shift=$(setup.shift)" for setup in setups
        fno = FourierNeuralOperator(; setup.chs, setup.modes, setup.shift)
        display(fno)
        ps, st = Lux.setup(rng, fno)

        x = rand(rng, Float32, setup.x_size...)
        y = rand(rng, Float32, setup.y_size...)

        @test size(first(fno(x, ps, st))) == setup.y_size

        ps_ra, st_ra = (ps, st) |> xdev
        x_ra, y_ra = (x, y) |> xdev

        res = first(fno(x, ps, st))
        res_ra, _ = @jit fno(x_ra, ps_ra, st_ra)
        @test res_ra ≈ res atol = 1.0f-2 rtol = 1.0f-2

        @testset "check gradients" begin
            ∂x_fd, ∂ps_fd = ∇sumabs2_reactant_fd(fno, x_ra, ps_ra, st_ra)
            ∂x_ra, ∂ps_ra = ∇sumabs2_reactant(fno, x_ra, ps_ra, st_ra)

            @test ∂x_fd ≈ ∂x_ra atol = 1.0f-2 rtol = 1.0f-2
            @test check_approx(∂ps_fd, ∂ps_ra; atol = 1.0f-2, rtol = 1.0f-2)
        end
    end
end
