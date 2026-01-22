@testitem "Fourier Neural Operator" setup = [SharedTestSetup] begin
    rng = StableRNG(12345)

    setups = [
        (
            modes = (16,),
            chs = (2, 64, 64, 64, 64, 64, 128, 1),
            x_size = (1024, 2, 5),
            y_size = (1024, 1, 5),
            shift = false,
        ),
        (
            modes = (16, 16),
            chs = (2, 64, 64, 64, 64, 64, 128, 4),
            x_size = (32, 32, 2, 5),
            y_size = (32, 32, 4, 5),
            shift = false,
        ),
        (
            modes = (16, 16),
            chs = (2, 64, 64, 64, 64, 64, 128, 4),
            x_size = (32, 32, 2, 5),
            y_size = (32, 32, 4, 5),
            shift = true,
        ),
    ]

    xdev = reactant_device(; force=true)

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

        # FIXME: re-enable this
        # @test begin
        #     l2, l1 = train!(
        #         MSELoss(), AutoEnzyme(), fno, ps_ra, st_ra, [(x_ra, y_ra)]; epochs = 10
        #     )
        #     l2 < l1
        # end

        # FIXME: https://github.com/EnzymeAD/Enzyme-JAX/issues/1961
        # @testset "check gradients" begin
        #     ∂x_fd, ∂ps_fd = ∇sumabs2_reactant_fd(fno, x_ra, ps_ra, st_ra)
        #     ∂x_ra, ∂ps_ra = ∇sumabs2_reactant(fno, x_ra, ps_ra, st_ra)

        #     @test ∂x_fd ≈ ∂x_ra atol = 1.0f-2 rtol = 1.0f-2
        #     @test check_approx(∂ps_fd, ∂ps_ra; atol = 1.0f-2, rtol = 1.0f-2)
        # end
    end
end
