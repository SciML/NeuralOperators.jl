@testitem "NOMAD" setup = [SharedTestSetup] begin
    rng = StableRNG(12345)

    setups = [
        (
            u_size = (1, 5),
            y_size = (1, 5),
            out_size = (1, 5),
            approximator = (1, 16, 16, 15),
            decoder = (16, 8, 4, 1),
            name = "Scalar",
        ),
        (
            u_size = (8, 5),
            y_size = (2, 5),
            out_size = (8, 5),
            approximator = (8, 32, 32, 16),
            decoder = (18, 16, 8, 8),
            name = "Vector",
        ),
    ]

    xdev = reactant_device(; force = true)

    @testset "$(setup.name)" for setup in setups
        u = rand(Float32, setup.u_size...)
        y = rand(Float32, setup.y_size...)
        nomad = NOMAD(; approximator = setup.approximator, decoder = setup.decoder)

        ps, st = Lux.setup(rng, nomad)

        pred = first(nomad((u, y), ps, st))
        @test setup.out_size == size(pred)

        ps_ra, st_ra = xdev((ps, st))
        u_ra, y_ra = xdev(u), xdev(y)

        pred_ra, _ = @jit nomad((u_ra, y_ra), ps_ra, st_ra)
        @test pred_ra ≈ pred atol = 1.0f-2 rtol = 1.0f-2

        @testset "check gradients" begin
            (∂u_fd, ∂y_fd), ∂ps_fd = ∇sumabs2_reactant_fd(nomad, (u_ra, y_ra), ps_ra, st_ra)
            (∂u_ra, ∂y_ra), ∂ps_ra = ∇sumabs2_reactant(nomad, (u_ra, y_ra), ps_ra, st_ra)

            @test ∂u_fd ≈ ∂u_ra atol = 1.0f-2 rtol = 1.0f-2
            @test ∂y_fd ≈ ∂y_ra atol = 1.0f-2 rtol = 1.0f-2
            @test check_approx(∂ps_fd, ∂ps_ra; atol = 1.0f-2, rtol = 1.0f-2)
        end
    end
end
