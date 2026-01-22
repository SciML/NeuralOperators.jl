@testitem "DeepONet" setup = [SharedTestSetup] begin
    rng = StableRNG(12345)

    setups = [
        (
            u_size = (64, 5),
            y_size = (1, 10),
            out_size = (10, 5),
            branch = (64, 32, 32, 16),
            trunk = (1, 8, 8, 16),
            name = "Scalar",
        ),
        (
            u_size = (64, 5),
            y_size = (4, 10),
            out_size = (10, 5),
            branch = (64, 32, 32, 16),
            trunk = (4, 8, 8, 16),
            name = "Vector",
        ),
    ]

    xdev = reactant_device(; force=true)

    @testset "$(setup.name)" for setup in setups
        u = rand(Float32, setup.u_size...)
        y = rand(Float32, setup.y_size...)
        deeponet = DeepONet(; branch = setup.branch, trunk = setup.trunk)

        ps, st = Lux.setup(rng, deeponet)

        pred = first(deeponet((u, y), ps, st))
        @test setup.out_size == size(pred)

        ps_ra, st_ra = (ps, st) |> xdev
        u_ra, y_ra = (u, y) |> xdev

        pred_ra = @jit deeponet((u_ra, y_ra), ps_ra, st_ra)
        @test first(pred_ra) ≈ pred atol = 1.0f-2 rtol = 1.0f-2

        @testset "check gradients" begin
            (∂u_fd, ∂y_fd), ∂ps_fd = ∇sumabs2_reactant_fd(
                deeponet, (u_ra, y_ra), ps_ra, st_ra
            )
            (∂u_ra, ∂y_ra), ∂ps_ra = ∇sumabs2_reactant(deeponet, (u_ra, y_ra), ps_ra, st_ra)

            @test ∂u_fd ≈ ∂u_ra atol = 1.0f-2 rtol = 1.0f-2
            @test ∂y_fd ≈ ∂y_ra atol = 1.0f-2 rtol = 1.0f-2
            @test check_approx(∂ps_fd, ∂ps_ra; atol = 1.0f-2, rtol = 1.0f-2)
        end
    end
end
