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

    xdev = reactant_device()

    @testset "$(setup.name)" for setup in setups
        u = rand(Float32, setup.u_size...)
        y = rand(Float32, setup.y_size...)
        deeponet = DeepONet(; branch = setup.branch, trunk = setup.trunk)

        ps, st = Lux.setup(rng, deeponet)

        pred = first(deeponet((u, y), ps, st))
        @test setup.out_size == size(pred)

        ps_ra, st_ra = (ps, st) |> xdev
        u_ra, y_ra = (u, y) |> xdev

        @testset "check gradients" begin
            ∂u_zyg, ∂ps_zyg = zygote_gradient(deeponet, (u, y), ps, st)

            ∂u_ra, ∂ps_ra = Reactant.with_config(;
                dot_general_precision = PrecisionConfig.HIGH,
                convolution_precision = PrecisionConfig.HIGH,
            ) do
                @jit enzyme_gradient(deeponet, (u_ra, y_ra), ps_ra, st_ra)
            end
            ∂u_ra, ∂ps_ra = (∂u_ra, ∂ps_ra) |> cpu_device()

            @test ∂u_zyg[1] ≈ ∂u_ra[1] atol = 1.0f-2 rtol = 1.0f-2
            @test ∂u_zyg[2] ≈ ∂u_ra[2] atol = 1.0f-2 rtol = 1.0f-2
            @test check_approx(∂ps_zyg, ∂ps_ra; atol = 1.0f-2, rtol = 1.0f-2)
        end
    end
end
