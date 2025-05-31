@testitem "NOMAD" setup = [SharedTestSetup] begin
    rng = StableRNG(12345)

    setups = [
        (
            u_size=(1, 5),
            y_size=(1, 5),
            out_size=(1, 5),
            approximator=(1, 16, 16, 15),
            decoder=(16, 8, 4, 1),
            name="Scalar",
        ),
        (
            u_size=(8, 5),
            y_size=(2, 5),
            out_size=(8, 5),
            approximator=(8, 32, 32, 16),
            decoder=(18, 16, 8, 8),
            name="Vector",
        ),
    ]

    xdev = reactant_device()

    @testset "$(setup.name)" for setup in setups
        u = rand(Float32, setup.u_size...)
        y = rand(Float32, setup.y_size...)
        nomad = NOMAD(; approximator=setup.approximator, decoder=setup.decoder)

        ps, st = Lux.setup(rng, nomad)

        pred = first(nomad((u, y), ps, st))
        @test setup.out_size == size(pred)

        ps_ra, st_ra = xdev((ps, st))
        u_ra, y_ra = xdev(u), xdev(y)

        @testset "check gradients" begin
            ∂u_zyg, ∂ps_zyg = zygote_gradient(nomad, (u, y), ps, st)

            ∂u_ra, ∂ps_ra = Reactant.with_config(;
                dot_general_precision=PrecisionConfig.HIGH,
                convolution_precision=PrecisionConfig.HIGH,
            ) do
                @jit enzyme_gradient(nomad, (u_ra, y_ra), ps_ra, st_ra)
            end
            ∂u_ra, ∂ps_ra = (∂u_ra, ∂ps_ra) |> cpu_device()

            @test ∂u_zyg[1] ≈ ∂u_ra[1] atol = 1.0f-3 rtol = 1.0f-3
            @test ∂u_zyg[2] ≈ ∂u_ra[2] atol = 1.0f-3 rtol = 1.0f-3
            @test check_approx(∂ps_zyg, ∂ps_ra; atol=1.0f-3, rtol=1.0f-3)
        end
    end
end
