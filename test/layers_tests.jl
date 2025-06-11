@testitem "SpectralConv & SpectralKernel" setup = [SharedTestSetup] begin
    rng = StableRNG(12345)

    opconv = [SpectralConv, SpectralKernel]
    setups = [
        (; m=(16,), x_size=(1024, 2, 5), y_size=(1024, 16, 5), shift=false),
        (; m=(10, 10), x_size=(22, 22, 1, 5), y_size=(22, 22, 16, 5), shift=false),
        (; m=(10, 10), x_size=(22, 22, 1, 5), y_size=(22, 22, 16, 5), shift=true),
    ]

    rdev = reactant_device()

    @testset "$(op) $(length(setup.m))D | shift=$(setup.shift)" for op in opconv,
        setup in setups

        in_chs = setup.x_size[end - 1]
        out_chs = setup.y_size[end - 1]
        ch = 4 => out_chs

        l1 = Conv(ntuple(_ -> 1, length(setup.m)), in_chs => first(ch))
        m = Chain(l1, op(ch, setup.m; setup.shift))
        display(m)
        ps, st = Lux.setup(rng, m)

        x = rand(rng, Float32, setup.x_size...)
        @test size(first(m(x, ps, st))) == setup.y_size
        res = first(m(x, ps, st))

        ps_ra, st_ra = rdev((ps, st))
        x_ra = rdev(x)
        y_ra = rdev(rand(rng, Float32, setup.y_size...))

        res_ra, _ = Reactant.with_config(;
            dot_general_precision=PrecisionConfig.HIGH,
            convolution_precision=PrecisionConfig.HIGH,
        ) do
            @jit m(x_ra, ps_ra, st_ra)
        end
        @test res_ra ≈ res atol = 1.0f-2 rtol = 1.0f-2

        @test begin
            l2, l1 = train!(
                MSELoss(), AutoEnzyme(), m, ps_ra, st_ra, [(x_ra, y_ra)]; epochs=10
            )
            l2 < l1
        end

        @testset "check gradients" begin
            ∂x_zyg, ∂ps_zyg = zygote_gradient(m, x, ps, st)

            ∂x_ra, ∂ps_ra = Reactant.with_config(;
                dot_general_precision=PrecisionConfig.HIGH,
                convolution_precision=PrecisionConfig.HIGH,
            ) do
                @jit enzyme_gradient(m, x_ra, ps_ra, st_ra)
            end
            ∂x_ra, ∂ps_ra = (∂x_ra, ∂ps_ra) |> cpu_device()

            @test ∂x_zyg ≈ ∂x_ra atol = 1.0f-2 rtol = 1.0f-2
            # TODO: is zygote off here?
            @test check_approx(∂ps_zyg, ∂ps_ra; atol=1.0f-2, rtol=1.0f-2) broken =
                setup.shift
        end
    end
end
