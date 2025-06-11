@testitem "Fourier Neural Operator" setup = [SharedTestSetup] begin
    rng = StableRNG(12345)

    setups = [
        (
            modes=(16,),
            chs=(2, 64, 64, 64, 64, 64, 128, 1),
            x_size=(1024, 2, 5),
            y_size=(1024, 1, 5),
            shift=false,
        ),
        (
            modes=(16, 16),
            chs=(2, 64, 64, 64, 64, 64, 128, 4),
            x_size=(32, 32, 2, 5),
            y_size=(32, 32, 4, 5),
            shift=false,
        ),
        (
            modes=(16, 16),
            chs=(2, 64, 64, 64, 64, 64, 128, 4),
            x_size=(32, 32, 2, 5),
            y_size=(32, 32, 4, 5),
            shift=true,
        ),
    ]

    @testset "$(length(setup.modes))D | shift=$(setup.shift)" for setup in setups
        fno = FourierNeuralOperator(; setup.chs, setup.modes, setup.shift)
        display(fno)
        ps, st = Lux.setup(rng, fno)

        x = rand(rng, Float32, setup.x_size...)
        y = rand(rng, Float32, setup.y_size...)

        @test size(first(fno(x, ps, st))) == setup.y_size

        ps_ra, st_ra = (ps, st) |> reactant_device()
        x_ra, y_ra = (x, y) |> reactant_device()

        res = first(fno(x, ps, st))
        res_ra, _ = Reactant.with_config(;
            dot_general_precision=PrecisionConfig.HIGH,
            convolution_precision=PrecisionConfig.HIGH,
        ) do
            @jit fno(x_ra, ps_ra, st_ra)
        end
        @test res_ra ≈ res atol = 1.0f-2 rtol = 1.0f-2

        @test begin
            l2, l1 = train!(
                MSELoss(), AutoEnzyme(), fno, ps_ra, st_ra, [(x_ra, y_ra)]; epochs=10
            )
            l2 < l1
        end

        @testset "check gradients" begin
            ∂x_zyg, ∂ps_zyg = zygote_gradient(fno, x, ps, st)

            ∂x_ra, ∂ps_ra = Reactant.with_config(;
                dot_general_precision=PrecisionConfig.HIGH,
                convolution_precision=PrecisionConfig.HIGH,
            ) do
                @jit enzyme_gradient(fno, x_ra, ps_ra, st_ra)
            end
            ∂x_ra, ∂ps_ra = (∂x_ra, ∂ps_ra) |> cpu_device()

            # TODO: is zygote off here?
            @test ∂x_zyg ≈ ∂x_ra atol = 1.0f-2 rtol = 1.0f-2 skip = setup.shift
            @test check_approx(∂ps_zyg, ∂ps_ra; atol=1.0f-2, rtol=1.0f-2) skip = setup.shift
        end
    end
end
