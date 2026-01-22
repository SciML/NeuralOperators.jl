@testitem "SpectralConv & SpectralKernel" setup = [SharedTestSetup] begin
    rng = StableRNG(12345)

    opconv = [SpectralConv, SpectralKernel]
    setups = [
        (; m = (16,), x_size = (1024, 2, 5), y_size = (1024, 16, 5), shift = false),
        (; m = (10, 10), x_size = (22, 22, 1, 5), y_size = (22, 22, 16, 5), shift = false),
        (; m = (10, 10), x_size = (22, 22, 1, 5), y_size = (22, 22, 16, 5), shift = true),
    ]

    xdev = reactant_device(; force = true)

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

        ps_ra, st_ra = xdev((ps, st))
        x_ra = xdev(x)
        y_ra = xdev(rand(rng, Float32, setup.y_size...))

        res_ra, _ = @jit m(x_ra, ps_ra, st_ra)
        @test res_ra ≈ res atol = 1.0f-2 rtol = 1.0f-2

        # FIXME: re-enable this
        # @test begin
        #     l2, l1 = train!(
        #         MSELoss(), AutoEnzyme(), m, ps_ra, st_ra, [(x_ra, y_ra)]; epochs = 10
        #     )
        #     l2 < l1
        # end

        # FIXME: https://github.com/EnzymeAD/Enzyme-JAX/issues/1961
        # @testset "check gradients" begin
        #     ∂x_fd, ∂ps_fd = ∇sumabs2_reactant_fd(m, x_ra, ps_ra, st_ra)
        #     ∂x_ra, ∂ps_ra = ∇sumabs2_reactant(m, x_ra, ps_ra, st_ra)

        #     @test ∂x_fd ≈ ∂x_ra atol = 1.0f-2 rtol = 1.0f-2
        #     @test check_approx(∂ps_fd, ∂ps_ra; atol = 1.0f-2, rtol = 1.0f-2)
        # end
    end
end
