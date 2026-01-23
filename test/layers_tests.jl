@testitem "SpectralConv & SpectralKernel" setup = [SharedTestSetup] begin
    rng = StableRNG(12345)

    opconv = [SpectralConv, SpectralKernel]
    setups = [
        (; m = (4,), x_size = (8, 2, 2), y_size = (8, 4, 2), shift = false),
        (; m = (4, 4), x_size = (8, 8, 1, 2), y_size = (8, 8, 4, 2), shift = false),
        (; m = (4, 4), x_size = (8, 8, 1, 2), y_size = (8, 8, 4, 2), shift = true),
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

        @testset "check gradients" begin
            ∂x_fd, ∂ps_fd = ∇sumabs2_reactant_fd(m, x_ra, ps_ra, st_ra)
            ∂x_ra, ∂ps_ra = ∇sumabs2_reactant(m, x_ra, ps_ra, st_ra)

            @test ∂x_fd ≈ ∂x_ra atol = 1.0f-2 rtol = 1.0f-2
            @test check_approx(∂ps_fd, ∂ps_ra; atol = 1.0f-2, rtol = 1.0f-2)
        end
    end
end
