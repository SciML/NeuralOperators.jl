using NeuralOperators, Test

include("../shared_testsetup.jl")

@testset "Fourier Neural Operator" begin
    rng = StableRNG(12345)

    @testset "complex finite-difference reference" begin
        parameters = (; complex = ComplexF64[1 + 2im, -3 + 4im], real = [5.0, -6.0])
        gradient = finite_difference_structure(parameters) do ps
            return sum(abs2, ps.complex) + sum(abs2, ps.real)
        end
        @test gradient.complex ≈ 2 .* parameters.complex
        @test gradient.real ≈ 2 .* parameters.real
    end

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
            ∂x_fd, ∂ps_fd = ∇sumabs2_finite_difference(fno, x, ps, st)
            ∂x_ra, ∂ps_ra = ∇sumabs2_reactant(fno, x_ra, ps_ra, st_ra)
            ∂x_ra, ∂ps_ra = (∂x_ra, ∂ps_ra) |> cpu_device()

            @test ∂x_fd ≈ ∂x_ra atol = 1.0f-2 rtol = 1.0f-2
            @test check_approx(∂ps_fd, ∂ps_ra; atol = 1.0f-2, rtol = 1.0f-2)
        end
    end
end
