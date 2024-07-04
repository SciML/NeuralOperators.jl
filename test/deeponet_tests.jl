@testitem "DeepONet" setup=[SharedTestSetup] begin
    @testset "BACKEND: $(mode)" for (mode, aType, dev, ongpu) in MODES
        rng = StableRNG(12345)

        setups = [
            (u_size=(64, 5), y_size=(1, 10, 5), out_size=(10, 5),
                branch=(64, 32, 32, 16), trunk=(1, 8, 8, 16), name="Scalar"),
            (u_size=(64, 3, 5), y_size=(4, 10, 5), out_size=(10, 3, 5),
                branch=(64, 32, 32, 16), trunk=(1, 8, 8, 16), name="Vector"),
            (u_size=(64, 4, 3, 3, 5), y_size=(4, 10, 5), out_size=(10, 4, 3, 3, 5),
                branch=(64, 32, 32, 16), trunk=(1, 8, 8, 16), name="Tensor")]

        @testset "$(setup.name)" for setup in setups
            u = rand(Float32, setup.u_size...) |> aType
            y = rand(Float32, setup.y_size...) |> aType
            deeponet = DeepONet(; branch=setup.branch, trunk=setup.trunk)
            display(deeponet)
            ps, st = Lux.setup(rng, deeponet) |> dev

            @inferred first(deeponet((u, y), ps, st))
            @jet first(deeponet((u, y), ps, st))

            pred = first(deeponet(u, y), ps, st)
            @test size(setup.out_size) == size(pred)
        end

        @testset "Additonal layer" begin
            u = rand(Float32, 64, 1, 5) |> aType # sensor_points x nb
            y = rand(Float32, 1, 10, 5) |> aType # ndims x N x nb
            out_size = (4, 10, 5)

            branch_net = Chain(Dense(64 => 32), Dense(32 => 32), Dense(32 => 16))
            trunk_net = Chain(Dense(1 => 8), Dense(8 => 8), Dense(8 => 16))
            additional = Chain(Dense(1 => 4))
            deeponet = DeepONet(branch_net, trunk_net; additional=additional)
            display(deeponet)
            ps, st = Lux.setup(rng, deeponet)
            @inferred deeponet((u, y), ps, st)
            @jet deeponet((u, y), ps, st)

            pred = first(deeponet((u, y), ps, st))
            @test size(out_size) == size(pred)
        end

        @testset "Embedding layer mismatch" begin
            u = rand(Float32, 64, 5) |> aType
            y = rand(Float32, 1, 10, 5) |> aType

            deeponet = DeepONet(Chain(Dense(64 => 32), Dense(32 => 32), Dense(32 => 20)),
                Chain(Dense(1 => 8), Dense(8 => 8), Dense(8 => 16)))
                display(deeponet)
            ps, st = Lux.setup(rng, deeponet) |> dev
            @test_throws ArgumentError deeponet((u, y), ps, st)
        end
    end
end
