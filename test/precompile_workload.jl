using Lux, NeuralOperators, Random, Test

@testset "Representative public workflows" begin
    rng = Xoshiro(0)

    fno = FourierNeuralOperator(
        (2,), 2, 1, 2; num_layers = 1, positional_embedding = :grid
    )
    ps, st = Lux.setup(rng, fno)
    @test size(first(fno(rand(rng, Float32, 8, 1, 1), ps, st))) == (8, 1, 1)

    deeponet = DeepONet(; branch = (4, 3), trunk = (1, 3))
    ps, st = Lux.setup(rng, deeponet)
    @test size(
        first(
            deeponet(
                (rand(rng, Float32, 4, 2), rand(rng, Float32, 1, 5)), ps, st
            ),
        ),
    ) == (5, 2)

    nomad = NOMAD(; approximator = (4, 3), decoder = (4, 2))
    ps, st = Lux.setup(rng, nomad)
    @test size(
        first(
            nomad(
                (rand(rng, Float32, 4, 2), rand(rng, Float32, 1, 2)), ps, st
            ),
        ),
    ) == (2, 2)
end
