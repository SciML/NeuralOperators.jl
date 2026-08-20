@setup_workload begin
    @compile_workload begin
        rng = Random.Xoshiro(0)

        fno = FourierNeuralOperator(
            (2,), 2, 1, 2; num_layers = 1, positional_embedding = :grid
        )
        fno_ps, fno_st = Lux.setup(rng, fno)
        x = rand(rng, Float32, 8, 1, 1)
        first(fno(x, fno_ps, fno_st))

        deeponet = DeepONet(; branch = (4, 3), trunk = (1, 3))
        deeponet_ps, deeponet_st = Lux.setup(rng, deeponet)
        first(
            deeponet(
                (rand(rng, Float32, 4, 2), rand(rng, Float32, 1, 5)),
                deeponet_ps,
                deeponet_st,
            ),
        )

        nomad = NOMAD(; approximator = (4, 3), decoder = (4, 2))
        nomad_ps, nomad_st = Lux.setup(rng, nomad)
        first(
            nomad(
                (rand(rng, Float32, 4, 2), rand(rng, Float32, 1, 2)),
                nomad_ps,
                nomad_st,
            ),
        )

        transform_type = FourierTransform{ComplexF32}((2,))
        transformed = transform(transform_type, x)
        inverse(transform_type, transformed, x)
    end
end
