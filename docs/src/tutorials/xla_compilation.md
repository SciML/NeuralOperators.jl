# Compiling NeuralOperators.jl using Reactant.jl

```@example xla_compilation
using NeuralOperators, Lux, Random, Enzyme, Reactant

function sumabs2first(model, ps, st, x)
    z, _ = model(x, ps, st)
    return sum(abs2, z)
end

dev = reactant_device()
```

## Compiling DeepONet

```@example xla_compilation
deeponet = DeepONet()
ps, st = Lux.setup(Random.default_rng(), deeponet) |> dev;

u = rand(Float32, 64, 1024) |> dev;
y = rand(Float32, 1, 128, 1024) |> dev;
nothing # hide

deeponet_compiled = @compile deeponet((u, y), ps, st)
deeponet_compiled((u, y), ps, st)[1]
```

Computing the gradient of the DeepONet model.

```@example xla_compilation
function ∇deeponet(model, ps, st, (u, y))
    dps = Enzyme.make_zero(ps)
    Enzyme.autodiff(
        Enzyme.Reverse,
        sumabs2first,
        Const(model),
        Duplicated(ps, dps),
        Const(st),
        Const((u, y))
    )
    return dps
end

∇deeponet_compiled = @compile ∇deeponet(deeponet, ps, st, (u, y))
∇deeponet_compiled(deeponet, ps, st, (u, y))
```

## Compiling FourierNeuralOperator

```@example xla_compilation
fno = FourierNeuralOperator()
ps, st = Lux.setup(Random.default_rng(), fno) |> dev;

x = rand(Float32, 2, 1024, 5) |> dev;

fno_compiled = @compile fno(x, ps, st)
fno_compiled(x, ps, st)[1]
```

Computing the gradient of the FourierNeuralOperator model.

```@example xla_compilation
function ∇fno(model, ps, st, x)
    dps = Enzyme.make_zero(ps)
    Enzyme.autodiff(
        Enzyme.Reverse,
        sumabs2first,
        Const(model),
        Duplicated(ps, dps),
        Const(st),
        Const(x)
    )
    return dps
end

∇fno_compiled = @compile ∇fno(fno, ps, st, x)
∇fno_compiled(fno, ps, st, x)
```
