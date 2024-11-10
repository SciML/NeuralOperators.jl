# Compiling NeuralOperators.jl using Reactant.jl

```@example reactant
using NeuralOperators, Lux, Random, Enzyme, Reactant

function sumabs2first(model, ps, st, x)
    z, _ = model(x, ps, st)
    return sum(abs2, z)
end

dev = reactant_device()
```

## Compiling DeepONet

```@example reactant
deeponet = DeepONet()
ps, st = Lux.setup(Random.default_rng(), deeponet) |> dev;

u = rand(Float32, 64, 32) |> dev;
y = rand(Float32, 1, 128, 32) |> dev;
nothing # hide

deeponet_compiled = @compile deeponet((u, y), ps, st)
deeponet_compiled((u, y), ps, st)[1]
```

Computing the gradient of the DeepONet model.

```@example reactant
function ∇deeponet(model, ps, st, (u, y))
    return Enzyme.gradient(
        Enzyme.Reverse, Const(sumabs2first), Const(model), ps, Const(st), Const((u, y))
    )
end

∇deeponet_compiled = @compile ∇deeponet(deeponet, ps, st, (u, y))
∇deeponet_compiled(deeponet, ps, st, (u, y))
```

## Compiling FourierNeuralOperator

```@example reactant
fno = FourierNeuralOperator()
ps, st = Lux.setup(Random.default_rng(), fno) |> dev;

x = rand(Float32, 2, 32, 5) |> dev;

fno_compiled = @compile fno(x, ps, st)
fno_compiled(x, ps, st)[1]
```

Computing the gradient of the FourierNeuralOperator model.

```@example reactant
function ∇fno(model, ps, st, x)
    return Enzyme.gradient(
        Enzyme.Reverse, Const(sumabs2first), Const(model), ps, Const(st), Const(x)
    )
end

∇fno_compiled = @compile ∇fno(fno, ps, st, x)
∇fno_compiled(fno, ps, st, x)
```
