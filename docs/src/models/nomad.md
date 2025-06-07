# Nonlinear Manifold Decoders for Operator Learning (NOMADs)

NOMADs are similar to DeepONets in the aspect that they can learn when the input and output
function spaces are defined on different domains. Their architecture is different and use
nonlinearity to the latent codes to obtain the operator approximation. The architecture
involves an approximator to encode the input function space, which is directly concatenated
with the input function coordinates, and passed into a decoder net to give the output
function at the given coordinate.

```math
\begin{align*}
u(y) \xrightarrow{\mathcal{A}} & \; \beta \\
& \quad \searrow\\
& \quad \quad \mathcal{G}_{\theta} u(y) = \mathcal{D}(\beta, y) \\
& \quad \nearrow \\
y
\end{align*}
```

## Usage

Let's try to learn the anti-derivative operator for

```math
u(x) = sin(\alpha x)
```

That is, we want to learn

```math
\mathcal{G} : u \rightarrow v \\
```

such that

```math
v(x) = \frac{du}{dx} \quad \forall \; x \in [0, 2\pi], \; \alpha \in [0.5, 1]
```

### Copy-pastable code

```@example nomad_tutorial
using NeuralOperators, Lux, Random, Optimisers, Reactant

using CairoMakie, AlgebraOfGraphics
set_aog_theme!()
const AoG = AlgebraOfGraphics

rng = Random.default_rng()
Random.seed!(rng, 1234)

xdev = reactant_device()

eval_points = 1
batch_size = 64
dim_y = 1
m = 32

xrange = range(0, 2π; length=m) .|> Float32
α = 0.5f0 .+ 0.5f0 .* rand(Float32, batch_size)

u_data = zeros(Float32, m, batch_size)
y_data = rand(rng, Float32, eval_points, batch_size) .* Float32(2π)
v_data = zeros(Float32, eval_points, batch_size)

for i in 1:batch_size
    u_data[:, i] .= sin.(α[i] .* xrange)
    v_data[:, i] .= -inv(α[i]) .* cos.(α[i] .* y_data[:, i])
end

nomad = NOMAD(
    Chain(Dense(m => 8, σ), Dense(8 => 8, σ), Dense(8 => 8 - eval_points)),
    Chain(Dense(8 => 4, σ), Dense(4 => eval_points))
)

ps, st = Lux.setup(rng, nomad) |> xdev;
u_data = u_data |> xdev;
y_data = y_data |> xdev;
v_data = v_data |> xdev;
data = [((u_data, y_data), v_data)];

function train!(model, ps, st, data; epochs=10)
    losses = []
    tstate = Training.TrainState(model, ps, st, Adam(0.001f0))
    for _ in 1:epochs, (x, y) in data
        (_, loss, _, tstate) = Training.single_train_step!(
            AutoEnzyme(), MSELoss(), (x, y), tstate; return_gradients=Val(false)
        )
        push!(losses, Float32(loss))
    end
    return losses
end

losses = train!(nomad, ps, st, data; epochs=1000)

draw(
    AoG.data((; losses, iteration=1:length(losses))) *
    mapping(:iteration => "Iteration", :losses => "Loss (log10 scale)") *
    visual(Lines);
    axis=(; yscale=log10),
    figure=(; title="Using NOMAD to learn the anti-derivative operator")
)
```
