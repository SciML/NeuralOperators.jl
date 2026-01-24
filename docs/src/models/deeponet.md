# DeepONets

DeepONets are another class of networks that learn the mapping between two function spaces
by encoding the input function space and the location of the output space. The latent code
of the input space is then projected on the location laten code to give the output. This
allows the network to learn the mapping between two functions defined on different spaces.

```math
\begin{align*}
u(y) \xrightarrow{\text{branch}} & \; b \\
& \quad \searrow\\
&\quad \quad \mathcal{G}_{\theta} u(y) = \sum_k b_k t_k \\
&  \quad \nearrow \\
y \; \; \xrightarrow{\text{trunk}} \; \; & t
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

```@example deeponet_tutorial
using NeuralOperators, Lux, Random, Optimisers, Reactant

using CairoMakie, AlgebraOfGraphics
set_aog_theme!()
const AoG = AlgebraOfGraphics

rng = Random.default_rng()
Random.seed!(rng, 1234)

xdev = reactant_device()

eval_points = 17
batch_size = 64
dim_y = 1
m = 32

xrange = range(0, 2π; length=m) .|> Float32
α = 0.5f0 .+ 0.5f0 .* rand(Float32, batch_size)

u_data = zeros(Float32, m, batch_size)
y_data = rand(rng, Float32, dim_y, eval_points) .* Float32(2π)
# for plotting, we want to evaluate points in order
rightorder = sortperm(vec(y_data))

v_data = zeros(Float32, eval_points, batch_size)

for i in 1:batch_size
    u_data[:, i] .= sin.(α[i] .* xrange)
    v_data[:, i] .= -inv(α[i]) .* cos.(α[i] .* y_data[1, :])
end

deeponet = DeepONet(
    Chain(Dense(m => 64, tanh), Dense(64 => 64, tanh), Dense(64 => 64, tanh)),
    Chain(Dense(1 => 16, tanh), Dense(16 => 64, tanh))
)

ps, st = Lux.setup(rng, deeponet) |> xdev;

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

losses = train!(deeponet, ps, st, data; epochs=20000)

draw(
    AoG.data((; losses, iteration=1:length(losses))) *
    mapping(:iteration => "Iteration", :losses => "Loss (log10 scale)") *
    visual(Lines);
    axis=(; yscale=log10),
    figure=(; title="Using DeepONet to learn the anti-derivative operator")
)

# plot the prediction for a new function
# that's not part of the training set
αₜ = 0.75
input_data = sin.(αₜ .* xrange) |> xdev
output_data, st = @jit Lux.apply(deeponet, (input_data, y_data), ps, st)
output_x = vec(cdev(y_data))[rightorder]
pred_y = vec(cdev(output_data))[rightorder]
true_y = -inv(αₜ) .* cos.(αₜ .* y_data[1, rightorder])
p = lines(Array(xrange), Array(input_data); label="u")
lines!(a, Array(output_x), Array(pred_y); label="Predicted")
lines!(a, Array(output_x), Array(true_y); label="Expected")
axislegend(a)
# Compute the absolute error and plot that, too
absolute_error = abs.(Array(pred_y) .- Array(true_y))
a2, p2 = lines(f[2, 1], Array(output_x), absolute_error; axis=(; ylabel="Error"))
rowsize!(f.layout, 2, Aspect(1, 1 / 8))
linkxaxes!(a, a2)
f
```
