# DeepONet for 1D Poisson Equation

## Mathematical Formulation

### Problem Statement

We consider the one-dimensional Poisson equation on the domain $\Omega = [0,1]$:

$$-\frac{d^2u(x)}{dx^2} = f(x), \quad x \in [0,1]$$

subject to homogeneous Dirichlet boundary conditions:

$$u(0) = u(1) = 0$$

In our specific case, the forcing function $f(x)$ is parameterized by $\alpha$:

$$f(x) = \alpha \sin(\pi x)$$

### Analytical Solution

$$u(x) = \frac{\alpha}{\pi^2} \sin(\pi x)$$

The DeepONet architecture consists of:

- **Branch network**: Processes the discretized input function $f(x)$
- **Trunk network**: Processes the spatial coordinates $x$
- **Output**: $u(x) \approx \sum_{k=1}^p b_k(f) t_k(x)$, where $b_k$ are outputs from the
  branch network and $t_k$ are outputs from the trunk network

## Implementation

```@example poisson_deeponet
using NeuralOperators, Lux, Random, Optimisers, Reactant, Statistics

using CairoMakie, AlgebraOfGraphics
const AoG = AlgebraOfGraphics
AoG.set_aog_theme!()

const cdev = cpu_device()
const xdev = reactant_device(; force=true)

rng = Random.default_rng()
Random.seed!(rng, 42)

# Problem setup
m = 64
data_size = 128
xrange = Float32.(range(0, 1; length=m))

forcing_function(x, α) = α * sinpi.(x)
poisson_solution(x, α) = (α / Float32(π)^2) * sinpi.(x)

# Generate training data
α_train = 0.5f0 .+ 0.8f0 .* rand(Float32, data_size)  # α ∈ [0.5, 1.3]
f_data = stack(Base.Fix1(forcing_function, xrange), α_train)
u_data = stack(Base.Fix1(poisson_solution, xrange), α_train)
x_data = reshape(xrange, 1, m)
max_u = maximum(abs.(u_data))
u_data ./= max_u

# Define DeepONet
deeponet = DeepONet(
    Chain(Dense(m => 64, tanh), Dense(64 => 64, tanh), Dense(64 => 32, tanh)),  # Branch
    Chain(Dense(1 => 32, tanh), Dense(32 => 32, tanh))                          # Trunk
)

ps, st = Lux.setup(Random.default_rng(), deeponet) |> xdev;

# Training
function train_model!(model, ps, st, data; epochs=3000)
    train_state = Training.TrainState(model, ps, st, Adam(0.001f0))
    losses = Float32[]

    for epoch in 1:epochs
        _, loss, _, train_state = Training.single_train_step!(
            AutoEnzyme(), MSELoss(), data, train_state; return_gradients=Val(false)
        )
        epoch % 500 == 0 && println("Epoch: $epoch, Loss: $loss")
        push!(losses, loss)
    end

    return train_state.parameters, train_state.states, losses
end

f_data = f_data |> xdev;
x_data = x_data |> xdev;
u_data = u_data |> xdev;
data = ((f_data, x_data), u_data)
ps_trained, st_trained, losses = train_model!(deeponet, ps, st, data)

# Prediction function
function predict(model, f_input, x_input, ps, st)
    pred, _ = model((f_input, x_input), ps, st)
    return vec(pred .* max_u)
end

compiled_predict_fn = Reactant.with_config(; dot_general_precision=PrecisionConfig.HIGH) do
    @compile predict(deeponet, f_data[:, 1:1], x_data, ps_trained, st_trained)
end

# Testing and visualization
begin
    results = Float32[]
    labels = AbstractString[]
    abs_errors = Float32[]
    x_values = Float32[]
    x_values2 = Float32[]
    alphas = AbstractString[]
    alphas2 = AbstractString[]

    for (i, α) in enumerate([0.6f0, 0.8f0, 1.2f0])
        f_test = reshape(forcing_function(xrange, α), :, 1)
        u_pred = compiled_predict_fn(
            deeponet, xdev(f_test), x_data, ps_trained, st_trained
        ) |> cdev
        u_true = reshape(poisson_solution(xrange, α), :, 1)

        l2_error = sqrt(mean(abs2, u_pred .- u_true))
        rel_error = l2_error / sqrt(mean(abs2, u_true)) * 100

        text = L"$ \alpha = %$(α) $ (Rel. Error: $ %$(round(rel_error, digits=2))% $)"

        append!(results, vec(u_pred))
        append!(labels, repeat(["Predictions"], length(xrange)))
        append!(results, vec(u_true))
        append!(labels, repeat(["Ground Truth"], length(xrange)))
        append!(x_values, repeat(vec(xrange), 2))
        append!(alphas, repeat([text], length(xrange) * 2))

        append!(abs_errors, abs.(vec(u_pred .- u_true)))
        append!(x_values2, vec(xrange))
        append!(alphas2, repeat([text], length(xrange)))
    end
end

plot_data = (; results, abs_errors, x_values, alphas, labels, x_values2, alphas2)

begin
    fig = Figure(;
        size=(1024, 512),
        title="DeepONet Results for 1D Poisson Equation",
        titlesize=25
    )

    axis_common = (;
        xlabelsize=20, ylabelsize=20, titlesize=20, xticklabelsize=20, yticklabelsize=20
    )

    axs1 = draw!(
        fig[1, 1],
        AoG.data(plot_data) *
        mapping(
            :x_values => L"x",
            :results => L"u(x)";
            color=:labels => "",
            col=:alphas => "",
            linestyle=:labels => "",
        ) *
        visual(Lines; linewidth=4),
        scales(; Color=(; palette=[:orange, :blue]), LineStyle = (; palette = [:solid, :dash]));
        axis=merge(axis_common, (; xlabel="")),
    )
    for ax in axs1
        hidexdecorations!(ax; grid=false)
    end

    axislegend(
        axs1[1, 1].axis,
        [
            LineElement(; linestyle=:solid, color=:orange),
            LineElement(; linestyle=:dash, color=:blue),
        ],
        ["Ground Truth", "Predictions"],
        labelsize=20,
    )

    axs2 = draw!(
        fig[2, 1],
        AoG.data(plot_data) *
        mapping(
            :x_values2 => L"x",
            :abs_errors => L"|u(x) - u(x_{true})|";
            col=:alphas2 => "",
        ) *
        visual(Lines; linewidth=4, color=:green);
        axis=merge(axis_common, (; titlevisible=false)),
    )

    fig
end
```
