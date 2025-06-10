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
-  **Trunk network**: Processes the spatial coordinates $x$
-  **Output**: $u(x) \approx \sum_{k=1}^p b_k(f) t_k(x)$, where $b_k$ are outputs from the branch network and $t_k$ are outputs from the trunk network

## Implementation

```julia
using NeuralOperators, Lux, Random, Optimisers, Zygote
using BSON, Plots, LinearAlgebra, Statistics

Random.seed!(42)

# Problem setup
m = 64                  
data_size = 100        
xrange = Float32.(range(0, 1; length=m))

forcing_function(x, α) = α * sin.(π * x)
poisson_solution(x, α) = (α / π^2) * sin.(π * x)

# Generate training data
α_train = 0.5f0 .+ 0.8f0 .* rand(Float32, data_size)  # α ∈ [0.5, 1.3]
f_data = zeros(Float32, m, data_size)
u_data = zeros(Float32, m, data_size)

for i in 1:data_size
    f_data[:, i] .= forcing_function(xrange, α_train[i])
    u_data[:, i] .= poisson_solution(xrange, α_train[i])
end

x_data = repeat(reshape(xrange, 1, m, 1), 1, 1, data_size)
max_u = maximum(abs.(u_data))
u_data ./= max_u

# Define DeepONet
deeponet = DeepONet(
    Chain(Dense(m => 64, tanh), Dense(64 => 64, tanh), Dense(64 => 32, tanh)),  # Branch
    Chain(Dense(1 => 32, tanh), Dense(32 => 32, tanh))                          # Trunk
)

ps, st = Lux.setup(Random.default_rng(), deeponet)

# Training
function train_model!(model, ps, st, data; epochs=3000)
    train_state = Training.TrainState(model, ps, st, Adam(0.001f0))
    losses = Float32[]
    
    for epoch in 1:epochs
        _, loss, _, train_state = Training.single_train_step!(
            AutoZygote(), MSELoss(), data, train_state
        )
        epoch % 500 == 0 && println("Epoch: $epoch, Loss: $loss")
        push!(losses, loss)
    end
    
    return train_state.parameters, train_state.states, losses
end

println("Training DeepONet...")
data = ((f_data, x_data), u_data)
ps_trained, st_trained, losses = train_model!(deeponet, ps, st, data)

# Prediction function
function predict(model, f_input, x_input, ps, st)
    f_input = reshape(Float32.(f_input), :, 1)
    x_input = reshape(Float32.(x_input), 1, :, 1)
    pred, _ = model((f_input, x_input), ps, st)
    return vec(pred .* max_u)
end

# Testing and visualization
test_alphas = [0.6f0, 0.8f0, 1.2f0]
plots_array = []

for (i, α) in enumerate(test_alphas)
    f_test = forcing_function(xrange, α)
    u_pred = predict(deeponet, f_test, xrange, ps_trained, st_trained)
    u_true = poisson_solution(xrange, α)
    
    l2_error = sqrt(mean((u_pred .- u_true).^2))
    rel_error = l2_error / sqrt(mean(u_true.^2)) * 100
    
    # Solution comparison
    plt1 = plot(xrange, u_true, label="Analytical", lw=2, color=:blue,
                title="α = $α (Rel. Error: $(round(rel_error, digits=2))%)")
    plot!(plt1, xrange, u_pred, label="DeepONet", lw=2, color=:red, ls=:dash)
    
    # Error plot
    plt2 = plot(xrange, u_pred .- u_true, lw=2, color=:green, legend=false,
                title="Absolute Error", xlabel="x")
    
    push!(plots_array, plot(plt1, plt2, layout=(2,1), size=(400,400)))
    
    println("α = $α: L2 Error = $(round(l2_error, digits=6)), Rel. Error = $(round(rel_error, digits=2))%")
end

# Combined plot
final_plot = plot(plots_array..., layout=(1,3), size=(1200,400),
                 plot_title="DeepONet Results for 1D Poisson Equation")
display(final_plot)
savefig(final_plot, "deeponet_poisson_results.png")

# Training loss
loss_plot = plot(losses, xlabel="Epoch", ylabel="Loss", title="Training Loss",
                yscale=:log10, lw=2, legend=false, size=(600,400))
display(loss_plot)

# Save model
BSON.@save "deeponet_poisson.bson" deeponet ps_trained st_trained max_u

println("Training completed! Model saved as 'deeponet_poisson.bson'")
```


