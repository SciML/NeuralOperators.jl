# Burgers Equation using DeepONet

## Data Loading

```@example burgers
using DataDeps, MAT, MLUtils
using PythonCall, CondaPkg # For `gdown`
using Printf

const gdown = pyimport("gdown")

register(
    DataDep(
    "Burgers",
    """
    Burgers' equation dataset from
    [fourier_neural_operator](https://github.com/zongyi-li/fourier_neural_operator)

    mapping between initial conditions to the solutions at the last point of time \
    evolution in some function space.

    u(x,0) -> u(x, time_end):

      * `a`: initial conditions u(x,0)
      * `u`: solutions u(x,t_end)
    """,
    "https://drive.google.com/uc?id=16a8od4vidbiNR3WtaBPCSZ0T3moxjhYe",
    "9cbbe5070556c777b1ba3bacd49da5c36ea8ed138ba51b6ee76a24b971066ecd";
    fetch_method=(url, local_dir) -> begin
        pyconvert(String, gdown.download(url, joinpath(local_dir, "Burgers_R10.zip")))
    end,
    post_fetch_method=unpack
)
)

filepath = joinpath(datadep"Burgers", "burgers_data_R10.mat")

const N = 2048
const Δsamples = 2^3
const grid_size = div(2^13, Δsamples)
const T = Float32

file = matopen(filepath)
x_data = reshape(T.(collect(read(file, "a")[1:N, 1:Δsamples:end])), N, :, 1)
y_data = reshape(T.(collect(read(file, "u")[1:N, 1:Δsamples:end])), N, :, 1)
close(file)

x_data = permutedims(x_data, (2, 1, 3))
grid = reshape(T.(collect(range(0, 1; length=grid_size)')), :, grid_size, 1)
```

## Model

```@example burgers
using Lux, NeuralOperators, Optimisers, Zygote, Random
using LuxCUDA

const cdev = cpu_device()
const gdev = gpu_device()

deeponet = DeepONet(;
    branch=(size(x_data, 1), ntuple(Returns(32), 5)...),
    trunk=(size(grid, 1), ntuple(Returns(32), 5)...),
    branch_activation=tanh,
    trunk_activation=tanh
)
ps, st = Lux.setup(Random.default_rng(), deeponet) |> gdev;
```

## Training

```@example burgers
x_data_dev = x_data |> gdev
y_data_dev = y_data |> gdev
grid_dev = grid |> gdev

function loss_function(model, ps, st, ((v, y), u))
    û, stₙ = model((v, y), ps, st)
    return MAELoss()(û, u), stₙ, (;)
end

begin
    train_state = Training.TrainState(deeponet, ps, st, Adam(0.0001f0))
    for epoch in 1:5000
        grads, loss, stats, train_state = Training.single_train_step!(
            AutoZygote(), loss_function, ((x_data_dev, grid_dev), y_data_dev), train_state)

        if epoch % 25 == 1 || epoch == 400
            @printf("Epoch %d: loss = %.6e\n", epoch, loss)
        end
    end
    ps_trained = train_state.parameters
    st_trained = train_state.states
end
```

## Plotting

```@example burgers
using CairoMakie

pred = first(deeponet((x_data_dev, grid_dev), ps_trained, st_trained)) |> cdev

begin
    fig = Figure(; size=(1024, 1024))

    axs = [Axis(fig[i, j]) for i in 1:4, j in 1:4]
    for i in 1:4, j in 1:4
        idx = i + (j - 1) * 4
        ax = axs[i, j]
        l1 = lines!(ax, vec(grid), pred[idx, :, 1])
        l2 = lines!(ax, vec(grid), y_data[idx, :, 1])

        i == 4 && (ax.xlabel = "x")
        j == 1 && (ax.ylabel = "u(x)")

        if i == 1 && j == 1
            axislegend(ax, [l1, l2], ["Predictions", "Ground Truth"])
        end
    end
    linkaxes!(axs...)

    fig[0, :] = Label(fig, "Burgers Equation using DeepONet"; tellwidth=false, font=:bold)

    fig
end
```
