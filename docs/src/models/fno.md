# Fourier Neural Operators (FNOs)

FNOs are a subclass of Neural Operators that learn the learn the kernel $\Kappa_{\theta}$,
parameterized on $\theta$ between function spaces:

```math
(\Kappa_{\theta}u)(x) = \int_D \kappa_{\theta}(a(x), a(y), x, y) dy  \quad \forall x \in D
```

The kernel makes up a block $v_t(x)$ which passes the information to the next block as:

```math
v^{(t+1)}(x) = \sigma((W^{(t)}v^{(t)} + \Kappa^{(t)}v^{(t)})(x))
```

FNOs choose a specific kernel $\kappa(x,y) = \kappa(x-y)$, converting the kernel into a
convolution operation, which can be efficiently computed in the fourier domain.

```math
\begin{align*}
(\Kappa_{\theta}u)(x)
&= \int_D \kappa_{\theta}(x - y) dy  \quad \forall x \in D\\
&= \mathcal{F}^{-1}(\mathcal{F}(\kappa_{\theta}) \mathcal{F}(u))(x) \quad \forall x \in D
\end{align*}
```

where $\mathcal{F}$ denotes the fourier transform. Usually, not all the modes in the
frequency domain are used with the higher modes often being truncated.

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

```@raw html
<details>

<summary> Click here to see copy-pastable code for this example! </summary>

```

```@example fno_tutorial
using NeuralOperators, Lux, Random, Optimisers, Reactant

using CairoMakie, AlgebraOfGraphics
set_aog_theme!()
const AoG = AlgebraOfGraphics

rng = Random.default_rng()
Random.seed!(rng, 1234)

xdev = reactant_device()

batch_size = 128
m = 32

xrange = range(0, 2π; length=m) .|> Float32;
u_data = zeros(Float32, m, 1, batch_size);
α = 0.5f0 .+ 0.5f0 .* rand(Float32, batch_size);
v_data = zeros(Float32, m, 1, batch_size);

for i in 1:batch_size
    u_data[:, 1, i] .= sin.(α[i] .* xrange)
    v_data[:, 1, i] .= -inv(α[i]) .* cos.(α[i] .* xrange)
end

fno = FourierNeuralOperator(gelu; chs=(1, 64, 64, 128, 1), modes=(16,))

ps, st = Lux.setup(rng, fno) |> xdev;
u_data = u_data |> xdev;
v_data = v_data |> xdev;
data = [(u_data, v_data)];

function train!(model, ps, st, data; epochs=10)
    losses = []
    tstate = Training.TrainState(model, ps, st, Adam(0.003f0))
    for _ in 1:epochs, (x, y) in data
        (_, loss, _, tstate) = Training.single_train_step!(
            AutoEnzyme(), MSELoss(), (x, y), tstate; return_gradients=Val(false)
        )
        push!(losses, Float32(loss))
    end
    return losses
end

losses = train!(fno, ps, st, data; epochs=1000)

draw(
    AoG.data((; losses, iteration=1:length(losses))) *
    mapping(:iteration => "Iteration", :losses => "Loss (log10 scale)") *
    visual(Lines);
    axis=(; yscale=log10),
    figure=(; title="Using Fourier Neural Operator to learn the anti-derivative operator")
)
```

```@raw html
</details>
```

```@example fno_tutorial_details
using NeuralOperators, Lux, Random, Optimisers, Reactant
```

We will use Reactant.jl to accelerate the training process.

```@example fno_tutorial_details
xdev = reactant_device()
```

### Constructing training data

First, we construct our training data.

```@example fno_tutorial_details
rng = Random.default_rng()
```

`batch_size` is the number of observations.

```@example fno_tutorial_details
batch_size = 128
```

`m` is the length of a single observation, you can also interpret this as the size of the
grid we're evaluating our function on.

```@example fno_tutorial_details
m = 32
```

We instantiate the domain that the function operates on as a range from `0` to `2π`, whose
length is the grid size.

```@example fno_tutorial_details
xrange = range(0, 2π; length=m) .|> Float32;
nothing #hide
```

Each value in the array here, `α`, will be the multiplicative factor on the input to the
sine function.

```@example fno_tutorial_details
α = 0.5f0 .+ 0.5f0 .* rand(Float32, batch_size);
nothing #hide
```

Now, we create our data arrays. We are storing all of the training data in a single array,
in order to batch process them more efficiently.

```@example fno_tutorial_details
u_data = zeros(Float32, m, 1, batch_size);
v_data = zeros(Float32, m, 1, batch_size);
nothing #hide
```

and fill the data arrays with values. Here, `u_data` is

```@example fno_tutorial_details
for i in 1:batch_size
    u_data[:, 1, i] .= sin.(α[i] .* xrange)
    v_data[:, 1, i] .= -inv(α[i]) .* cos.(α[i] .* xrange)
end
```

### Creating the model

Finally, we get to the model itself. We instantiate a `FourierNeuralOperator` and provide
it several parameters.

The first argument is the "activation function" for each neuron.

The keyword arguments are:

- `chs` is a tuple, representing the layer sizes for each layer.
- `modes` is a 1-tuple, where the number represents the number of Fourier modes that
  are preserved, and the size of the tuple represents the number of dimensions.

```@example fno_tutorial_details
fno = FourierNeuralOperator(
    gelu;                    # activation function
    chs=(1, 64, 64, 128, 1), # channel weights
    modes=(16,),             # number of Fourier modes to retain
)
```

Now, we set up the model. This function returns two things,
a set of parameters and a set of states. Since the operator is
"stateless", the states are empty and will remain so. The parameters
are the weights of the neural network, and we will be modifying them in the training loop.

```@example fno_tutorial_details
ps, st = Lux.setup(rng, fno) |> xdev;
nothing #hide
```

We construct data as a vector of tuples (input, output). These are pre-batched,
but for example if we had a lot of training data, we could dynamically load it,
or create multiple batches.

```@example fno_tutorial_details
u_data = u_data |> xdev;
v_data = v_data |> xdev;
data = [(u_data, v_data)];
nothing #hide
```

### Training the model

Now, we create a function to train the model. An "epoch" is basically a run over all
input data, and the more epochs we have, the better the neural network gets!

```@example fno_tutorial_details
function train!(model, ps, st, data; epochs=10)
    # The `losses` array is used only for visualization,
    # you don't actually need it to train.
    losses = []
    # Initialize a training state and an optimizer (Adam, in this case).
    tstate = Training.TrainState(model, ps, st, Adam(0.003f0))
    # Loop over epochs, then loop over each batch of training data, and step into the
    # training:
    for _ in 1:epochs
        for (x, y) in data
            (_, loss, _, tstate) = Training.single_train_step!(
                AutoEnzyme(), MSELoss(), (x, y), tstate; return_gradients=Val(false)
            )
            push!(losses, Float32(loss))
        end
    end
    return losses, tstate.parameters, tstate.states
end
```

Now we train our model!

```@example fno_tutorial_details
losses, ps, st = @time train!(fno, ps, st, data; epochs=500)
```

### Applying the model

Let's try to actually apply this model using some input data.

```@example fno_tutorial_details
input_data = u_data[:, 1, 1]
```

This is our input data. It's currently one-dimensional,
but our neural network expects input in batched form, so
we simply `reshape` it (a no-cost operation) to a 3d array with singleton dimensions.

```@example fno_tutorial_details
reshaped_input = reshape(input_data, length(input_data), 1, 1)
```

Now we can pass this to `Lux.apply` (`@jit` is used to run the function with Reactant.jl):

```@example fno_tutorial_details
output_data, st = @jit Lux.apply(fno, reshaped_input, ps, st)
```

and plot it:

```@example fno_tutorial_details
using CairoMakie, AlgebraOfGraphics
const AoG = AlgebraOfGraphics
AoG.set_aog_theme!()

f, a, p = lines(dropdims(Array(reshaped_input); dims=(2, 3)); label="u")
lines!(a, dropdims(Array(output_data); dims=(2, 3)); label="Predicted")
lines!(a, Array(v_data)[:, 1, 1]; label="Expected")
axislegend(a)
# Compute the absolute error and plot that too,
# on a separate axis.
absolute_error = Array(v_data)[:, 1, 1] .- dropdims(Array(output_data); dims=(2, 3))
a2, p2 = lines(f[2, 1], absolute_error; axis=(; ylabel="Error"))
rowsize!(f.layout, 2, Aspect(1, 1 / 8))
linkxaxes!(a, a2)
f
```
