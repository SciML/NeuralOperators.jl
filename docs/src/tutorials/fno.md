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
using NeuralOperators, Lux, Random, Optimisers, Zygote, CairoMakie

rng = Random.default_rng()

data_size = 128
m = 32

xrange = range(0, 2π; length=m) .|> Float32;
u_data = zeros(Float32, m, 1, data_size);
α = 0.5f0 .+ 0.5f0 .* rand(Float32, data_size);
v_data = zeros(Float32, m, 1, data_size);

for i in 1:data_size
    u_data[:, 1, i] .= sin.(α[i] .* xrange)
    v_data[:, 1, i] .= -inv(α[i]) .* cos.(α[i] .* xrange)
end

fno = FourierNeuralOperator(gelu; chs=(1, 64, 64, 128, 1), modes=(16,), permuted=Val(true))

ps, st = Lux.setup(rng, fno);
data = [(u_data, v_data)];

function train!(model, ps, st, data; epochs=10)
    losses = []
    tstate = Training.TrainState(model, ps, st, Adam(0.01f0))
    for _ in 1:epochs, (x, y) in data
        _, loss, _, tstate = Training.single_train_step!(AutoZygote(), MSELoss(), (x, y),
            tstate)
        push!(losses, loss)
    end
    return losses
end

losses = train!(fno, ps, st, data; epochs=100)

lines(losses)
```

```@raw html
</details>
```

````@example minimal_lux
using NeuralOperators, Lux, Random, Optimisers, Zygote, CairoMakie
````

### Constructing training data

First, we construct our training data.

````@example minimal_lux
rng = Random.default_rng()
````

`data_size` is the number of observations.

````@example minimal_lux
data_size = 128
````

`m` is the length of a single observation, you can also interpret this as the size of the grid we're evaluating our function on.

````@example minimal_lux
m = 32
````

We instantiate the domain that the function operates on
as a range from `0` to `2π`, whose length is the grid size.

````@example minimal_lux
xrange = range(0, 2π; length=m) .|> Float32;
nothing #hide
````

Each value in the array here, `α`, will be the multiplicative
factor on the input to the sine function.

````@example minimal_lux
α = 0.5f0 .+ 0.5f0 .* rand(Float32, data_size);
nothing #hide
````

Now, we create our data arrays.  We are storing all
of the training data in a single array, in order to
batch process them more efficiently.

````@example minimal_lux
u_data = zeros(Float32, m, 1, data_size);
v_data = zeros(Float32, m, 1, data_size);
nothing #hide
````

and fill the data arrays with values.
Here, `u_data` is

````@example minimal_lux
for i in 1:data_size
    u_data[:, 1, i] .= sin.(α[i] .* xrange)
    v_data[:, 1, i] .= -inv(α[i]) .* cos.(α[i] .* xrange)
end
````

### Creating the model

Finally, we get to the model itself.  We instantiate a `FourierNeuralOperator` and provide it several parameters.

The first argument is the "activation function" for each neuron.

The keyword arguments are:

  - `chs` is a tuple, representing the layer sizes for each layer.
  - `modes` is a 1-tuple, where the number represents the number of Fourier modes that
    are preserved, and the size of the tuple represents the number of dimensions.
  - `permuted` indicates that the order of the arguments is permuted such that each column
    of the array represents a single observation.  This is substantially faster than the usual
    row access pattern, since Julia stores arrays by concatenating columns.
    `Val(true)` is another way of expressing `true`, but in the type domain, so that
    the compiler can see the value and use the appropriate optimizations.

````@example minimal_lux
fno = FourierNeuralOperator(
    gelu;                    # activation function
    chs=(1, 64, 64, 128, 1), # channel weights
    modes=(16,),             # number of Fourier modes to retain
    permuted=Val(true)       # structure of the data means that columns are observations
)
````

Now, we set up the model.  This function returns two things,
a set of parameters and a set of states.  Since the operator is
"stateless", the states are empty and will remain so.  The parameters
are the weights of the neural network, and we will be modifying them in the training loop.

````@example minimal_lux
ps, st = Lux.setup(rng, fno);
nothing #hide
````

We construct data as a vector of tuples (input, output).  These are pre-batched,
but for example if we had a lot of training data, we could dynamically load it,
or create multiple batches.

````@example minimal_lux
data = [(u_data, v_data)];
nothing #hide
````

### Training the model

Now, we create a function to train the model.
An "epoch" is basically a run over all input data,
and the more epochs we have, the better the neural network gets!

````@example minimal_lux
function train!(model, ps, st, data; epochs=10)
    # The `losses` array is used only for visualization,
    # you don't actually need it to train.
    losses = []
    # Initialize a training state and an optimizer (Adam, in this case).
    tstate = Training.TrainState(model, ps, st, Adam(0.01f0))
    # Loop over epochs, then loop over each batch of training data, and step into the training:
    for _ in 1:epochs
        for (x, y) in data
            _, loss, _, tstate = Training.single_train_step!(
                AutoZygote(), MSELoss(), (x, y),
                tstate)
            push!(losses, loss)
        end
    end
    return losses
end
````

Now we train our model!

````@example minimal_lux
losses = @time train!(fno, ps, st, data; epochs=500)
````

We can plot the losses - you can see that at some point, we hit diminishing returns.

````@example minimal_lux
lines(losses; axis=(; yscale=log10, ylabel="Loss", xlabel="Epoch"))
````

### Applying the model

Let's try to actually apply this model using some input data.

````@example minimal_lux
input_data = u_data[:, 1, 1]
````

This is our input data.  It's currently one-dimensional,
but our neural network expects input in batched form, so
we simply `reshape` it (a no-cost operation) to a 3d array with singleton dimensions.

````@example minimal_lux
reshaped_input = reshape(input_data, length(input_data), 1, 1)
````

Now we can pass this to `Lux.apply`:

````@example minimal_lux
output_data, st = Lux.apply(fno, reshaped_input, ps, st)
````

and plot it:

````@example minimal_lux
f, a, p = lines(dropdims(reshaped_input; dims=(2, 3)); label="u")
lines!(a, dropdims(output_data; dims=(2, 3)); label="Predicted")
lines!(a, v_data[:, 1, 1]; label="Expected")
axislegend(a)
# Compute the absolute error and plot that too,
# on a separate axis.
absolute_error = v_data[:, 1, 1] .- dropdims(output_data; dims=(2, 3))
a2, p2 = lines(f[2, 1], absolute_error; axis=(; ylabel="Error"))
rowsize!(f.layout, 2, Aspect(1, 1 / 8))
linkxaxes!(a, a2)
f
````
