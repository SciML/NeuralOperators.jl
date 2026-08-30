"""
    FourierNeuralOperator(
        σ = gelu;
        chs::Dims = (2, 64, 64, 64, 64, 64, 128, 1),
        modes::Dims = (16,),
        kwargs...,
    ) -> FourierNeuralOperator

Construct a Fourier neural operator from an explicit sequence of channel widths. The model
uses Fourier kernels to learn mappings between discretized functions.

The model lifts inputs with a pointwise convolution, applies Fourier operator kernels, and
projects their output with two pointwise convolutions.

# Arguments

- `σ = gelu`: Activation function used by the spectral kernels and penultimate
  projection layer.

# Keywords

- `chs::Dims`: Channel widths. The first and last entries are input and output widths; the
  intermediate entries configure lifting, spectral, and projection layers. At least five
  entries are required.
- `modes::Dims`: Number of retained Fourier modes along each spatial dimension.
- `kwargs...`: Additional keywords forwarded to every [`SpectralKernel`](@ref).

# Fields

- `model::AbstractLuxLayer`: Assembled Lux model.

# Returns

- A `FourierNeuralOperator` Lux layer.

# Examples

```jldoctest
julia> fno = FourierNeuralOperator(gelu; chs=(2, 64, 64, 128, 1), modes=(16,));

julia> ps, st = Lux.setup(Xoshiro(), fno);

julia> u = rand(Float32, 1024, 2, 5);

julia> size(first(fno(u, ps, st)))
(1024, 1, 5)
```
"""
@concrete struct FourierNeuralOperator <: AbstractLuxWrapperLayer{:model}
    model <: AbstractLuxLayer
end

function FourierNeuralOperator(
        σ = gelu; chs::Dims{C} = (2, 64, 64, 64, 64, 64, 128, 1), modes::Dims{M} = (16,), kwargs...
    ) where {C, M}
    @assert length(chs) ≥ 5

    return FourierNeuralOperator(
        Chain(
            Conv(map(Returns(1), modes), chs[1] => chs[2]),
            Chain(
                [
                    SpectralKernel(chs[i] => chs[i + 1], modes, σ; kwargs...) for
                        i in 2:(C - 3)
                ]...,
            ),
            Chain(
                Conv(map(Returns(1), modes), chs[C - 2] => chs[C - 1], σ),
                Conv(map(Returns(1), modes), chs[C - 1] => chs[C]),
            ),
        ),
    )
end

"""
    FourierNeuralOperator(
        modes::Dims{N},
        in_channels::Integer,
        out_channels::Integer,
        hidden_channels::Integer;
        num_layers::Integer=4,
        lifting_channel_ratio::Integer=2,
        projection_channel_ratio::Integer=2,
        positional_embedding::Union{Symbol,AbstractLuxLayer}=:grid, # :grid | :none
        activation=gelu,
        use_channel_mlp::Bool=true,
        channel_mlp_expansion::Real=0.5,
        channel_mlp_skip::Symbol=:soft_gating,
        fno_skip::Symbol=:linear,
        complex_data::Bool=false,
        stabilizer=tanh,
        shift::Bool=false,
    ) -> FourierNeuralOperator

Construct a configurable Fourier neural operator with lifting and projection networks.

# Arguments

- `modes::Dims{N}`: Number of retained Fourier modes along each of the `N` spatial
  dimensions.
- `in_channels::Integer`: Channel count presented to the lifting network. With
  `positional_embedding = :grid`, this is the input data's channels plus `N` coordinate
  channels.
- `out_channels::Integer`: Number of output channels.
- `hidden_channels::Integer`: Width of each Fourier operator block.

# Keywords

- `num_layers::Integer = 4`: Number of Fourier operator blocks.
- `lifting_channel_ratio::Integer = 2`: Lifting hidden width divided by
  `hidden_channels`.
- `projection_channel_ratio::Integer = 2`: Projection hidden width divided by
  `out_channels`.
- `positional_embedding::Union{Symbol,AbstractLuxLayer} = :grid`: Positional layer, or
  `:grid` to append uniform coordinates, or `:none` to append nothing.
- `activation = gelu`: Activation used by lifting, projection, and operator blocks.
- `use_channel_mlp::Bool = true`: Whether each operator block includes a channel MLP.
- `channel_mlp_expansion::Real = 0.5`: Channel-MLP hidden width as a fraction of
  `hidden_channels`.
- `channel_mlp_skip::Symbol = :soft_gating`: Channel-MLP skip connection; one of
  `:linear`, `:soft_gating`, or `:none`.
- `fno_skip::Symbol = :linear`: Fourier-block skip connection; one of `:linear`,
  `:soft_gating`, or `:none`.
- `complex_data::Bool = false`: Whether lifting, operator, and projection layers process
  complex-valued data.
- `stabilizer = tanh`: Function applied before each transformed convolution.
- `shift::Bool = false`: Whether Fourier coefficients are shifted before truncation.

# Returns

- A `FourierNeuralOperator` Lux layer.

# Examples

```jldoctest
julia> fno = FourierNeuralOperator((4,), 3, 1, 8; num_layers=2);

julia> ps, st = Lux.setup(Xoshiro(0), fno);

julia> size(first(fno(rand(Float32, 8, 2, 3), ps, st)))
(8, 1, 3)
```
"""
function FourierNeuralOperator(
        modes::Dims{N},
        in_channels::Integer,
        out_channels::Integer,
        hidden_channels::Integer;
        num_layers::Integer = 4,
        lifting_channel_ratio::Integer = 2,
        projection_channel_ratio::Integer = 2,
        positional_embedding::Union{Symbol, AbstractLuxLayer} = :grid, # :grid | :none
        activation = gelu,
        use_channel_mlp::Bool = true,
        channel_mlp_expansion::Real = 0.5,
        channel_mlp_skip::Symbol = :soft_gating,
        fno_skip::Symbol = :linear,
        complex_data::Bool = false,
        stabilizer = tanh,
        shift::Bool = false,
    ) where {N}
    lifting_channels = hidden_channels * lifting_channel_ratio
    projection_channels = out_channels * projection_channel_ratio

    if positional_embedding isa Symbol
        @assert positional_embedding in (:grid, :none)
        if positional_embedding == :grid
            positional_embedding = GridEmbedding([(0.0f0, 1.0f0) for _ in 1:N])
        else
            positional_embedding = NoOpLayer()
        end
    end

    lifting = Chain(
        Conv(map(Returns(1), modes), in_channels => lifting_channels, activation),
        Conv(map(Returns(1), modes), lifting_channels => hidden_channels),
    )
    complex_data && (lifting = ComplexDecomposedLayer(lifting))

    projection = Chain(
        Conv(map(Returns(1), modes), hidden_channels => projection_channels, activation),
        Conv(map(Returns(1), modes), projection_channels => out_channels),
    )
    complex_data && (projection = ComplexDecomposedLayer(projection))

    fno_blocks = Chain(
        [
            SpectralKernel(
                hidden_channels => hidden_channels,
                modes,
                activation;
                stabilizer,
                shift,
                use_channel_mlp,
                channel_mlp_expansion,
                channel_mlp_skip,
                fno_skip,
                complex_data,
            ) for _ in 1:num_layers
        ]...,
    )

    return FourierNeuralOperator(
        Chain(; positional_embedding, lifting, fno_blocks, projection)
    )
end
