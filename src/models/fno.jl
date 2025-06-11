"""
    FourierNeuralOperator(
        σ=gelu;
        chs::Dims{C}=(2, 64, 64, 64, 64, 64, 128, 1),
        modes::Dims{M}=(16,),
        kwargs...
    ) where {C, M}

The Fourier neural operator is a operator learning model that uses a Fourier kernel to
perform spectral convolutions. It is a promising operator for surrogate methods, and can be
regarded as a physics operator.

The model is composed of a `Dense` layer to lift a `(d + 1)`-dimensional vector field to an
`n`-dimensional vector field, an integral kernel operator which consists of four Fourier
kernels, and two `Dense` layers to project data back to the scalar field of the space of
interest.

## Arguments

  - `σ`: Activation function for all layers in the model.

## Keyword Arguments

  - `chs`: A `Tuple` or `Vector` of the size of each of the 8 channels.
  - `modes`: The modes to be preserved. A tuple of length `d`, where `d` is the dimension
    of data.  For example, one-dimensional data would have a 1-element tuple, and
    two-dimensional data would have a 2-element tuple.

## Example

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
    σ=gelu; chs::Dims{C}=(2, 64, 64, 64, 64, 64, 128, 1), modes::Dims{M}=(16,), kwargs...
) where {C,M}
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

function FourierNeuralOperator(
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
    channel_mlp_dropout_rate::Real=0.0,
    channel_mlp_expansion::Real=0.5,
    channel_mlp_skip::Symbol=:soft_gating,
    fno_skip::Symbol=:linear,
    complex_input::Bool=false,
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
end
