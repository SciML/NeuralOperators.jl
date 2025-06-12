"""
    OperatorConv(
        ch::Pair{<:Integer, <:Integer}, modes::Dims, tr::AbstractTransform;
        init_weight=glorot_uniform
    )

## Arguments

  - `ch`: A `Pair` of input and output channel size `ch_in => ch_out`, e.g. `64 => 64`.
  - `modes`: The modes to be preserved. A tuple of length `d`, where `d` is the dimension of
    data.
  - `tr`: The transform to operate the transformation.

## Keyword Arguments

  - `init_weight`: Initial function to initialize parameters.

## Example

```jldoctest
julia> OperatorConv(2 => 5, (16,), FourierTransform{ComplexF32}((16,)));

```
"""
@concrete struct OperatorConv <: AbstractLuxLayer
    in_chs::Int
    out_chs::Int
    prod_modes::Int
    tform <: AbstractTransform
    init_weight
end

function Base.show(io::IO, layer::OperatorConv)
    print(io, "OperatorConv(")
    print(io, layer.in_chs, " => ", layer.out_chs, ", ")
    print(io, layer.tform, ")")
    return nothing
end

function LuxCore.initialparameters(rng::AbstractRNG, layer::OperatorConv)
    in_chs, out_chs = layer.in_chs, layer.out_chs
    scale = real(one(eltype(layer.tform))) / (in_chs * out_chs)
    return (;
        weight=scale * layer.init_weight(
            rng, eltype(layer.tform), out_chs, in_chs, layer.prod_modes
        )
    )
end

function LuxCore.parameterlength(layer::OperatorConv)
    return layer.prod_modes * layer.in_chs * layer.out_chs
end

function OperatorConv(
    ch::Pair{<:Integer,<:Integer},
    modes::Dims,
    tform::AbstractTransform;
    init_weight=glorot_uniform,
)
    return OperatorConv(ch..., prod(modes), tform, init_weight)
end

function (conv::OperatorConv)(x::AbstractArray{T,N}, ps, st) where {T,N}
    x_t = transform(conv.tform, x)
    x_tr = truncate_modes(conv.tform, x_t)
    x_p = apply_pattern(x_tr, ps.weight)

    pad_dims = size(x_t)[1:(end - 2)] .- size(x_p)[1:(end - 2)]
    x_padded = pad_constant(
        x_p, expand_pad_dims(pad_dims), false; dims=ntuple(identity, ndims(x_p) - 2)
    )
    out = inverse(conv.tform, x_padded, x)

    return out, st
end

"""
    SpectralConv(args...; kwargs...)

Construct a `OperatorConv` with `FourierTransform{ComplexF32}` as the transform. See
[`OperatorConv`](@ref) for the individual arguments.

## Example

```jldoctest
julia> SpectralConv(2 => 5, (16,));

```
"""
function SpectralConv(
    ch::Pair{<:Integer,<:Integer}, modes::Dims; shift::Bool=false, kwargs...
)
    return OperatorConv(ch, modes, FourierTransform{ComplexF32}(modes, shift); kwargs...)
end

"""
    OperatorKernel(
        ch::Pair{<:Integer, <:Integer}, modes::Dims, transform::AbstractTransform,
        act=identity; kwargs...
    )

## Arguments

  - `ch`: A `Pair` of input and output channel size `ch_in => ch_out`, e.g. `64 => 64`.
  - `modes`: The modes to be preserved. A tuple of length `d`, where `d` is the dimension of
    data.
  - `transform`: The transform to operate the transformation.
  - `act`: Activation function.

All the keyword arguments are passed to the [`OperatorConv`](@ref) constructor.

## Example

```jldoctest
julia> OperatorKernel(2 => 5, (16,), FourierTransform{ComplexF64}((16,)));

```
"""
@concrete struct OperatorKernel <: AbstractLuxWrapperLayer{:layer}
    layer
end

function OperatorKernel(
    ch::Pair{<:Integer,<:Integer},
    modes::Dims{N},
    transform::AbstractTransform,
    act=identity;
    stabilizer=identity,
    complex_data::Bool=false,
    fno_skip::Symbol=:linear,
    channel_mlp_skip::Symbol=:soft_gating,
    use_channel_mlp::Bool=false,
    channel_mlp_expansion::Real=0.5,
    kwargs...,
) where {N}
    in_chs, out_chs = ch

    complex_data && (stabilizer = Base.Fix1(decomposed_activation, stabilizer))
    stabilizer = WrappedFunction(Base.BroadcastFunction(stabilizer))

    activation = complex_data ? Base.Fix1(decomposed_activation, act) : act

    conv_layer = OperatorConv(ch, modes, transform; kwargs...)

    fno_skip_layer = __fno_skip_connection(in_chs, out_chs, N, false, fno_skip)
    complex_data && (fno_skip_layer = ComplexDecomposedLayer(fno_skip_layer))

    if use_channel_mlp
        channel_mlp_hidden_channels = round(Int, out_chs * channel_mlp_expansion)
        channel_mlp = Chain(
            Conv(ntuple(Returns(1), N), out_chs => channel_mlp_hidden_channels),
            Conv(ntuple(Returns(1), N), channel_mlp_hidden_channels => out_chs),
        )
        complex_data && (channel_mlp = ComplexDecomposedLayer(channel_mlp))

        channel_mlp_skip_layer = __fno_skip_connection(
            in_chs, out_chs, N, false, channel_mlp_skip
        )
        complex_data &&
            (channel_mlp_skip_layer = ComplexDecomposedLayer(channel_mlp_skip_layer))

        return OperatorKernel(
            Parallel(
                Fix1(add_act, activation),
                Chain(
                    Parallel(
                        Fix1(add_act, act), fno_skip_layer, Chain(; stabilizer, conv_layer)
                    ),
                    channel_mlp,
                ),
                channel_mlp_skip_layer,
            ),
        )
    end

    return OperatorKernel(
        Parallel(Fix1(add_act, act), fno_skip_layer, Chain(; stabilizer, conv_layer))
    )
end

function __fno_skip_connection(in_chs, out_chs, n_dims, use_bias, skip_type)
    if skip_type == :linear
        return Conv(ntuple(Returns(1), n_dims), in_chs => out_chs; use_bias)
    elseif skip_type == :soft_gating
        @assert in_chs == out_chs "For soft gating, in_chs must equal out_chs"
        return SoftGating(out_chs, n_dims; use_bias)
    elseif skip_type == :none
        return NoOpLayer()
    else
        error("Invalid skip_type: $(skip_type)")
    end
end

"""
    SpectralKernel(args...; kwargs...)

Construct a `OperatorKernel` with `FourierTransform{ComplexF32}` as the transform. See
[`OperatorKernel`](@ref) for the individual arguments.

## Example

```jldoctest
julia> SpectralKernel(2 => 5, (16,));

```
"""
function SpectralKernel(
    ch::Pair{<:Integer,<:Integer}, modes::Dims, act=identity; shift::Bool=false, kwargs...
)
    return OperatorKernel(
        ch, modes, FourierTransform{ComplexF32}(modes, shift), act; kwargs...
    )
end

"""
    GridEmbedding(grid_boundaries::Vector{<:Tuple{<:Real,<:Real}})

Appends a uniform grid embedding to the input data along the penultimate dimension.
"""
@concrete struct GridEmbedding <: AbstractLuxLayer
    grid_boundaries <: Vector{<:Tuple{<:Real,<:Real}}
end

function Base.show(io::IO, layer::GridEmbedding)
    return print(io, "GridEmbedding(", join(layer.grid_boundaries, ", "), ")")
end

function (layer::GridEmbedding)(x::AbstractArray{T,N}, ps, st) where {T,N}
    @assert length(layer.grid_boundaries) == N - 2

    grid = meshgrid(map(enumerate(layer.grid_boundaries)) do (i, (min, max))
        return range(T(min), T(max); length=size(x, i))
    end...)

    grid = repeat(
        Lux.Utils.contiguous(reshape(grid, size(grid)..., 1)),
        ntuple(Returns(1), N - 1)...,
        size(x, N),
    )
    return cat(grid, x; dims=N - 1), st
end

"""
    ComplexDecomposedLayer(layer::AbstractLuxLayer)

Decomposes complex activations into real and imaginary parts and applies the given layer to
each component separately, and then recombines the real and imaginary parts.
"""
@concrete struct ComplexDecomposedLayer <: AbstractLuxWrapperLayer{:layer}
    layer <: AbstractLuxLayer
end

function LuxCore.initialparameters(rng::AbstractRNG, layer::ComplexDecomposedLayer)
    return (;
        real=LuxCore.initialparameters(rng, layer.layer),
        imag=LuxCore.initialparameters(rng, layer.layer),
    )
end

function LuxCore.initialstates(rng::AbstractRNG, layer::ComplexDecomposedLayer)
    return (;
        real=LuxCore.initialstates(rng, layer.layer),
        imag=LuxCore.initialstates(rng, layer.layer),
    )
end

function (layer::ComplexDecomposedLayer)(x::AbstractArray{T,N}, ps, st) where {T,N}
    rx = real.(x)
    ix = imag.(x)

    rfn_rx, st_real = layer.layer(rx, ps.real, st.real)
    rfn_ix, st_real = layer.layer(ix, ps.real, st_real)

    ifn_rx, st_imag = layer.layer(rx, ps.imag, st.imag)
    ifn_ix, st_imag = layer.layer(ix, ps.imag, st_imag)

    out = Complex.(rfn_rx .- ifn_ix, rfn_ix .+ ifn_rx)
    return out, (; real=st_real, imag=st_imag)
end

"""
    SoftGating(chs::Integer, ndims::Integer; kwargs...)

Constructs a wrapper over `Scale` with `dims = (ntuple(Returns(1), ndims)..., chs)`. All
keyword arguments are passed to the `Scale` constructor.
"""
@concrete struct SoftGating <: AbstractLuxWrapperLayer{:layer}
    layer <: Scale
end

function SoftGating(chs::Integer, ndims::Integer; kwargs...)
    return SoftGating(Scale(ntuple(Returns(1), ndims)..., chs; kwargs...))
end
