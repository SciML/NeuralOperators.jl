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
    return operator_conv(x, conv.tform, ps.weight), st
end

function operator_conv(x, tform::AbstractTransform, weights)
    x_t = transform(tform, x)
    x_tr = truncate_modes(tform, x_t)
    x_p = apply_pattern(x_tr, weights)

    pad_dims = size(x_t)[1:(end - 2)] .- size(x_p)[1:(end - 2)]
    x_padded = pad_constant(
        x_p, expand_pad_dims(pad_dims), false; dims=ntuple(identity, ndims(x_p) - 2)
    )

    return inverse(tform, x_padded, size(x))
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
function SpectralConv(ch::Pair{<:Integer,<:Integer}, modes::Dims; kwargs...)
    return OperatorConv(ch, modes, FourierTransform{ComplexF32}(modes); kwargs...)
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
    kwargs...,
) where {N}
    return OperatorKernel(
        Parallel(
            Fix1(add_act, act),
            Conv(ntuple(one, N), ch),
            OperatorConv(ch, modes, transform; kwargs...),
        ),
    )
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
function SpectralKernel(ch::Pair{<:Integer,<:Integer}, modes::Dims, act=identity; kwargs...)
    return OperatorKernel(ch, modes, FourierTransform{ComplexF32}(modes), act; kwargs...)
end
