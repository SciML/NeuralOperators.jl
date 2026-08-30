"""
    OperatorConv(
        ch::Pair{<:Integer,<:Integer},
        modes::Dims,
        tform::AbstractTransform;
        init_weight = glorot_uniform,
    ) -> OperatorConv

Construct a Lux layer that transforms its input, applies learned weights to retained modes,
and maps the result back to physical space.

# Arguments

- `ch::Pair{<:Integer,<:Integer}`: Input and output channels as `in_chs => out_chs`.
- `modes::Dims`: Number of retained modes along each spatial dimension.
- `tform::AbstractTransform`: Transform implementation satisfying the
  [`AbstractTransform`](@ref) interface.

# Keywords

- `init_weight = glorot_uniform`: Function called as
  `init_weight(rng, coefficient_type, out_chs, in_chs, prod(modes))`.

# Fields

- `in_chs::Int`: Number of input channels.
- `out_chs::Int`: Number of output channels.
- `prod_modes::Int`: Product of the retained mode counts.
- `tform::AbstractTransform`: Transform used by the layer.
- `init_weight`: Parameter initializer for transformed-space weights.

# Returns

- An `OperatorConv` Lux layer whose input and output shapes are
  `(spatial..., channels, batches)`.

# Examples

```jldoctest
julia> layer = OperatorConv(2 => 5, (8,), FourierTransform{ComplexF32}((8,)));

julia> ps, st = Lux.setup(Xoshiro(0), layer);

julia> size(first(layer(rand(Float32, 16, 2, 1), ps, st)))
(16, 5, 1)
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
        weight = scale * layer.init_weight(
            rng, eltype(layer.tform), out_chs, in_chs, layer.prod_modes
        ),
    )
end

function LuxCore.parameterlength(layer::OperatorConv)
    return layer.prod_modes * layer.in_chs * layer.out_chs
end

function OperatorConv(
        ch::Pair{<:Integer, <:Integer},
        modes::Dims,
        tform::AbstractTransform;
        init_weight = glorot_uniform,
    )
    return OperatorConv(ch..., prod(modes), tform, init_weight)
end

function (conv::OperatorConv)(x::AbstractArray{T, N}, ps, st) where {T, N}
    x_t = transform(conv.tform, x)
    x_tr = truncate_modes(conv.tform, x_t)
    x_p = apply_pattern(x_tr, ps.weight)

    pad_dims = size(x_t)[1:(end - 2)] .- size(x_p)[1:(end - 2)]
    x_padded = pad_constant(
        x_p, expand_pad_dims(pad_dims), false; dims = ntuple(identity, ndims(x_p) - 2)
    )
    out = inverse(conv.tform, x_padded, x)

    return out, st
end

"""
    SpectralConv(
        ch::Pair{<:Integer,<:Integer}, modes::Dims; shift::Bool = false, kwargs...
    ) -> OperatorConv

Construct an [`OperatorConv`](@ref) with a [`FourierTransform`](@ref) using `ComplexF32`
coefficients.

# Arguments

- `ch::Pair{<:Integer,<:Integer}`: Input and output channels as `in_chs => out_chs`.
- `modes::Dims`: Number of retained Fourier modes along each spatial dimension.

# Keywords

- `shift::Bool = false`: Whether to shift non-redundant Fourier dimensions before
  truncation.
- `kwargs...`: Additional keywords forwarded to [`OperatorConv`](@ref).

# Returns

- An `OperatorConv` configured for Fourier transforms with `ComplexF32` weights.

# Examples

```jldoctest
julia> SpectralConv(2 => 5, (8,));

```
"""
function SpectralConv(
        ch::Pair{<:Integer, <:Integer}, modes::Dims; shift::Bool = false, kwargs...
    )
    return OperatorConv(ch, modes, FourierTransform{ComplexF32}(modes, shift); kwargs...)
end

"""
    OperatorKernel(
        ch::Pair{<:Integer, <:Integer}, modes::Dims, transform::AbstractTransform,
        act = identity;
        kwargs...,
    ) -> OperatorKernel

Construct a Lux operator block that combines a transformed convolution with a skip
connection and optional channel MLP.

# Arguments

- `ch::Pair{<:Integer,<:Integer}`: Input and output channels as `in_chs => out_chs`.
- `modes::Dims`: Number of retained modes along each spatial dimension.
- `transform::AbstractTransform`: Transform used by the convolution branch.
- `act = identity`: Activation applied after branch outputs are combined.

# Keywords

- `stabilizer = identity`: Function broadcast over inputs before the transformed
  convolution.
- `complex_data::Bool = false`: Whether the block operates on complex-valued data.
- `fno_skip::Symbol = :linear`: Spectral-branch skip connection; one of `:linear`,
  `:soft_gating`, or `:none`.
- `channel_mlp_skip::Symbol = :soft_gating`: Skip connection for the optional channel MLP;
  one of `:linear`, `:soft_gating`, or `:none`.
- `use_channel_mlp::Bool = false`: Whether to add a two-layer channel MLP.
- `channel_mlp_expansion::Real = 0.5`: Hidden width as a fraction of output channels.
- `kwargs...`: Additional keywords forwarded to [`OperatorConv`](@ref).

# Fields

- `layer::AbstractLuxLayer`: Assembled Lux layer implementing the operator block.

# Returns

- An `OperatorKernel` wrapping the assembled Lux block.

# Examples

```jldoctest
julia> OperatorKernel(2 => 5, (16,), FourierTransform{ComplexF64}((16,)));

```
"""
@concrete struct OperatorKernel <: AbstractLuxWrapperLayer{:layer}
    layer
end

function OperatorKernel(
        ch::Pair{<:Integer, <:Integer},
        modes::Dims{N},
        transform::AbstractTransform,
        act = identity;
        stabilizer = identity,
        complex_data::Bool = false,
        fno_skip::Symbol = :linear,
        channel_mlp_skip::Symbol = :soft_gating,
        use_channel_mlp::Bool = false,
        channel_mlp_expansion::Real = 0.5,
        kwargs...,
    ) where {N}
    in_chs, out_chs = ch

    complex_data && (stabilizer = Base.Fix1(decomposed_activation, stabilizer))
    stabilizer = WrappedFunction(BroadcastFunction(stabilizer))

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
    SpectralKernel(
        ch::Pair{<:Integer,<:Integer}, modes::Dims, act = identity;
        shift::Bool = false, kwargs...,
    ) -> OperatorKernel

Construct an [`OperatorKernel`](@ref) with a [`FourierTransform`](@ref) using `ComplexF32`
coefficients.

# Arguments

- `ch::Pair{<:Integer,<:Integer}`: Input and output channels as `in_chs => out_chs`.
- `modes::Dims`: Number of retained Fourier modes along each spatial dimension.
- `act = identity`: Activation applied after branch outputs are combined.

# Keywords

- `shift::Bool = false`: Whether to shift non-redundant Fourier dimensions before
  truncation.
- `kwargs...`: Additional keywords forwarded to [`OperatorKernel`](@ref).

# Returns

- An `OperatorKernel` configured for Fourier transforms with `ComplexF32` weights.

# Examples

```jldoctest
julia> SpectralKernel(2 => 5, (16,));

```
"""
function SpectralKernel(
        ch::Pair{<:Integer, <:Integer}, modes::Dims, act = identity; shift::Bool = false, kwargs...
    )
    return OperatorKernel(
        ch, modes, FourierTransform{ComplexF32}(modes, shift), act; kwargs...
    )
end

"""
    GridEmbedding(grid_boundaries::Vector{<:Tuple{<:Real,<:Real}}) -> GridEmbedding

Appends a uniform grid embedding to the input data along the penultimate dimension.

# Arguments

- `grid_boundaries`: Inclusive lower and upper coordinate bounds for each spatial
  dimension.

# Fields

- `grid_boundaries::Vector{<:Tuple{<:Real,<:Real}}`: Coordinate bounds used to construct
  the embedding.

# Returns

- A `GridEmbedding` Lux layer. Applying it to `(spatial..., channels, batches)` appends one
  coordinate channel per spatial dimension.

# Examples

```jldoctest
julia> layer = GridEmbedding([(0.0f0, 1.0f0)]);

julia> size(first(layer(zeros(Float32, 4, 2, 1), NamedTuple(), NamedTuple())))
(4, 3, 1)
```
"""
@concrete struct GridEmbedding <: AbstractLuxLayer
    grid_boundaries <: Vector{<:Tuple{<:Real, <:Real}}
end

function Base.show(io::IO, layer::GridEmbedding)
    return print(io, "GridEmbedding(", join(layer.grid_boundaries, ", "), ")")
end

function (layer::GridEmbedding)(x::AbstractArray{T, N}, ps, st) where {T, N}
    @assert length(layer.grid_boundaries) == N - 2

    grid = meshgrid(
        map(enumerate(layer.grid_boundaries)) do (i, (min, max))
            return range(T(min), T(max); length = size(x, i))
        end...,
    )

    grid = repeat(
        contiguous(reshape(grid, size(grid)..., 1)),
        ntuple(Returns(1), N - 1)...,
        size(x, N),
    )

    # Move the CPU-built grid to the same device as x (fixes CUDA scalar indexing, #125)
    grid = Lux.get_device(x)(grid)

    return cat(grid, x; dims = N - 1), st
end

"""
    ComplexDecomposedLayer(layer::AbstractLuxLayer) -> ComplexDecomposedLayer

Decomposes complex activations into real and imaginary parts and applies the given layer to
each component separately, and then recombines the real and imaginary parts.

# Arguments

- `layer::AbstractLuxLayer`: Real-valued Lux layer applied to each component.

# Fields

- `layer::AbstractLuxLayer`: Wrapped real-valued Lux layer.

# Returns

- A `ComplexDecomposedLayer` with independent real and imaginary parameter sets.

# Examples

```jldoctest
julia> layer = ComplexDecomposedLayer(Dense(2 => 2));

julia> ps, st = Lux.setup(Xoshiro(0), layer);

julia> size(first(layer(ones(ComplexF32, 2, 1), ps, st)))
(2, 1)
```
"""
@concrete struct ComplexDecomposedLayer <: AbstractLuxWrapperLayer{:layer}
    layer <: AbstractLuxLayer
end

function LuxCore.initialparameters(rng::AbstractRNG, layer::ComplexDecomposedLayer)
    return (;
        real = LuxCore.initialparameters(rng, layer.layer),
        imag = LuxCore.initialparameters(rng, layer.layer),
    )
end

function LuxCore.initialstates(rng::AbstractRNG, layer::ComplexDecomposedLayer)
    return (;
        real = LuxCore.initialstates(rng, layer.layer),
        imag = LuxCore.initialstates(rng, layer.layer),
    )
end

function (layer::ComplexDecomposedLayer)(x::AbstractArray{T, N}, ps, st) where {T, N}
    rx = real.(x)
    ix = imag.(x)

    rfn_rx, st_real = layer.layer(rx, ps.real, st.real)
    rfn_ix, st_real = layer.layer(ix, ps.real, st_real)

    ifn_rx, st_imag = layer.layer(rx, ps.imag, st.imag)
    ifn_ix, st_imag = layer.layer(ix, ps.imag, st_imag)

    out = Complex.(rfn_rx .- ifn_ix, rfn_ix .+ ifn_rx)
    return out, (; real = st_real, imag = st_imag)
end

"""
    SoftGating(chs::Integer, ndims::Integer; kwargs...) -> SoftGating

Constructs a wrapper over `Scale` with `dims = (ntuple(Returns(1), ndims)..., chs)`. All
keyword arguments are passed to the `Scale` constructor.

# Arguments

- `chs::Integer`: Number of channels in the scaled dimension.
- `ndims::Integer`: Number of spatial dimensions before the channel dimension.

# Keywords

- `kwargs...`: Additional keywords forwarded to `Lux.Scale`.

# Fields

- `layer::Scale`: Wrapped Lux scale layer.

# Returns

- A `SoftGating` layer with one trainable scale per channel.

# Examples

```jldoctest
julia> layer = SoftGating(3, 1);

julia> ps, st = Lux.setup(Xoshiro(0), layer);

julia> size(first(layer(ones(Float32, 4, 3, 2), ps, st)))
(4, 3, 2)
```
"""
@concrete struct SoftGating <: AbstractLuxWrapperLayer{:layer}
    layer <: Scale
end

function SoftGating(chs::Integer, ndims::Integer; kwargs...)
    return SoftGating(Scale(ntuple(Returns(1), ndims)..., chs; kwargs...))
end
