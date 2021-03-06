export
       OperatorConv,
       SpectralConv,
       OperatorKernel

struct OperatorConv{P, T, S, TT}
    weight::T
    in_channel::S
    out_channel::S
    transform::TT
end

function OperatorConv{P}(weight::T,
                         in_channel::S,
                         out_channel::S,
                         transform::TT) where {P, T, S, TT <: AbstractTransform}
    return OperatorConv{P, T, S, TT}(weight, in_channel, out_channel, transform)
end

"""
    OperatorConv(ch, modes, transform;
                 init=c_glorot_uniform, permuted=false, T=ComplexF32)

## Arguments

* `ch`: A `Pair` of input and output channel size `ch_in=>ch_out`, e.g. `64=>64`.
* `modes`: The modes to be preserved. A tuple of length `d`,
    where `d` is the dimension of data.
* `Transform`: The trafo to operate the transformation.

## Keyword Arguments

* `init`: Initial function to initialize parameters.
* `permuted`: Whether the dim is permuted. If `permuted=true`, layer accepts
    data in the order of `(ch, x_1, ... , x_d , batch)`,
    otherwise the order is `(x_1, ... , x_d, ch, batch)`.
* `T`: Data type of parameters.

## Example

```jldoctest
julia> OperatorConv(2=>5, (16, ), FourierTransform)
OperatorConv(2 => 5, (16,), FourierTransform, permuted=false)

julia> OperatorConv(2=>5, (16, ), FourierTransform, permuted=true)
OperatorConv(2 => 5, (16,), FourierTransform, permuted=true)
```
"""
function OperatorConv(ch::Pair{S, S},
                      modes::NTuple{N, S},
                      Transform::Type{<:AbstractTransform};
                      init = c_glorot_uniform,
                      permuted = false,
                      T::DataType = ComplexF32) where {S <: Integer, N}
    in_chs, out_chs = ch
    scale = one(T) / (in_chs * out_chs)
    weights = scale * init(prod(modes), in_chs, out_chs)
    transform = Transform(modes)

    return OperatorConv{permuted}(weights, in_chs, out_chs, transform)
end

function SpectralConv(ch::Pair{S, S},
                      modes::NTuple{N, S};
                      init = c_glorot_uniform,
                      permuted = false,
                      T::DataType = ComplexF32) where {S <: Integer, N}
    return OperatorConv(ch, modes, FourierTransform,
                        init = init, permuted = permuted, T = T)
end

Flux.@functor OperatorConv{true}
Flux.@functor OperatorConv{false}

Base.ndims(oc::OperatorConv) = ndims(oc.transform)

ispermuted(::OperatorConv{P}) where {P} = P

function Base.show(io::IO, l::OperatorConv{P}) where {P}
    print(io,
          "OperatorConv(" *
          "$(l.in_channel) => $(l.out_channel), " *
          "$(l.transform.modes), " *
          "$(nameof(typeof(l.transform))), " *
          "permuted=$P)")
end

function operator_conv(m::OperatorConv, ????::AbstractArray)
    ????_transformed = transform(m.transform, ????) # [size(x)..., in_chs, batch]
    ????_truncated = truncate_modes(m.transform, ????_transformed) # [modes..., in_chs, batch]
    ????_applied_pattern = apply_pattern(????_truncated, m.weight) # [modes..., out_chs, batch]
    ????_padded = pad_modes(????_applied_pattern,
                         (size(????_transformed)[1:(end - 2)]...,
                          size(????_applied_pattern)[(end - 1):end]...)) # [size(x)..., out_chs, batch] <- [modes..., out_chs, batch]
    ????_inversed = inverse(m.transform, ????_padded)

    return ????_inversed
end

function (m::OperatorConv{false})(????)
    ??????? = permutedims(????, (ntuple(i -> i + 1, ndims(m))..., 1, ndims(m) + 2)) # [x, in_chs, batch] <- [in_chs, x, batch]
    ????_out = operator_conv(m, ???????) # [x, out_chs, batch]
    ????_out??? = permutedims(????_out, (ndims(m) + 1, 1:ndims(m)..., ndims(m) + 2)) # [out_chs, x, batch] <- [x, out_chs, batch]

    return ????_out???
end

function (m::OperatorConv{true})(????)
    return operator_conv(m, ????) # [x, out_chs, batch]
end

############
# operator #
############

struct OperatorKernel{L, C, F}
    linear::L
    conv::C
    ??::F
end

"""
    OperatorKernel(ch, modes, ??=identity; permuted=false)

## Arguments

* `ch`: A `Pair` of input and output channel size for spectral convolution `in_ch=>out_ch`,
    e.g. `64=>64`.
* `modes`: The modes to be preserved for spectral convolution. A tuple of length `d`,
    where `d` is the dimension of data.
* `??`: Activation function.

## Keyword Arguments

* `permuted`: Whether the dim is permuted. If `permuted=true`, layer accepts
    data in the order of `(ch, x_1, ... , x_d , batch)`,
    otherwise the order is `(x_1, ... , x_d, ch, batch)`.

## Example

```jldoctest
julia> OperatorKernel(2=>5, (16, ), FourierTransform)
OperatorKernel(2 => 5, (16,), FourierTransform, ??=identity, permuted=false)

julia> using Flux

julia> OperatorKernel(2=>5, (16, ), FourierTransform, relu)
OperatorKernel(2 => 5, (16,), FourierTransform, ??=relu, permuted=false)

julia> OperatorKernel(2=>5, (16, ), FourierTransform, relu, permuted=true)
OperatorKernel(2 => 5, (16,), FourierTransform, ??=relu, permuted=true)
```
"""
function OperatorKernel(ch::Pair{S, S},
                        modes::NTuple{N, S},
                        Transform::Type{<:AbstractTransform},
                        ?? = identity;
                        permuted = false) where {S <: Integer, N}
    linear = permuted ? Conv(Tuple(ones(Int, length(modes))), ch) :
             Dense(ch.first, ch.second)
    conv = OperatorConv(ch, modes, Transform; permuted = permuted)

    return OperatorKernel(linear, conv, ??)
end

Flux.@functor OperatorKernel

function Base.show(io::IO, l::OperatorKernel)
    print(io,
          "OperatorKernel(" *
          "$(l.conv.in_channel) => $(l.conv.out_channel), " *
          "$(l.conv.transform.modes), " *
          "$(nameof(typeof(l.conv.transform))), " *
          "??=$(string(l.??)), " *
          "permuted=$(ispermuted(l.conv))" *
          ")")
end

function (m::OperatorKernel)(????)
    return m.??.(m.linear(????) + m.conv(????))
end

#########
# utils #
#########

c_glorot_uniform(dims...) = Flux.glorot_uniform(dims...) + Flux.glorot_uniform(dims...) * im

# [prod(modes), out_chs, batch] <- [prod(modes), in_chs, batch] * [out_chs, in_chs, prod(modes)]
einsum(???????, ???????) = @tullio ????[m, o, b] := ???????[m, i, b] * ???????[m, i, o]

function apply_pattern(????_truncated, ????)
    x_size = size(????_truncated) # [m.modes..., in_chs, batch]

    ????_flattened = reshape(????_truncated, :, x_size[(end - 1):end]...) # [prod(m.modes), in_chs, batch], only 3-dims
    ????_weighted = einsum(????_flattened, ????) # [prod(m.modes), out_chs, batch], only 3-dims
    ????_shaped = reshape(????_weighted, x_size[1:(end - 2)]..., size(????_weighted)[2:3]...) # [m.modes..., out_chs, batch]

    return ????_shaped
end

pad_modes(????::AbstractArray, dims::NTuple) = pad_modes!(similar(????, dims), ????)

function pad_modes!(????_padded::AbstractArray, ????::AbstractArray)
    fill!(????_padded, eltype(????)(0)) # zeros(eltype(????), dims)
    ????_padded[map(d -> 1:d, size(????))...] .= ????

    return ????_padded
end

function ChainRulesCore.rrule(::typeof(pad_modes), ????::AbstractArray, dims::NTuple)
    function pad_modes_pullback(??????)
        return NoTangent(), view(??????, map(d -> 1:d, size(????))...), NoTangent()
    end

    return pad_modes(????, dims), pad_modes_pullback
end
