export
    OperatorConv,
    SpectralConv,
    OperatorKernel,
    SparseKernel,
    SparseKernel1D,
    SparseKernel2D,
    SparseKernel3D,
    MWT_CZ1d

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

function operator_conv(m::OperatorConv, 𝐱::AbstractArray)
    𝐱_transformed = transform(m.transform, 𝐱) # [size(x)..., in_chs, batch]
    𝐱_truncated = truncate_modes(m.transform, 𝐱_transformed) # [modes..., in_chs, batch]
    𝐱_applied_pattern = apply_pattern(𝐱_truncated, m.weight) # [modes..., out_chs, batch]
    𝐱_padded = pad_modes(𝐱_applied_pattern,
                         (size(𝐱_transformed)[1:(end - 2)]...,
                          size(𝐱_applied_pattern)[(end - 1):end]...)) # [size(x)..., out_chs, batch] <- [modes..., out_chs, batch]
    𝐱_inversed = inverse(m.transform, 𝐱_padded)

    return 𝐱_inversed
end

function (m::OperatorConv{false})(𝐱)
    𝐱ᵀ = permutedims(𝐱, (ntuple(i -> i + 1, ndims(m))..., 1, ndims(m) + 2)) # [x, in_chs, batch] <- [in_chs, x, batch]
    𝐱_out = operator_conv(m, 𝐱ᵀ) # [x, out_chs, batch]
    𝐱_outᵀ = permutedims(𝐱_out, (ndims(m) + 1, 1:ndims(m)..., ndims(m) + 2)) # [out_chs, x, batch] <- [x, out_chs, batch]

    return 𝐱_outᵀ
end

function (m::OperatorConv{true})(𝐱)
    return operator_conv(m, 𝐱) # [x, out_chs, batch]
end

############
# operator #
############

struct OperatorKernel{L, C, F}
    linear::L
    conv::C
    σ::F
end

"""
    OperatorKernel(ch, modes, σ=identity; permuted=false)

## Arguments

* `ch`: A `Pair` of input and output channel size for spectral convolution `in_ch=>out_ch`,
    e.g. `64=>64`.
* `modes`: The modes to be preserved for spectral convolution. A tuple of length `d`,
    where `d` is the dimension of data.
* `σ`: Activation function.

## Keyword Arguments

* `permuted`: Whether the dim is permuted. If `permuted=true`, layer accepts
    data in the order of `(ch, x_1, ... , x_d , batch)`,
    otherwise the order is `(x_1, ... , x_d, ch, batch)`.

## Example

```jldoctest
julia> OperatorKernel(2=>5, (16, ), FourierTransform)
OperatorKernel(2 => 5, (16,), FourierTransform, σ=identity, permuted=false)

julia> using Flux

julia> OperatorKernel(2=>5, (16, ), FourierTransform, relu)
OperatorKernel(2 => 5, (16,), FourierTransform, σ=relu, permuted=false)

julia> OperatorKernel(2=>5, (16, ), FourierTransform, relu, permuted=true)
OperatorKernel(2 => 5, (16,), FourierTransform, σ=relu, permuted=true)
```
"""
function OperatorKernel(ch::Pair{S, S},
                        modes::NTuple{N, S},
                        Transform::Type{<:AbstractTransform},
                        σ = identity;
                        permuted = false) where {S <: Integer, N}
    linear = permuted ? Conv(Tuple(ones(Int, length(modes))), ch) :
             Dense(ch.first, ch.second)
    conv = OperatorConv(ch, modes, Transform; permuted = permuted)

    return OperatorKernel(linear, conv, σ)
end

Flux.@functor OperatorKernel

function Base.show(io::IO, l::OperatorKernel)
    print(io,
          "OperatorKernel(" *
          "$(l.conv.in_channel) => $(l.conv.out_channel), " *
          "$(l.conv.transform.modes), " *
          "$(nameof(typeof(l.conv.transform))), " *
          "σ=$(string(l.σ)), " *
          "permuted=$(ispermuted(l.conv))" *
          ")")
end

function (m::OperatorKernel)(𝐱)
    return m.σ.(m.linear(𝐱) + m.conv(𝐱))
end

"""
    SparseKernel(κ, ch, σ=identity)

Sparse kernel layer.

## Arguments

* `κ`: A neural network layer for approximation, e.g. a `Dense` layer or a MLP.
* `ch`: Channel size for linear transform, e.g. `32`.
* `σ`: Activation function.
"""
struct SparseKernel{N,T,S}
    conv_blk::T
    out_weight::S
end

function SparseKernel(filter::NTuple{N,T}, ch::Pair{S, S}; init=Flux.glorot_uniform) where {N,T,S}
    input_dim, emb_dim = ch
    conv = Conv(filter, input_dim=>emb_dim, relu; stride=1, pad=1, init=init)
    W_out = Dense(emb_dim, input_dim; init=init)
    return SparseKernel{N,typeof(conv),typeof(W_out)}(conv, W_out)
end

function SparseKernel1D(k::Int, α, c::Int=1; init=Flux.glorot_uniform)
    input_dim = c*k
    emb_dim = 128
    return SparseKernel((3, ), input_dim=>emb_dim; init=init)
end

function SparseKernel2D(k::Int, α, c::Int=1; init=Flux.glorot_uniform)
    input_dim = c*k^2
    emb_dim = α*k^2
    return SparseKernel((3, 3), input_dim=>emb_dim; init=init)
end

function SparseKernel3D(k::Int, α, c::Int=1; init=Flux.glorot_uniform)
    input_dim = c*k^2
    emb_dim = α*k^2
    conv = Conv((3, 3, 3), emb_dim=>emb_dim, relu; stride=1, pad=1, init=init)
    W_out = Dense(emb_dim, input_dim; init=init)
    return SparseKernel{3,typeof(conv),typeof(W_out)}(conv, W_out)
end

Flux.@functor SparseKernel

function (l::SparseKernel)(X::AbstractArray)
    bch_sz, _, dims_r... = reverse(size(X))
    dims = reverse(dims_r)

    X_ = l.conv_blk(X)  # (dims..., emb_dims, B)
    X_ = reshape(X_, prod(dims), :, bch_sz)  # (prod(dims), emb_dims, B)
    Y = l.out_weight(batched_transpose(X_))  # (in_dims, prod(dims), B)
    Y = reshape(batched_transpose(Y), dims..., :, bch_sz)  # (dims..., in_dims, B)
    return collect(Y)
end


struct MWT_CZ1d{T,S,R,Q,P}
    k::Int
    L::Int
    A::T
    B::S
    C::R
    T0::Q
    ec_s::P
    ec_d::P
    rc_e::P
    rc_o::P
end

function MWT_CZ1d(k::Int=3, α::Int=5, L::Int=0, c::Int=1; base::Symbol=:legendre, init=Flux.glorot_uniform)
    H0, H1, G0, G1, Φ0, Φ1 = get_filter(base, k)
    H0r = zero_out!(H0 * Φ0)
    G0r = zero_out!(G0 * Φ0)
    H1r = zero_out!(H1 * Φ1)
    G1r = zero_out!(G1 * Φ1)

    dim = c*k
    A = SpectralConv(dim=>dim, (α,); init=init)
    B = SpectralConv(dim=>dim, (α,); init=init)
    C = SpectralConv(dim=>dim, (α,); init=init)
    T0 = Dense(k, k)

    ec_s = vcat(H0', H1')
    ec_d = vcat(G0', G1')
    rc_e = vcat(H0r, G0r)
    rc_o = vcat(H1r, G1r)
    return MWT_CZ1d(k, L, A, B, C, T0, ec_s, ec_d, rc_e, rc_o)
end

function wavelet_transform(l::MWT_CZ1d, X::AbstractArray{T,4}) where {T}
    N = size(X, 3)
    Xa = vcat(view(X, :, :, 1:2:N, :), view(X, :, :, 2:2:N, :))
    d = NNlib.batched_mul(Xa, l.ec_d)
    s = NNlib.batched_mul(Xa, l.ec_s)
    return d, s
end

function even_odd(l::MWT_CZ1d, X::AbstractArray{T,4}) where {T}
    bch_sz, N, dims_r... = reverse(size(X))
    dims = reverse(dims_r)
    @assert dims[1] == 2*l.k
    Y = similar(X, bch_sz, 2N, l.c, l.k)
    view(Y, :, :, 1:2:N, :) .= NNlib.batched_mul(X, l.rc_e)
    view(Y, :, :, 2:2:N, :) .= NNlib.batched_mul(X, l.rc_o)
    return Y
end

function (l::MWT_CZ1d)(X::T) where {T<:AbstractArray}
    bch_sz, N, dims_r... = reverse(size(X))
    ns = floor(log2(N))
    stop = ns - l.L

    # decompose
    Ud = T[]
    Us = T[]
    for i in 1:stop
        d, X = wavelet_transform(l, X)
        push!(Ud, l.A(d)+l.B(d))
        push!(Us, l.C(d))
    end
    X = l.T0(X)

    # reconstruct
    for i in stop:-1:1
        X += Us[i]
        X = vcat(X, Ud[i])  # x = torch.cat((x, Ud[i]), -1)
        X = even_odd(l, X)
    end
    return X
end

# function Base.show(io::IO, l::MWT_CZ1d)
#     print(io, "MWT_CZ($(l.in_channel) => $(l.out_channel), $(l.transform.modes), $(nameof(typeof(l.transform))), permuted=$P)")
# end


#########
# utils #
#########

c_glorot_uniform(dims...) = Flux.glorot_uniform(dims...) + Flux.glorot_uniform(dims...) * im

# [prod(modes), out_chs, batch] <- [prod(modes), in_chs, batch] * [out_chs, in_chs, prod(modes)]
einsum(𝐱₁, 𝐱₂) = @tullio 𝐲[m, o, b] := 𝐱₁[m, i, b] * 𝐱₂[m, i, o]

function apply_pattern(𝐱_truncated, 𝐰)
    x_size = size(𝐱_truncated) # [m.modes..., in_chs, batch]

    𝐱_flattened = reshape(𝐱_truncated, :, x_size[(end - 1):end]...) # [prod(m.modes), in_chs, batch], only 3-dims
    𝐱_weighted = einsum(𝐱_flattened, 𝐰) # [prod(m.modes), out_chs, batch], only 3-dims
    𝐱_shaped = reshape(𝐱_weighted, x_size[1:(end - 2)]..., size(𝐱_weighted)[2:3]...) # [m.modes..., out_chs, batch]

    return 𝐱_shaped
end

pad_modes(𝐱::AbstractArray, dims::NTuple) = pad_modes!(similar(𝐱, dims), 𝐱)

function pad_modes!(𝐱_padded::AbstractArray, 𝐱::AbstractArray)
    fill!(𝐱_padded, eltype(𝐱)(0)) # zeros(eltype(𝐱), dims)
    𝐱_padded[map(d -> 1:d, size(𝐱))...] .= 𝐱

    return 𝐱_padded
end

function ChainRulesCore.rrule(::typeof(pad_modes), 𝐱::AbstractArray, dims::NTuple)
    function pad_modes_pullback(𝐲̄)
        return NoTangent(), view(𝐲̄, map(d -> 1:d, size(𝐱))...), NoTangent()
    end

    return pad_modes(𝐱, dims), pad_modes_pullback
end
