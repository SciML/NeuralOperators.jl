function apply_pattern(
        x_tr::AbstractArray{T1, N}, weights::AbstractArray{T2, 3}
    ) where {T1, T2, N}
    x_size = size(x_tr)
    x_flat = reshape(x_tr, :, x_size[N - 1], x_size[N])

    x_flat_t = permutedims(x_flat, (2, 3, 1))                               # i x b x m
    x_weighted = permutedims(batched_mul(weights, x_flat_t), (3, 1, 2))     # m x o x b

    return reshape(x_weighted, x_size[1:(N - 2)]..., size(x_weighted)[2:3]...)
end

function add_act(act::F, x1, x2) where {F}
    y = x1 .+ x2
    return fast_activation!!(NNlib.fast_act(act, y), y)
end

@concrete struct Fix1 <: Function
    f
    x
end

Base.show(io::IO, f::Fix1) = print(io, "Fix1($(f.f), $(f.x))")

(f::Fix1)(args...) = f.f(f.x, args...)

function expand_pad_dims(pad_dims::Dims{N}) where {N}
    return ntuple(i -> isodd(i) ? 0 : pad_dims[i ÷ 2], 2N)
end

# Type-stable zero-padding for FFT operations.
# Pads an array with zeros in the first M spatial dimensions.
# The last two dimensions (channels, batch) are not padded.
function pad_zeros_spatial(
        x::AbstractArray{T, N}, target_sizes::Dims{M}
    ) where {T, N, M}
    @assert M == N - 2 "target_sizes must have N-2 elements (spatial dimensions only)"
    current_sizes = ntuple(i -> size(x, i), Val(M))
    # If no padding needed, return the original array
    all(current_sizes .== target_sizes) && return x
    # Create output array with target sizes + unchanged channel/batch dims
    out_size = (target_sizes..., size(x, N - 1), size(x, N))
    y = zeros(T, out_size)
    # Copy the input data to the beginning of each spatial dimension
    src_indices = (ntuple(i -> 1:size(x, i), Val(M))..., :, :)
    y[src_indices...] = x
    return y
end

function meshgrid(args::Vararg{AbstractVector, N}) where {N}
    return stack(enumerate(args)) do (i, arg)
        new_shape = ntuple(j -> j == i ? length(arg) : 1, Val(N))
        repeat_sizes = ntuple(j -> j == i ? 1 : length(args[j]), Val(N))
        return repeat(Lux.Utils.contiguous(reshape(arg, new_shape)), repeat_sizes...)
    end
end

function decomposed_activation(f::F, x::Number) where {F}
    Lux.Utils.eltype(x) <: Complex && return Complex(f(real(x)), f(imag(x)))
    return f(x)
end

apply_complex((rfn, ifn), x::Number) = apply_complex(rfn, ifn, x)
function apply_complex(rfn, ifn, x::Number)
    @assert Lux.Utils.eltype(x) <: Complex "Expected a complex number, got \
                                            $(Lux.Utils.eltype(x))"
    rl, img = real(x), imag(x)
    return Complex(rfn(rl) - ifn(img), rfn(img) + ifn(rl))
end
