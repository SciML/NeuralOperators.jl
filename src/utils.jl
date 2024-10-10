function apply_pattern(
        x_tr::AbstractArray{T1, N}, weights::AbstractArray{T2, 3}) where {T1, T2, N}
    x_size = size(x_tr)
    x_flat = reshape(x_tr, :, x_size[N - 1], x_size[N])

    x_flat_t = permutedims(x_flat, (2, 3, 1))                               # i x b x m
    x_weighted = permutedims(batched_matmul(weights, x_flat_t), (3, 1, 2))  # m x o x b

    return reshape(x_weighted, x_size[1:(N - 2)]..., size(x_weighted)[2:3]...)
end

function add_act(act::F, x1, x2) where {F}
    y = x1 .+ x2
    act = NNlib.fast_act(act, y)
    return fast_activation!!(act, y)
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

@non_differentiable expand_pad_dims(::Any)

# Handling Wrapper Types are hard. Make sure to not construct a ReshapedArray of
# BatchedAdjoint
safe_batched_adjoint(x::AbstractArray) = NNlib.batched_adjoint(x)

function CRC.rrule(::typeof(safe_batched_adjoint), x::AbstractArray)
    return safe_batched_adjoint(x), ∇safe_batched_adjoint
end

∇safe_batched_adjoint(Δ) = NoTangent(), safe_batched_adjoint(Δ)
function ∇safe_batched_adjoint(Δ::AbstractArray{T, 3}) where {T}
    return ∇safe_batched_adjoint(get_device_type(Δ), Δ)
end

function ∇safe_batched_adjoint(::Type{<:AbstractDevice}, Δ::AbstractArray{T, 3}) where {T}
    return NoTangent(), safe_batched_adjoint(Δ)
end

function ∇safe_batched_adjoint(
        ::Type{<:AbstractGPUDevice}, Δ::AbstractArray{T, 3}) where {T}
    return NoTangent(), stack(adjoint, eachslice(Δ; dims=3))
end
