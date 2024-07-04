# Temporarily capture certain calls like AMDGPU for ComplexFloats
@inline __batched_mul(x, y) = __batched_mul(x, y, get_device((x, y)))
@inline function __batched_mul(
        x::AbstractArray{<:Number, 3}, y::AbstractArray{<:Number, 3}, _)
    return x ⊠ y
end
@inline function __batched_mul(
        x::AbstractArray{<:Complex, 3}, y::AbstractArray{<:Complex, 3}, ::LuxAMDGPUDevice)
    # FIXME: This is not good for performance but that is okay for now
    return stack(*, eachslice(x; dims=3), eachslice(y; dims=3))
end

@inline function __project(b::AbstractArray{T1, 2}, t::AbstractArray{T2, 3}) where {T1, T2}
    # b : p x nb
    # t : p x N x nb
    @show size.([b, t])
    b_ = reshape(b, size(b, 1), 1, size(b, 2)) # p x 1 x nb
    return dropdims(sum(b_ .*t, dims = 1), dims = 1) # N x nb
end

@inline function __project(b::AbstractArray{T1, 3}, t::AbstractArray{T2, 3}) where {T1, T2}
    # b : p x u x nb
    # t : p x N x nb
    @show size.([b, t])
    if size(b, 2) == 1 || size(t, 2) == 1
        return sum(b .* t, dims = 1) # 1 x N x nb
    else
        return LuxNeuralOperators.__batched_mul(batched_adjoint(t), b) # N x p x nb
    end
end

@inline function __project(b::AbstractArray{T1, N}, t::AbstractArray{T2, 3}) where {T1, T2, N}
    # b : p x u_size x nb
    # t : p x N x nb
    @show size.([b, t])

    u_size = size(b)[2:end-1]

    b_ = reshape(b, size(b,1), 1, u_size..., size(b)[end])
    # p x u_size x 1 x nb

    t_ = reshape(t, size(t)[1:2]..., ones(eltype(u_size), length(u_size))..., size(t)[end])
    # p x (1,1,1...) X N x nb

    return dropdims(sum(b_ .* t_; dims = 1), dims = 1) # u_size x N x nb
end