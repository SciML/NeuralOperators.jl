# Temporarily capture certain calls like AMDGPU for ComplexFloats
@inline __batched_mul(x, y) = x ⊠ y

@inline function __project(b::AbstractArray{T1, 2}, t::AbstractArray{T2, 3}) where {T1, T2}
    # b : p x nb
    # t : p x N x nb
    b_ = reshape(b, size(b, 1), 1, size(b, 2)) # p x 1 x nb
    return dropdims(sum(b_ .* t; dims=1); dims=1) # N x nb
end

@inline function __project(b::AbstractArray{T1, 3}, t::AbstractArray{T2, 3}) where {T1, T2}
    # b : p x u x nb
    # t : p x N x nb
    if size(b, 2) == 1 || size(t, 2) == 1
        return sum(b .* t; dims=1) # 1 x N x nb
    else
        return __batched_mul(batched_adjoint(t), b) # N x p x nb
    end
end

@inline function __project(
        b::AbstractArray{T1, N}, t::AbstractArray{T2, 3}) where {T1, T2, N}
    # b : p x u_size x nb
    # t : p x N x nb
    u_size = size(b)[2:(end - 1)]

    b_ = reshape(b, size(b, 1), 1, u_size..., size(b)[end])
    # p x 1 x u_size x nb

    t_ = reshape(t, size(t)[1:2]..., ones(eltype(u_size), length(u_size))..., size(t)[end])
    # p x N x (1,1,1...) x nb

    return dropdims(sum(b_ .* t_; dims=1); dims=1) # N x u_size x nb
end