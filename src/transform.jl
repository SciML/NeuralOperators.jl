"""
    AbstractTransform

## Interface

  - `Base.ndims(<:AbstractTransform)`: N dims of modes
  - `transform(<:AbstractTransform, x::AbstractArray)`: Apply the transform to x
  - `truncate_modes(<:AbstractTransform, x_transformed::AbstractArray)`: Truncate modes
    that contribute to the noise
  - `inverse(<:AbstractTransform, x_transformed::AbstractArray)`: Apply the inverse
    transform to `x_transformed`
"""
abstract type AbstractTransform{T} end

Base.eltype(::Type{<:AbstractTransform{T}}) where {T} = T

function transform end
function truncate_modes end
function inverse end

"""
    FourierTransform{T}(modes)

A concrete implementation of `AbstractTransform` for Fourier transforms.
"""
@concrete struct FourierTransform{T} <: AbstractTransform{T}
    modes
end

function Base.show(io::IO, ft::FourierTransform)
    print(io, "FourierTransform{", eltype(ft), "}(")
    print(io, ft.modes, ")")
    return nothing
end

Base.ndims(T::FourierTransform) = length(T.modes)

function transform(ft::FourierTransform, x::AbstractArray)
    res = unwrapped_eltype(x) <: Complex ? fft(x, 1:ndims(ft)) : rfft(x, 1:ndims(ft))
    ndims(ft) > 1 && (res = fftshift(res, 1:ndims(ft)))
    return res
end

function low_pass(ft::FourierTransform, x_fft::AbstractArray)
    return view(x_fft, (map(d -> 1:d, ft.modes)...), :, :)
end

truncate_modes(ft::FourierTransform, x_fft::AbstractArray) = low_pass(ft, x_fft)

function inverse(
    ft::FourierTransform, x_fft::AbstractArray{T,N}, x::AbstractArray{T2,N}
) where {T,T2,N}
    ndims(ft) > 1 && (x_fft = fftshift(x_fft, 1:ndims(ft)))

    if unwrapped_eltype(x) <: Complex
        return ifft(x_fft, 1:ndims(ft))
    else
        return real(irfft(x_fft, size(x, 1), 1:ndims(ft)))
    end
end
