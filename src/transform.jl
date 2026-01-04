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
    FourierTransform{T}(modes, shift::Bool=false)

A concrete implementation of `AbstractTransform` for Fourier transforms.

If `shift` is `true`, we apply a `fftshift` before truncating the modes.
"""
struct FourierTransform{T, M} <: AbstractTransform{T}
    modes::M
    shift::Bool
end

function FourierTransform{T}(modes::Dims, shift::Bool = false) where {T}
    return FourierTransform{T, typeof(modes)}(modes, shift)
end

function Base.show(io::IO, ft::FourierTransform)
    print(io, "FourierTransform{", eltype(ft), "}(")
    print(io, ft.modes, ", shift=", ft.shift, ")")
    return nothing
end

Base.ndims(T::FourierTransform) = length(T.modes)

function transform(ft::FourierTransform, x::AbstractArray)
    complex_data = Lux.Utils.eltype(x) <: Complex
    res = complex_data ? fft(x, 1:ndims(ft)) : rfft(x, 1:ndims(ft))
    if ft.shift && ndims(ft) > 1
        res = fftshift(res, (1 + !complex_data):ndims(ft))
    end
    return res
end

function low_pass(ft::FourierTransform, x_fft::AbstractArray)
    return view(x_fft, (map(d -> 1:d, ft.modes)...), :, :)
end

truncate_modes(ft::FourierTransform, x_fft::AbstractArray) = low_pass(ft, x_fft)

function inverse(
        ft::FourierTransform, x_fft::AbstractArray{T, N}, x::AbstractArray{T2, N}
    ) where {T, T2, N}
    complex_data = Lux.Utils.eltype(x) <: Complex

    if ft.shift && ndims(ft) > 1
        x_fft = fftshift(x_fft, (1 + !complex_data):ndims(ft))
    end

    if complex_data
        return ifft(x_fft, 1:ndims(ft))
    else
        return real(irfft(x_fft, size(x, 1), 1:ndims(ft)))
    end
end
