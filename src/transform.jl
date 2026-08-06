"""
    AbstractTransform{T}

Developer interface for transforms used by [`OperatorConv`](@ref). `T` is the coefficient
element type used to initialize the operator's transformed-space weights.

# Required Methods

An implementation must define:

- `Base.ndims(tform::MyTransform)`: Return the number of transformed spatial dimensions.
- [`transform(tform::MyTransform, x)`](@ref transform): Transform the spatial dimensions.
- [`truncate_modes(tform::MyTransform, transformed)`](@ref truncate_modes): Retain the
  modes represented by the `modes` passed to [`OperatorConv`](@ref).
- [`inverse(tform::MyTransform, transformed, x)`](@ref inverse): Return to physical space.

The final two dimensions of every array are channels and batches. `transform` and
`truncate_modes` must preserve those dimensions, and `inverse` must return an array with
the same shape as its reference input `x`. The product of the spatial dimensions returned
by `truncate_modes` must equal `prod(modes)` for the associated operator.

# Examples

```julia
module IdentityTransforms
import NeuralOperators: AbstractTransform, inverse, transform, truncate_modes

struct IdentityTransform <: AbstractTransform{Float32}
    dimensions::Int
end

Base.ndims(tform::IdentityTransform) = tform.dimensions
transform(::IdentityTransform, x::AbstractArray) = x
truncate_modes(::IdentityTransform, transformed::AbstractArray) = transformed
inverse(::IdentityTransform, transformed::AbstractArray, ::AbstractArray) = transformed
end
```
"""
abstract type AbstractTransform{T} end

Base.eltype(::Type{<:AbstractTransform{T}}) where {T} = T

"""
    transform(tform::AbstractTransform, x::AbstractArray) -> AbstractArray

Transform the spatial dimensions of `x` while preserving its final channel and batch
dimensions.

# Arguments

- `tform::AbstractTransform`: Transform implementation and its configuration.
- `x::AbstractArray`: Physical-space data shaped as `(spatial..., channels, batches)`.

# Returns

- An array whose final two dimensions equal the channel and batch dimensions of `x`.

# Extension Rules

Define `NeuralOperators.transform(::MyTransform, x::AbstractArray)` for each
[`AbstractTransform`](@ref) subtype. The returned coefficient representation must be
accepted by [`truncate_modes`](@ref) and [`inverse`](@ref).

# Examples

```julia
NeuralOperators.transform(::IdentityTransform, x::AbstractArray) = x
```
"""
function transform end

"""
    truncate_modes(tform::AbstractTransform, transformed::AbstractArray) -> AbstractArray

Select the transformed modes used by an operator layer.

# Arguments

- `tform::AbstractTransform`: Transform implementation and its mode configuration.
- `transformed::AbstractArray`: Coefficients returned by [`transform`](@ref).

# Returns

- Retained coefficients with the original channel and batch dimensions.

# Extension Rules

Define `NeuralOperators.truncate_modes(::MyTransform, transformed::AbstractArray)` for each
[`AbstractTransform`](@ref) subtype. For an [`OperatorConv`](@ref) constructed with
`modes`, the product of the retained spatial dimensions must equal `prod(modes)`.

# Examples

```julia
NeuralOperators.truncate_modes(
    ::IdentityTransform, transformed::AbstractArray
) = transformed
```
"""
function truncate_modes end

"""
    inverse(
        tform::AbstractTransform,
        transformed::AbstractArray,
        x::AbstractArray,
    ) -> AbstractArray

Map transformed data back to the physical-space shape of the reference input `x`.

# Arguments

- `tform::AbstractTransform`: Transform implementation and its configuration.
- `transformed::AbstractArray`: Full transformed array after retained modes are padded.
- `x::AbstractArray`: Reference input that defines the required output shape and value kind.

# Returns

- Physical-space data with `size(result) == size(x)`.

# Extension Rules

Define `NeuralOperators.inverse(::MyTransform, transformed::AbstractArray,
x::AbstractArray)` for each [`AbstractTransform`](@ref) subtype. The implementation must
handle the padded representation produced by [`OperatorConv`](@ref) and preserve whether
the reference input is real or complex.

# Examples

```julia
NeuralOperators.inverse(
    ::IdentityTransform, transformed::AbstractArray, ::AbstractArray
) = transformed
```
"""
function inverse end

"""
    FourierTransform{T}(modes::Dims, shift::Bool = false) -> FourierTransform

A Fourier implementation of [`AbstractTransform`](@ref).

# Type Parameters

- `T`: Complex coefficient type used for transformed-space weights, such as `ComplexF32`.

# Arguments

- `modes::Dims`: Number of retained modes along each spatial dimension.
- `shift::Bool = false`: Whether to shift non-redundant Fourier dimensions before mode
  truncation.

# Fields

- `modes::M`: Retained modes along each transformed spatial dimension.
- `shift::Bool`: Whether transformed coefficients are shifted before truncation.

# Returns

- A configured `FourierTransform{T}`.

# Examples

```jldoctest
julia> FourierTransform{ComplexF32}((8,))
FourierTransform{ComplexF32}((8,), shift=false)
```
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
    complex_data = recursive_eltype(x) <: Complex
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
    complex_data = recursive_eltype(x) <: Complex

    if ft.shift && ndims(ft) > 1
        x_fft = fftshift(x_fft, (1 + !complex_data):ndims(ft))
    end

    if complex_data
        return ifft(x_fft, 1:ndims(ft))
    else
        return real(irfft(x_fft, size(x, 1), 1:ndims(ft)))
    end
end
