"""
    AbstractTransform

## Interface

  - `Base.ndims(<:AbstractTransform)`: N dims of modes
  - `truncate_modes(<:AbstractTransform, x_transformed::AbstractArray)`: Truncate modes
    that contribute to the noise

### Transform Interface

  - `plan_transform(<:AbstractTransform, x::AbstractArray, prev_plan)`: Construct a plan to
    apply the transform to x. Might reuse the previous plan if possible
  - `transform(<:AbstractTransform, x::AbstractArray, plan)`: Apply the transform to x using
    the plan

### Inverse Transform Interface

  - `plan_inverse(<:AbstractTransform, x_transformed::AbstractArray, prev_plan, M)`:
    Construct a plan to apply the inverse transform to `x_transformed`. Might reuse the
    previous plan if possible
  - `inverse(<:AbstractTransform, x_transformed::AbstractArray, plan, M)`: Apply the inverse
    transform to `x_transformed`
"""
abstract type AbstractTransform{T} end

Base.eltype(::Type{<:AbstractTransform{T}}) where {T} = T

printable_type(T::AbstractTransform) = "$(nameof(typeof(T))){$(eltype(T))}"

@concrete struct FourierTransform{T} <: AbstractTransform{T}
    modes
end

Base.ndims(T::FourierTransform) = length(T.modes)

function plan_transform(ft::FourierTransform, x::AbstractArray, ::Nothing)
    return plan_rfft(x, 1:ndims(ft))
end

function plan_transform(ft::FourierTransform, x::AbstractArray, prev_plan)
    size(prev_plan) == size(x) && eltype(prev_plan) == eltype(x) && return prev_plan
    return plan_transform(ft, x, nothing)
end

@non_differentiable plan_transform(::Any...)

transform(::FourierTransform, x::AbstractArray, plan) = plan * x

function low_pass(ft::FourierTransform, x_fft::AbstractArray)
    return view(x_fft, map(d -> 1:d, ft.modes)..., :, :)
end

truncate_modes(ft::FourierTransform, x_fft::AbstractArray) = low_pass(ft, x_fft)

function plan_inverse(ft::FourierTransform, x_transformed::AbstractArray{T, N},
        ::Nothing, M::NTuple{N, Int64}) where {T, N}
    return plan_irfft(x_transformed, first(M), 1:ndims(ft))
end

function plan_inverse(ft::FourierTransform, x_transformed::AbstractArray{T, N},
        prev_plan, M::NTuple{N, Int64}) where {T, N}
    size(prev_plan) == size(x_transformed) && eltype(prev_plan) == eltype(x_transformed) &&
        return prev_plan
    return plan_inverse(ft, x_transformed, nothing, M)
end

@non_differentiable plan_inverse(::Any...)

function inverse(::FourierTransform, x_transformed::AbstractArray{T, N}, plan,
        ::NTuple{N, Int64}) where {T, N}
    return real(plan * x_transformed)
end
