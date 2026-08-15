using Lux, Optimisers, Random, StableRNGs, Reactant, Enzyme, FastTransforms
using LuxTestUtils: check_approx
using MLDataDevices: cpu_device, reactant_device

sumabs2first(model, x, ps, st) = sum(abs2, first(model(x, ps, st)))

function central_finite_difference(f, x::AbstractArray{T}) where {T <: Real}
    epsilon = cbrt(eps(T))
    gradient = similar(x)
    for i in eachindex(x)
        x_plus, x_minus = copy(x), copy(x)
        x_plus[i] += epsilon
        x_minus[i] -= epsilon
        gradient[i] = (f(x_plus) - f(x_minus)) / (2 * epsilon)
    end
    return gradient
end

function central_finite_difference(
        f, x::AbstractArray{Complex{T}}, complex_mask::AbstractArray{Bool}
    ) where {T <: Real}
    epsilon = cbrt(eps(T))
    gradient = similar(x)
    for i in eachindex(x)
        x_plus, x_minus = copy(x), copy(x)
        x_plus[i] += epsilon
        x_minus[i] -= epsilon
        real_gradient = (f(x_plus) - f(x_minus)) / (2 * epsilon)

        imaginary_gradient = zero(T)
        if complex_mask[i]
            x_plus[i] = x[i] + im * epsilon
            x_minus[i] = x[i] - im * epsilon
            imaginary_gradient = (f(x_plus) - f(x_minus)) / (2 * epsilon)
        end
        gradient[i] = real_gradient + im * imaginary_gradient
    end
    return gradient
end

function complex_parameter_mask(flat_parameters, restructure)
    shifted_parameters, _ = Optimisers.destructure(restructure(flat_parameters .+ im))
    return .!iszero.(imag.(shifted_parameters .- flat_parameters))
end

function finite_difference_structure(f, structure)
    flat_structure, restructure = Optimisers.destructure(structure)
    if eltype(flat_structure) <: Complex
        mask = complex_parameter_mask(flat_structure, restructure)
        flat_gradient = central_finite_difference(f ∘ restructure, flat_structure, mask)
    else
        flat_gradient = central_finite_difference(f ∘ restructure, flat_structure)
    end
    return restructure(flat_gradient)
end

function ∇sumabs2_finite_difference(model, x, ps, st)
    x, ps, st = f64((x, ps, st))
    ∂x_fd = finite_difference_structure(x) do x_perturbed
        return sumabs2first(model, x_perturbed, ps, st)
    end

    ∂ps_fd = finite_difference_structure(ps) do ps_perturbed
        return sumabs2first(model, x, ps_perturbed, st)
    end
    return ∂x_fd, ∂ps_fd
end

function ∇sumabs2_enzyme(model, x, ps, st)
    dx = Enzyme.make_zero(x)
    dps = Enzyme.make_zero(ps)
    Enzyme.autodiff(
        Enzyme.Reverse,
        sumabs2first,
        Active,
        Const(model),
        Duplicated(x, dx),
        Duplicated(ps, dps),
        Const(st),
    )
    return dx, dps
end

function ∇sumabs2_reactant(model, x, ps, st)
    return @jit ∇sumabs2_enzyme(model, x, ps, st)
end
