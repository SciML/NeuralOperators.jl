
function IntegralKernel(W::Tuple, κ::Tuple; # bias,
        W_activation=identity, kernel_activation=identity,
        activation=identity, alg=Integrals.HCubatureJL(), kwargs...)
    W_ = Chain([Dense(W[i] => W[i + 1], W_activation) for i in 1:(length(W) - 1)]...)
    κ_ = Chain([Dense(κ[i] => κ[i + 1], kernel_activation) for i in 1:(length(κ) - 1)]...)

    IntegralKernel(W_, κ_; activation=activation, alg=alg, kwargs...)
end

"""
    IntegralKernel(W::L1, κ::L2, domain; # bias,
            activation=identity,
            alg=Integrals.HCubatureJL(), kwargs...) where {L1, L2}

returns the Integral kernel evaluated the given data point: ``σ(W_t + \\mathcal{K}_t)(v_t)(x)``
where ``W_t`` is a linear mapping and

```math
(\\mathcal{K}_t(v_t))(x) = \\int_{D_t} \\kappa^{(t)}(x,y) v_t(y) dy \\quad \\forall x \\in D_t
```

## Arguments

  - `W` : network for linear mapping
  - `κ` : network to evaluate the integral kernel
  - `domain` : domain of integration to perform integration of `κ`

## Keyword arguments

  - `activation` : activation function to be applied at the end σ # bias,
  - `alg` : `Integrals.jl` algorithm to compute the integral
  - `kwargs` : Additional arguments to be splatted into `Integrals.solve(...)`
"""
function IntegralKernel(W::L1, κ::L2; # bias,
        activation=identity,
        alg=Integrals.HCubatureJL(), kwargs...) where {L1, L2}

    # name

    return @compact(; W, κ, activation, alg, kwargs,
        dispatch=:IntegralKernel) do (x, domain)
        W_ = W(x)
        f(u, p) = κ(vcat(u, x))
        prototype = zero(x)
        prob = IntegralProblem(IntegralFunction(f, prototype), domain)
        sol = solve(prob, alg; kwargs...)
        # print("wewe \n")
        @return W_ #broadcast(activation, W_)
    end
end

"""
    function IntegralKernelOperator(
        lifting::L1, kernels::Vector{L2},
        projection::L3, domain) where {L1, L2 <: CompactLuxLayer{:IntegralKernel}, L3}

returns the continuous variant of Neural Operator

## Arguments

  - `lifting`: lifting layer
  - `kernels`: Vector of `IntegralKernel` to applied in chain after lifting
  - `projection`: projection layer
"""
function IntegralKernelOperator(lifting::L1, kernels::Vector{L2}, projection::L3,
        domain) where {L1, L2 <: CompactLuxLayer{:IntegralKernel}, L3}
    return @compact(; lifting, kernels, projection, domain,
        dispatch=:IntegralKernelOperator) do x
        v = lifting(x)
        D = sort(lifting(domain))

        for kernel in kernels
            v = kernel((v, D)) # kernel evaluation

            D = sort(kernel((D, D))) # update domain of integration for next kernel
        end

        v = projection(v)
        @return v
    end
end
