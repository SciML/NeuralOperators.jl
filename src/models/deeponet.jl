"""
    DeepONet(branch, trunk, additional)

Constructs a DeepONet from a `branch` and `trunk` architectures. Make sure that both the
nets output should have the same first dimension.

## Arguments

  - `branch`: `Lux` network to be used as branch net.
  - `trunk`: `Lux` network to be used as trunk net.

## References

[1] Lu Lu, Pengzhan Jin, George Em Karniadakis, "DeepONet: Learning nonlinear operators for
identifying differential equations based on the universal approximation theorem of
operators", doi: https://arxiv.org/abs/1910.03193

## Input Output Dimensions

Consider a transient 1D advection problem ∂ₜu + u ⋅ ∇u = 0, with an IC u(x,0) = g(x).
We are given several (b = 200) instances of the IC, discretized at 50 points each, and want
to query the solution for 100 different locations and times [0;1].

That makes the branch input of shape [50 x 200] and the trunk input of shape [2 x 100]. So,
the input for the branch net is 50 and 100 for the trunk net.

## Example

```jldoctest
julia> branch_net = Chain(Dense(64 => 32), Dense(32 => 32), Dense(32 => 16));

julia> trunk_net = Chain(Dense(1 => 8), Dense(8 => 8), Dense(8 => 16));

julia> deeponet = DeepONet(branch_net, trunk_net);

julia> ps, st = Lux.setup(Xoshiro(), deeponet);

julia> u = rand(Float32, 64, 5);

julia> y = rand(Float32, 1, 10);

julia> size(first(deeponet((u, y), ps, st)))
(10, 5)
```
"""
@concrete struct DeepONet <: AbstractLuxWrapperLayer{:model}
    model
end

function DeepONet(branch, trunk)
    return DeepONet(
        Chain(
            Parallel(*; branch = Chain(branch, WrappedFunction(adjoint)), trunk = trunk),
            WrappedFunction(adjoint),
        ),
    )
end

"""
    DeepONet(;
        branch = (64, 32, 32, 16), trunk = (1, 8, 8, 16),
        branch_activation = identity, trunk_activation = identity
    )

Constructs a DeepONet composed of Dense layers. Make sure the last node of `branch` and
`trunk` are same.

## Keyword arguments:

  - `branch`: Tuple of integers containing the number of nodes in each layer for branch net
  - `trunk`: Tuple of integers containing the number of nodes in each layer for trunk net
  - `branch_activation`: activation function for branch net
  - `trunk_activation`: activation function for trunk net

## References

[1] Lu Lu, Pengzhan Jin, George Em Karniadakis, "DeepONet: Learning nonlinear operators for
identifying differential equations based on the universal approximation theorem of
operators", doi: https://arxiv.org/abs/1910.03193

## Example

```jldoctest
julia> deeponet = DeepONet(; branch=(64, 32, 32, 16), trunk=(1, 8, 8, 16));

julia> ps, st = Lux.setup(Xoshiro(), deeponet);

julia> u = rand(Float32, 64, 5);

julia> y = rand(Float32, 1, 10);

julia> size(first(deeponet((u, y), ps, st)))
(10, 5)
```
"""
function DeepONet(;
        branch = (64, 32, 32, 16),
        trunk = (1, 8, 8, 16),
        branch_activation = identity,
        trunk_activation = identity,
    )

    # checks for last dimension size
    @assert branch[end] == trunk[end] "Branch and Trunk net must share the same amount \
                                       of nodes in the last layer. Otherwise Σᵢ bᵢⱼ tᵢₖ \
                                       won't work."

    branch_net = Chain(
        [
            Dense(
                    branch[i] => branch[i + 1],
                    ifelse(i == length(branch) - 1, identity, branch_activation),
                ) for i in 1:(length(branch) - 1)
        ]...,
    )

    trunk_net = Chain(
        [
            Dense(
                    trunk[i] => trunk[i + 1],
                    ifelse(i == length(trunk) - 1, identity, trunk_activation),
                ) for i in 1:(length(trunk) - 1)
        ]...,
    )

    return DeepONet(branch_net, trunk_net)
end
