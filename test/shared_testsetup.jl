@testsetup module SharedTestSetup
import Reexport: @reexport

@reexport using Lux, Optimisers, Random, StableRNGs, Reactant, Enzyme
using LuxTestUtils: check_approx
using FFTW

const BACKEND_GROUP = lowercase(get(ENV, "BACKEND_GROUP", "All"))

train!(args...; kwargs...) = train!(MSELoss(), AutoZygote(), args...; kwargs...)

function train!(loss, backend, model, ps, st, data; epochs = 10)
    l1 = @jit loss(model, ps, st, first(data))

    tstate = Training.TrainState(model, ps, st, Adam(0.01f0))
    for _ in 1:epochs, (x, y) in data
        _, _, _, tstate = Training.single_train_step!(backend, loss, (x, y), tstate)
    end

    l2 = @jit loss(model, tstate.parameters, tstate.states, first(data))

    return l2, l1
end

sumabs2first(model, x, ps, st) = sum(abs2, first(model(x, ps, st)))

function ∇sumabs2_reactant_fd(model, x, ps, st)
    _, ∂x_fd, ∂ps_fd, _ = @jit Reactant.TestUtils.finite_difference_gradient(
        sumabs2first, Const(model), f64(x), f64(ps), Const(f64(st))
    )
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

export check_approx, BACKEND_GROUP, train!, ∇sumabs2_reactant_fd, ∇sumabs2_reactant

end
