@testsetup module SharedTestSetup
import Reexport: @reexport

@reexport using Lux, Zygote, Optimisers, Random, StableRNGs, Reactant
using LuxTestUtils: check_approx
using FFTW

const BACKEND_GROUP = lowercase(get(ENV, "BACKEND_GROUP", "All"))

train!(args...; kwargs...) = train!(MSELoss(), AutoZygote(), args...; kwargs...)

function train!(loss, backend, model, ps, st, data; epochs=10)
    l1 = @jit loss(model, ps, st, first(data))

    tstate = Training.TrainState(model, ps, st, Adam(0.01f0))
    for _ in 1:epochs, (x, y) in data
        _, _, _, tstate = Training.single_train_step!(backend, loss, (x, y), tstate)
    end

    l2 = @jit loss(model, ps, st, first(data))

    return l2, l1
end

sumabs2first(model, x, ps, st) = sum(abs2, first(model(x, ps, st)))

function zygote_gradient(model, x, ps, st)
    return Zygote.gradient(sumabs2first, model, x, ps, st)[2:3]
end

function enzyme_gradient(model, x, ps, st)
    return Enzyme.gradient(Reverse, sumabs2first, Const(model), x, ps, Const(st))[2:3]
end

export check_approx
export BACKEND_GROUP, train!
export zygote_gradient, enzyme_gradient

end
