using ReTestItems, Test, Hwloc, NeuralOperators, Reactant

const BACKEND_GROUP = lowercase(get(ENV, "BACKEND_GROUP", "all"))

const RETESTITEMS_NWORKER_THREADS = parse(
    Int, get(ENV, "RETESTITEMS_NWORKER_THREADS", string(Hwloc.num_virtual_cores()))
)

@testset "NeuralOperators.jl Tests" begin
    ReTestItems.runtests(
        NeuralOperators;
        nworkers=1,
        nworker_threads=RETESTITEMS_NWORKER_THREADS,
        testitem_timeout=3600,
    )
end
