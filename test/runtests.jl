using NeuralOperators, Test, ParallelTestRunner

const GROUP = get(ENV, "GROUP", "All")

parsed_args = parse_args(@isdefined(TEST_ARGS) ? TEST_ARGS : ARGS)

# Find all tests
testsuite = find_tests(@__DIR__)

filter_tests!(testsuite, parsed_args)

# Remove shared setup files that shouldn't be run directly
delete!(testsuite, "shared_testsetup")
delete!(testsuite, "layers/layers_testsetup")

# Dispatch on the CI test GROUP. "QA" runs only the quality-assurance suite
# (Aqua / ExplicitImports / doctests); "Core" runs the functional suite; "All"
# (the default for a bare local `Pkg.test()`) runs everything.
const QA_TESTS = ("qa_tests", "doctests")
if GROUP == "QA"
    for name in collect(keys(testsuite))
        name in QA_TESTS || delete!(testsuite, name)
    end
elseif GROUP == "Core"
    for name in QA_TESTS
        delete!(testsuite, name)
    end
end

total_jobs = min(
    something(parsed_args.jobs, ParallelTestRunner.default_njobs()), length(keys(testsuite))
)

withenv(
    "XLA_REACTANT_GPU_MEM_FRACTION" => 1 / (total_jobs + 0.1),
    "XLA_REACTANT_GPU_PREALLOCATE" => false,
    "JULIA_CUDA_HARD_MEMORY_LIMIT" => "$(100 / (total_jobs + 0.1))%",
) do
    runtests(NeuralOperators, parsed_args; testsuite)
end
