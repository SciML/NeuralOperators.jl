using NeuralOperators, Test, ParallelTestRunner

parsed_args = parse_args(@isdefined(TEST_ARGS) ? TEST_ARGS : ARGS)

# Find all tests
testsuite = find_tests(@__DIR__)

filter_tests!(testsuite, parsed_args)

# Remove shared setup files that shouldn't be run directly
delete!(testsuite, "shared_testsetup")
delete!(testsuite, "layers/layers_testsetup")

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
