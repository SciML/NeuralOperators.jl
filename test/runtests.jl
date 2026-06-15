using SafeTestsets: @safetestset
using SciMLTesting

# Each group cell is its own CI process (grouped-tests.yml@v1 launches one job per
# GROUP), so Reactant/CUDA gets the whole device: pin the memory knobs to a single
# job rather than dividing across the in-process workers the old ParallelTestRunner
# layout used.
withenv(
    "XLA_REACTANT_GPU_MEM_FRACTION" => 0.9,
    "XLA_REACTANT_GPU_PREALLOCATE" => false,
    "JULIA_CUDA_HARD_MEMORY_LIMIT" => "90%",
) do
    run_tests(;
        core = function ()
            @time @safetestset "Fourier Neural Operator" begin
                include(joinpath(@__DIR__, "models", "fno_tests.jl"))
            end
            @time @safetestset "DeepONet" begin
                include(joinpath(@__DIR__, "models", "deeponet_tests.jl"))
            end
            @time @safetestset "NOMAD" begin
                include(joinpath(@__DIR__, "models", "nomad_tests.jl"))
            end
            @time @safetestset "SpectralConv" begin
                include(joinpath(@__DIR__, "layers", "spectral_conv_tests.jl"))
            end
            return @time @safetestset "SpectralKernel" begin
                include(joinpath(@__DIR__, "layers", "spectral_kernel_tests.jl"))
            end
        end,
        qa = (;
            env = joinpath(@__DIR__, "qa"),
            body = function ()
                @time @safetestset "Aqua / Explicit Imports" begin
                    include(joinpath(@__DIR__, "qa", "qa_tests.jl"))
                end
                return @time @safetestset "Doctests" begin
                    include(joinpath(@__DIR__, "qa", "doctests.jl"))
                end
            end,
        ),
        # The "GPU" cell historically ran the full functional + QA suite on the
        # self-hosted CUDA runner, where the Reactant tests pick up CUDA through
        # `reactant_device()` (the old GPU.yml workflow, folded into this group).
        # Expand it to Core then QA to preserve that behavior.
        umbrellas = Dict("GPU" => ["Core", "QA"]),
    )
end
