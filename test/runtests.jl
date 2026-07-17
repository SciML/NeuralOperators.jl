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
    core_tests = function ()
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
    end

    run_tests(;
        core = core_tests,
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
        groups = Dict(
            "GPUCore" => (;
                env = joinpath(@__DIR__, "gpu"),
                body = core_tests,
            ),
        ),
        # Keep the historical full functional + QA coverage on the CUDA runner.
        # GPUCore needs its own environment so Reactant's CUDA artifact preference
        # is active before Reactant_jll is loaded.
        umbrellas = Dict("GPU" => ["GPUCore", "QA"]),
    )
end
