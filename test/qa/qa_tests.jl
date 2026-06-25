using SciMLTesting, NeuralOperators, Test

run_qa(
    NeuralOperators;
    explicit_imports = true,
    # Mirror the previous `Aqua.test_all(...; ambiguities = false)` +
    # `Aqua.test_ambiguities(...; recursive = false)` pair: run the ambiguities
    # sub-check non-recursively (recursing into deps surfaces unrelated upstream
    # ambiguities).
    aqua_kwargs = (; ambiguities = (; recursive = false)),
    # Non-public names of *other* packages that NeuralOperators qualifies. These go
    # public as the base libraries declare them; until then, ignore by name.
    ei_kwargs = (;
        all_qualified_accesses_are_public = (;
            ignore = (
                :BroadcastFunction,   # Base
                :Fix1,                # Base
                :Utils,               # Lux
                :contiguous,          # Lux.Utils
                :eltype,              # Lux.Utils
                :fast_act,            # NNlib
                :initialparameters,   # LuxCore (interface method NeuralOperators extends)
                :initialstates,       # LuxCore (interface method NeuralOperators extends)
                :parameterlength,     # LuxCore (interface method NeuralOperators extends)
            ),
        ),
    ),
)
