using SciMLTesting, NeuralOperators, Test

run_qa(
    NeuralOperators;
    explicit_imports = true,
    # Mirror the previous `Aqua.test_all(...; ambiguities = false)` +
    # `Aqua.test_ambiguities(...; recursive = false)` pair: run the ambiguities
    # sub-check non-recursively (recursing into deps surfaces unrelated upstream
    # ambiguities).
    aqua_kwargs = (; ambiguities = (; recursive = false)),
    ei_kwargs = (;
        all_qualified_accesses_are_public = (;
            # NNlib.fast_act picks the fast activation variant (tanh_fast/sigmoid_fast)
            # for the LuxLib fast_activation!! call in `add_act`. LuxLib's
            # fast_activation!! docstring states it does NOT apply this replacement
            # itself ("that needs to be done by the user if needed"), so we must call
            # NNlib.fast_act directly. It is an NNlib internal with no public equivalent.
            ignore = (:fast_act,),
        ),
    ),
)
