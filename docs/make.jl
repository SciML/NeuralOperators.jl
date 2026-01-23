using Documenter, NeuralOperators

cp("./Manifest.toml", "./docs/src/assets/Manifest.toml"; force = true)
cp("./docs/Project.toml", "./docs/src/assets/Project.toml"; force = true)

ENV["GKSwstype"] = "100"
ENV["DATADEPS_ALWAYS_ACCEPT"] = true

include("pages.jl")

makedocs(;
    sitename = "NeuralOperators.jl",
    clean = true,
    doctest = false,
    linkcheck = true,
    modules = [NeuralOperators],
    format = Documenter.HTML(;
        prettyurls = get(ENV, "CI", "false") == "true",
        canonical = "https://docs.sciml.ai/NeuralOperators/stable/",
        assets = ["assets/favicon.ico"],
    ),
    pages,
)

deploydocs(; repo = "github.com/SciML/NeuralOperators.jl.git", push_preview = true)
