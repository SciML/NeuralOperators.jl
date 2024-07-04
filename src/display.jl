function Base.show(io::IO, model::conv) where {conv <: OperatorConv}
    # print(io, model.name*"()  # "*string(Lux.parameterlength(model))*" parameters")
    print(io, model.name)
end

function Base.show(io::IO, ::MIME"text/plain", model::conv) where {conv <: OperatorConv}
    show(io, model.name)
end

function Base.show(io::IO, model::Lux.CompactLuxLayer{:DeepONet})
    _print_wrapper_model(io, "Branch net :\n", model.layers.branch)
    print(io, "\n \n")
    _print_wrapper_model(io, "Trunk net :\n", model.layers.trunk)
    if :additional in keys(model.layers)
        print(io, "\n \n")
        _print_wrapper_model(io, "Additional net :\n", model.layers.additional)
    end
end

function Base.show(io::IO, ::MIME"text/plain", x::CompactLuxLayer{:DeepONet})
    show(io, x)
end
