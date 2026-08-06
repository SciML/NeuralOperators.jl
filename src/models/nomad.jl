"""
    NOMAD(approximator, decoder) -> NOMAD

Construct a nonlinear manifold decoder model from approximator and decoder Lux
architectures. The approximator output is concatenated with each coordinate input before
being passed to the decoder.

# Arguments

- `approximator`: Lux layer mapping the sampled input function to a latent representation.
- `decoder`: Lux layer mapping the concatenated latent representation and coordinates to
  the model output.

# Fields

- `model::AbstractLuxLayer`: Assembled Lux model containing approximator and decoder.

# Returns

- A `NOMAD` Lux layer accepting `(approximator_input, coordinates)`.

# References

[1] Jacob H. Seidman and Georgios Kissas and Paris Perdikaris and George J. Pappas, "NOMAD:
Nonlinear Manifold Decoders for Operator Learning", doi: https://arxiv.org/abs/2206.03551

# Examples

```jldoctest
julia> approximator_net = Chain(Dense(8 => 32), Dense(32 => 32), Dense(32 => 16));

julia> decoder_net = Chain(Dense(18 => 16), Dense(16 => 16), Dense(16 => 8));

julia> nomad = NOMAD(approximator_net, decoder_net);

julia> ps, st = Lux.setup(Xoshiro(), nomad);

julia> u = rand(Float32, 8, 5);

julia> y = rand(Float32, 2, 5);

julia> size(first(nomad((u, y), ps, st)))
(8, 5)
```
"""
@concrete struct NOMAD <: AbstractLuxWrapperLayer{:model}
    model
end

function NOMAD(approximator, decoder)
    return NOMAD(Chain(; approximator = Parallel(vcat, approximator, NoOpLayer()), decoder))
end

"""
    NOMAD(;
        approximator = (8, 32, 32, 16), decoder = (18, 16, 8, 8),
        approximator_activation = identity, decoder_activation = identity
    ) -> NOMAD

Construct a NOMAD whose approximator and decoder are chains of dense layers. The first
decoder width must equal the final approximator width plus the coordinate width supplied at
evaluation time.

# Keywords

- `approximator = (8, 32, 32, 16)`: Widths of the approximator network, including input and
  latent output widths.
- `decoder = (18, 16, 8, 8)`: Widths of the decoder network. Its input width must include
  both latent and coordinate features.
- `approximator_activation = identity`: Activation applied to every approximator layer.
- `decoder_activation = identity`: Activation applied to every decoder layer.

# Returns

- A `NOMAD` Lux layer accepting `(approximator_input, coordinates)`.

# References

[1] Jacob H. Seidman and Georgios Kissas and Paris Perdikaris and George J. Pappas, "NOMAD:
Nonlinear Manifold Decoders for Operator Learning", doi: https://arxiv.org/abs/2206.03551

# Examples

```jldoctest
julia> nomad = NOMAD(; approximator=(8, 32, 32, 16), decoder=(18, 16, 8, 8));

julia> ps, st = Lux.setup(Xoshiro(), nomad);

julia> u = rand(Float32, 8, 5);

julia> y = rand(Float32, 2, 5);

julia> size(first(nomad((u, y), ps, st)))
(8, 5)
```
"""
function NOMAD(;
        approximator = (8, 32, 32, 16),
        decoder = (18, 16, 8, 8),
        approximator_activation = identity,
        decoder_activation = identity,
    )
    approximator_net = Chain(
        [
            Dense(approximator[i] => approximator[i + 1], approximator_activation) for
                i in 1:(length(approximator) - 1)
        ]...,
    )

    decoder_net = Chain(
        [
            Dense(decoder[i] => decoder[i + 1], decoder_activation) for
                i in 1:(length(decoder) - 1)
        ]...,
    )

    return NOMAD(approximator_net, decoder_net)
end
