@concrete struct NOMAD <: AbstractExplicitContainerLayer{(:branch, :trunk, :additional)}
    approximator
    decoder
    concatenate <: Function
end

NOMAD(approximator, decoder) = NOMAD(approximator, decoder, __merge)

function NOMAD(; approximator=(64, 32, 32, 16), decoder=(1, 8, 8, 16),
                  approximator_activation=identity, decoder_activation=identity,
                  concatenate=__merge)
    approximator_net = Chain([Dense(approximator[i] => approximator[i + 1], approximator_activation)
                              for i in 1:(length(approximator) - 1)]...)

    decoder_net = Chain([Dense(decoder[i] => decoder[i + 1], decoder_activation)
                         for i in 1:(length(decoder) - 1)]...)

    return DeepONet(approximator_net, decoder_net, concatenate)
end

function (nomad::NOMAD)(x, ps, st::NamedTuple)
    a, st_a = nomad.approximator(x[1], ps.approximator, st.approximator)
    out, st_d = nomad.approximator(nomad.concatenate(a, x[2]), ps.decoder, st.decoder)

    return out, (approximator=st_a, decoder=st_d)
end
