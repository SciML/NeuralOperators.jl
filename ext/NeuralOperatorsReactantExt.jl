module NeuralOperatorsReactantExt

using FFTW: FFTW
using NeuralOperators: NeuralOperators, FourierTransform
using NNlib: NNlib
using Reactant: Reactant, TracedRArray

# XXX: Reevaluate after https://github.com/EnzymeAD/Reactant.jl/issues/246 is fixed
function NeuralOperators.transform(ft::FourierTransform, x::TracedRArray{T, N}) where {T, N}
    x_c = Reactant.promote_to(TracedRArray{Complex{T}, N}, x)
    return FFTW.fft(x_c, 1:ndims(ft))
end

function NeuralOperators.inverse(
        ft::FourierTransform, x::TracedRArray{T, N}, ::NTuple{N, Int64}) where {T, N}
    return real(FFTW.ifft(x, 1:ndims(ft)))
end

function NeuralOperators.fast_pad_zeros(x::TracedRArray, pad_dims)
    return NNlib.pad_zeros(
        x, NeuralOperators.expand_pad_dims(pad_dims); dims=ntuple(identity, ndims(x) - 2))
end

end
