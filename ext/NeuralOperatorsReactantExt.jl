module NeuralOperatorsReactantExt

using FFTW: FFTW
using NeuralOperators: NeuralOperators, FourierTransform
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

end
