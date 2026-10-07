"""
    CNOBlock(
        in_channels::Integer,
        out_channels::Integer,
        modes::Dims{N},
        activation = gelu;
        upsample_factor::Integer = 2,
    ) where {N}

A single Convolutional Neural Operator (CNO) block.

Each block applies a `3×(…×3)` convolution followed by the CNO activation operator: the
signal is upsampled by `upsample_factor` with band-limited (sinc) interpolation, the
activation is applied pointwise at the higher resolution, and the result is low-pass
filtered and downsampled back to the input resolution. Filtering before downsampling keeps
the harmonics created by the activation from aliasing into lower frequencies.

Resampling uses the FFT, so the spatial dimensions are treated as periodic.

## Arguments

  - `in_channels`: Number of input channels.
  - `out_channels`: Number of output channels.
  - `modes`: Spatial dimensions tuple (length = data dimensionality). Only the length is
    used to set the kernel dimensionality.
  - `activation`: Pointwise activation applied at the upsampled resolution.

## Keyword Arguments

  - `upsample_factor`: Integer upsampling factor of the activation operator. Default is `2`.

## References

[1] Raonic et al., "Convolutional Neural Operators for robust and accurate learning of
PDEs," NeurIPS 2023. https://arxiv.org/abs/2302.01178
"""
@concrete struct CNOBlock <: AbstractLuxWrapperLayer{:model}
    model
end

function CNOBlock(
        in_channels::Integer,
        out_channels::Integer,
        modes::Dims{N},
        activation = gelu;
        upsample_factor::Integer = 2,
    ) where {N}
    kernel = ntuple(Returns(3), N)
    return CNOBlock(
        Chain(
            Conv(kernel, in_channels => out_channels; pad = SamePad()),
            CNOActivation(activation, upsample_factor),
        ),
    )
end

@concrete struct CNOActivation <: AbstractLuxLayer
    activation
    upsample_factor::Int
end

function (act::CNOActivation)(x::AbstractArray{T, M}, _, st::NamedTuple) where {T, M}
    sz = size(x)[1:(M - 2)]
    y = act.activation.(spectral_resample(x, sz .* act.upsample_factor))
    return spectral_resample(y, sz), st
end

# Band-limited resampling of the spatial dimensions of `x` to size `sz`. Frequencies at or
# above the lower of the two Nyquist limits are dropped.
function spectral_resample(x::AbstractArray{T}, sz::Dims{N}) where {T, N}
    in_sz = size(x)[1:N]
    in_sz == sz && return x
    x_fft = resize_half_spectrum(rfft(x, 1:N), first(in_sz), first(sz))
    for d in 2:N
        x_fft = resize_full_spectrum(x_fft, d, sz[d])
    end
    y = irfft(x_fft, first(sz), 1:N)
    return y .* T(length(y) / length(x))
end

function select_range(x::AbstractArray, d::Int, r::AbstractUnitRange)
    return x[ntuple(i -> i == d ? r : Colon(), ndims(x))...]
end

function resize_half_spectrum(x_fft::AbstractArray, n::Int, m::Int)
    k = (min(n, m) - 1) ÷ 2
    return pad_constant(select_range(x_fft, 1, 1:(k + 1)), (0, m ÷ 2 - k), false; dims = 1)
end

function resize_full_spectrum(x_fft::AbstractArray, d::Int, m::Int)
    n = size(x_fft, d)
    k = (min(n, m) - 1) ÷ 2
    pos = pad_constant(select_range(x_fft, d, 1:(k + 1)), (0, m - 2k - 1), false; dims = d)
    return cat(pos, select_range(x_fft, d, (n - k + 1):n); dims = d)
end

"""
    ConvolutionalNeuralOperator(
        modes::Dims{N},
        in_channels::Integer,
        out_channels::Integer,
        hidden_channels::Integer;
        num_layers::Integer = 4,
        activation = gelu,
        upsample_factor::Integer = 2,
    ) where {N}

Convolutional Neural Operator (CNO) for learning PDE solution operators.

CNO applies a sequence of resolution-preserving blocks. Each block is a convolution
followed by an anti-aliased activation: the signal is upsampled with band-limited (sinc)
interpolation, the activation is applied at the higher resolution, and the result is
low-pass filtered and downsampled back. Resampling uses the FFT, so the spatial dimensions
are treated as periodic.

**Architecture**:
1. **Lifting** `Conv(1×…×1)`: maps `in_channels → hidden_channels`
2. **CNO blocks** × `num_layers`: each is conv → anti-aliased activation
3. **Projection**: `Conv(1×…×1)` → anti-aliased activation → `Conv(1×…×1)` maps to
   `out_channels`

## Arguments

  - `modes`: Spatial size tuple (length `d` for d-dimensional data). Only its length
    matters — kept consistent with the FNO API.
  - `in_channels`: Number of input channels.
  - `out_channels`: Number of output channels.
  - `hidden_channels`: Number of channels inside the CNO blocks.

## Keyword Arguments

  - `num_layers`: Number of `CNOBlock` layers. Default is `4`.
  - `activation`: Activation function used by the anti-aliased activations. Default is
    `gelu`.
  - `upsample_factor`: Upsampling factor of the anti-aliased activations. Default is `2`.

## References

[1] Raonic et al., "Convolutional Neural Operators for robust and accurate learning of
PDEs," NeurIPS 2023. https://arxiv.org/abs/2302.01178

## Example

```jldoctest
julia> cno = ConvolutionalNeuralOperator((16,), 1, 1, 32; num_layers=3);

julia> ps, st = Lux.setup(Xoshiro(), cno);

julia> u = rand(Float32, 64, 1, 5);

julia> size(first(cno(u, ps, st)))
(64, 1, 5)
```
"""
@concrete struct ConvolutionalNeuralOperator <: AbstractLuxWrapperLayer{:model}
    model <: AbstractLuxLayer
end

function ConvolutionalNeuralOperator(
        modes::Dims{N},
        in_channels::Integer,
        out_channels::Integer,
        hidden_channels::Integer;
        num_layers::Integer = 4,
        activation = gelu,
        upsample_factor::Integer = 2,
    ) where {N}
    ones_kernel = ntuple(Returns(1), N)

    lifting = Conv(ones_kernel, in_channels => hidden_channels)

    cno_blocks = Chain(
        [
            CNOBlock(
                hidden_channels, hidden_channels, modes, activation; upsample_factor,
            ) for _ in 1:num_layers
        ]...,
    )

    projection = Chain(
        Conv(ones_kernel, hidden_channels => hidden_channels),
        CNOActivation(activation, upsample_factor),
        Conv(ones_kernel, hidden_channels => out_channels),
    )

    return ConvolutionalNeuralOperator(Chain(; lifting, cno_blocks, projection))
end
