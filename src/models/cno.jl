"""
    CNOBlock(
        in_channels::Integer,
        out_channels::Integer,
        modes::Dims{N},
        activation = gelu;
        upsample_factor::Integer = 2,
    ) where {N}

A single Convolutional Neural Operator (CNO) block.

Each block:
1. **Upsamples** the input by `upsample_factor` using bilinear/trilinear interpolation.
2. Applies a **3×(…×3) convolution** in the higher-resolution space with `SamePad`.
3. Applies the **activation function** pointwise.
4. **Downsamples** back to the original resolution via average pooling.

This design ensures the discrete operator converges to a continuous limit as resolution
increases, unlike standard CNNs which only approximate finite-dimensional maps.

## Arguments

  - `in_channels`: Number of input channels.
  - `out_channels`: Number of output channels.
  - `modes`: Spatial dimensions tuple (length = data dimensionality). Only the length is
    used to set the kernel dimensionality.
  - `activation`: Pointwise activation applied after convolution.

## Keyword Arguments

  - `upsample_factor`: Integer upsampling factor. Default is `2`.

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
    spatial_dims = ntuple(identity, N)
    kernel = ntuple(Returns(3), N)
    pool_window = ntuple(Returns(upsample_factor), N)

    return CNOBlock(
        Chain(
            # 1. Upsample to higher resolution
            Upsample(:bilinear; scale = upsample_factor),
            # 2. Convolution at high resolution (preserves spatial size via SamePad)
            Conv(kernel, in_channels => out_channels, activation; pad = SamePad()),
            # 3. Downsample back via average pooling (paper Section 3.1)
            MeanPool(pool_window),
        ),
    )
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

CNO applies a sequence of resolution-preserving continuous convolutional blocks. Each
block upsamples the input, applies a convolution in the higher-resolution space, and
downsamples back. This design is proven to converge to a well-defined continuous operator
as resolution increases, making CNO resolution-invariant by construction.

**Architecture**:
1. **Lifting** `Conv(1×…×1)`: maps `in_channels → hidden_channels`
2. **CNO blocks** × `num_layers`: each is upsample → conv → activation → avgpool
3. **Projection**: `Conv(1×…×1, act)` → `Conv(1×…×1)` maps to `out_channels`

## Arguments

  - `modes`: Spatial size tuple (length `d` for d-dimensional data). Only its length
    matters — kept consistent with the FNO API.
  - `in_channels`: Number of input channels.
  - `out_channels`: Number of output channels.
  - `hidden_channels`: Number of channels inside the CNO blocks.

## Keyword Arguments

  - `num_layers`: Number of `CNOBlock` layers. Default is `4`.
  - `activation`: Activation function used inside each block. Default is `gelu`.
  - `upsample_factor`: Spatial upsampling factor inside each block. Default is `2`.

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
        Conv(ones_kernel, hidden_channels => hidden_channels, activation),
        Conv(ones_kernel, hidden_channels => out_channels),
    )

    return ConvolutionalNeuralOperator(Chain(; lifting, cno_blocks, projection))
end
