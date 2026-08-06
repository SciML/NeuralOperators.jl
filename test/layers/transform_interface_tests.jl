module ExternalTransform

    import NeuralOperators: AbstractTransform, inverse, transform, truncate_modes

    struct IdentityTransform <: AbstractTransform{Float32}
        dimensions::Int
    end

    Base.ndims(transform::IdentityTransform) = transform.dimensions
    transform(::IdentityTransform, x::AbstractArray) = x
    truncate_modes(::IdentityTransform, transformed::AbstractArray) = transformed
    inverse(::IdentityTransform, transformed::AbstractArray, ::AbstractArray) = transformed

end


using .ExternalTransform: IdentityTransform
using Lux, NeuralOperators, Random, Test
import NeuralOperators: inverse, transform, truncate_modes

@testset "Developer interface visibility" begin
    for name in (:AbstractTransform, :transform, :truncate_modes, :inverse)
        @test !Base.isexported(NeuralOperators, name)
        @static if VERSION >= v"1.11"
            @test Base.ispublic(NeuralOperators, name)
        end
    end
end

@testset "External transform implementation" begin
    tform = IdentityTransform(1)
    layer = OperatorConv(
        1 => 1,
        (3,),
        tform;
        init_weight = (rng, T, dims...) -> ones(T, dims...),
    )
    parameters, states = Lux.setup(Xoshiro(0), layer)
    x = reshape(Float32.(1:6), 3, 1, 2)

    y, new_states = layer(x, parameters, states)

    @test eltype(tform) === Float32
    @test ndims(tform) == 1
    @test y == x
    @test new_states == states
end

@testset "Fourier transform implementation" begin
    tform = FourierTransform{ComplexF32}((2,))
    x = reshape(Float32.(1:8), 4, 1, 2)
    transformed = transform(tform, x)

    @test size(transformed) == (3, 1, 2)
    @test size(truncate_modes(tform, transformed)) == (2, 1, 2)
    @test inverse(tform, transformed, x) ≈ x
end
