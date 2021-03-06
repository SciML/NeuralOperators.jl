@testset "loss" begin
    ð² = rand(1, 3, 3, 5)
    ð²Ì = rand(1, 3, 3, 5)

    feature_dims = 2:3
    loss = sum(.â(sum(abs2, ð²Ì - ð², dims = feature_dims)))
    y_norm = sum(.â(sum(abs2, ð², dims = feature_dims)))

    @test lâloss(ð²Ì, ð²) â loss / y_norm
end
