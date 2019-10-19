using CuTextures
using Test

@testset "CuTextures.jl" begin    
    tm = CuTextureMemory{Float32, 2}((32, 32))
    t = CuTexture{Float32, 2}(tm)
end
