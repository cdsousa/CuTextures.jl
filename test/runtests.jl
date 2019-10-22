using CuTextures
using Test

tm = CuTextureMemory{Float32, 2}((32, 32))
t = CuTexture{Float32, 2}(tm)



using CUDAnative
using CuArrays

h, w = 10, 30
d_a = CuArray{Float32}(undef, h, w)

function kernel_texture_warp_native(texture, dst, h, w)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    j = (blockIdx().y - 1) * blockDim().y + threadIdx().y
    u = Float32(i - 1) / Float32(h - 1);
    v = Float32(j - 1) / Float32(w - 1);
    dst[i,j] = texture[u, v][1];
    return nothing
end
@cuda threads = (h, w) kernel_texture_warp_native(t, d_a, Float32(h), Float32(w))


