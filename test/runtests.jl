using CuTextures
using Test


th,tw = 32, 32
ta = CuTextureArray{Float32}(th,tw)
t = CuTexture(ta)



using CuArrays


a = convert(Array{Float32}, repeat(1:th, 1, tw) + repeat(0.001 * (1:tw)', th, 1))
d_a = CuArray(a)


tex1 = CuTexture{Float32}(th,tw)
copyto!(tex1, d_a)

tex2 = CuTexture(d_a)

tex = tex2


h, w = 10, 30
d_b = CuArray{Float32}(undef, h, w)



using CUDAnative

function kernel_texture_warp_native(texture, dst, h, w)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    j = (blockIdx().y - 1) * blockDim().y + threadIdx().y
    u = Float32(i - 1) / Float32(h - 1);
    v = Float32(j - 1) / Float32(w - 1);
    dst[i,j] = texture[u,v][1];
    return nothing
end
@cuda threads = (h, w) kernel_texture_warp_native(tex, d_b, Float32(h), Float32(w))

d_b

