using CuTextures
using CuArrays
using CUDAnative

using Test



@inline function calcpoint(blockIdx, blockDim, threadIdx, size)
    i = (blockIdx - 1) * blockDim + threadIdx
    x = (Float32(i) - 0.5f0) / Float32(size)
    return i, x
end
function kernel_texture_warp_native(dst::CuDeviceArray{T,1}, texture::CuDeviceTexture{T,1}, h) where {T}
    i, u = calcpoint(blockIdx().x, blockDim().x, threadIdx().x, h)
    dst[i] = texture[u];
    return nothing
end
function kernel_texture_warp_native(dst::CuDeviceArray{T,2}, texture::CuDeviceTexture{T,2}, h, w) where {T}
    i, u = calcpoint(blockIdx().x, blockDim().x, threadIdx().x, h)
    j, v = calcpoint(blockIdx().y, blockDim().y, threadIdx().y, w)
    dst[i,j] = texture[u,v];
    return nothing
end
function kernel_texture_warp_native(dst::CuDeviceArray{T,3}, texture::CuDeviceTexture{T,3}, h, w, d) where {T}
    i, u = calcpoint(blockIdx().x, blockDim().x, threadIdx().x, h)
    j, v = calcpoint(blockIdx().y, blockDim().y, threadIdx().y, w)
    k, w = calcpoint(blockIdx().z, blockDim().z, threadIdx().z, d)
    dst[i,j,k] = texture[u,v,w];
    return nothing
end

function fetch_all(texture)
    dims = size(texture)
    d_out = CuArray{eltype(texture)}(undef, dims...)
    @cuda threads = dims kernel_texture_warp_native(d_out, texture, Float32.(dims)...)
    d_out
end



testheight, testwidth, testdepth = 16, 16, 4
a1D = convert(Array{Float32}, 1:testheight)
a2D = convert(Array{Float32}, repeat(1:testheight, 1, testwidth) + repeat(0.01 * (1:testwidth)', testheight, 1))
a3D = convert(Array{Float32}, repeat(a2D, 1, 1, testdepth))
for k = 1:testdepth; a3D[:,:,k] .+= 0.0001 * k; end
d_a1D = CuArray(a1D)
d_a2D = CuArray(a2D)
d_a3D = CuArray(a3D)

@testset "Use CuTextureArray" begin
    texarr1D = CuTextureArray{Float32}(testheight)
    copyto!(texarr1D, d_a1D)
    tex1D = CuTexture(texarr1D)
    fetched1D = fetch_all(tex1D)
    @test fetched1D == d_a1D

    texarr2D = CuTextureArray{Float32}(testheight, testwidth)
    copyto!(texarr2D, d_a2D)
    tex2D = CuTexture(texarr2D)
    fetched2D = fetch_all(tex2D)
    @test fetched2D == d_a2D 

    texarr3D = CuTextureArray{Float32}(testheight, testwidth, testdepth)
    copyto!(texarr3D, d_a3D)
    tex3D = CuTexture(texarr3D)
    fetched3D = fetch_all(tex3D)
    @test fetched3D == d_a3D  
end

@testset "Wrap CuArray" begin
    texwrap1D = CuTexture(d_a1D)
    fetched1D = fetch_all(texwrap1D)
    @test_broken fetched1D == d_a1D  

    texwrap2D = CuTexture(d_a2D)
    fetched2D = fetch_all(texwrap2D)
    @test fetched2D == d_a2D 
end

##

@testset "All CUDA types" begin

    for T in (Int32,UInt32,Int16,UInt16,Int8,UInt8,Float32,Float16)
        testheight, testwidth, testdepth = 32,32, 4
        a2D = rand(T, testheight, testwidth)
        d_a2D = CuArray(a2D)

        tex_2D = CuTexture(CuTextureArray{T}(testheight, testwidth))
        copyto!(tex_2D.mem, d_a2D)

        # tex_2D = CuTexture(d_a2D)

        fetched2D = fetch_all(tex_2D)
        @test fetched2D == d_a2D
    end

end