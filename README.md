# CuTextures

**CUDA textures ("CUDA arrays") interface for native Julia**

CUDA Textures are handled though two main types, `CuTextureArray` and `CuTexture`.

`CuTextureArray` is a type to handle CUDA arrays: opaque device memory buffers optimized for texture fetching.
The only way to initialize the content of these objects is by copying from host or device arrays using the constructor or `copyto!` calls.

`CuTexture` is a type to handle CUDA texture objects. These objects do not hold data by themselves,
but instead are bound either to `CuTextureArray`s (CUDA arrays) or to `CuArray`s (device linear memory). `CuArray`s must have the memory well aligned (good pitch) for correct wrapping.

`CuTexture` objects are meant to be used to do texture fetching inside *CUDAnative.jl* kernels.
When passed to *CUDAnative.jl* kernels, `CuTexture` objects are transformed into lightweight `CuDeviceTexture` objects.
Fetching (sampling) to textures from within the kernels can then be done through indexing operations on the `CuTexture`/`CuDeviceTexture` objects, like `interpolatedval = sometexture2d[0.2f0, 0.9f0]`.

CUDA textures elements are limited to a set of supported element types: `Float32`, `Float16`, `Int32`, `UInt32`, `Int16`, `UInt16`, `Int8` and `UInt8`, which can be packed as single elements or in 2 or 4 channels just like if they were NTuples of 2 or 4 elements.
`CuTextures` is able to cast to and from Julia types that are composed of compatible types. For that, `CuTextures` must be informed of the "alias" type by overloading the function `CuTextures.cuda_texture_alias_type`. For example, for the *FixedPointNumbers.jl* type `N0f8`: `CuTextures.cuda_texture_alias_type(::Type{N0f8}) = UInt8`, and for a `RBGA{N0f8}` pixel type from *ColorTypes.jl*: `CuTextures.cuda_texture_alias_type(::Type{RGBA{N0f8}}) = NTuple{4,UInt8}`


## To do

- Assert good alignment when wrapping `CuArray`s.
- Deal with CUDA contexts.
- Add support to choose non-normalized coordinate access instead of normalized access.
- Add support to choose texture nearest-neighbor interpolation instead of linear interpolation.
- Add support to choose texture address mode: clamp, border, etc., along with support to define the out-of-bounds value.
- Improve code using wrapped CUDA drive API C structures.
- Check potential performance optimizations
- Check potential better (more optimized) ways to wrap fetch intrinsics (currently relying on `llvm.nvvm`)
- Check potential better (more optimized) ways to cast Julia types to and from CUDA texture formats


## Usage example

```julia
using Images, TestImages, ColorTypes, FixedPointNumbers
using CuArrays, CUDAnative
using CuTextures

# Get the input image. Use RGBA to have 4 channels since CUDA textures can have only 1, 2 or 4 channels.
img = RGBA{N0f8}.(testimage("lighthouse"))

# Tell CuTextures the alias type of RGBA{N0f8} in the CUDA textures world
CuTextures.cuda_texture_alias_type(::Type{RGBA{N0f8}}) = NTuple{4,UInt8}

# Create a texture memory object (CUDA array) and initilaize it with the input image content (from host).
texturearray = CuTextureArray(img)

# Create a texture object and bind it to the texture memory created above
texture = CuTexture(texturearray)

# Define an image warping kernel
function warp(dst, texture)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    j = (blockIdx().y - 1) * blockDim().y + threadIdx().y
    u = (Float32(i) - 1f0) / (Float32(size(dst, 1)) - 1f0)
    v = (Float32(j) - 1f0) / (Float32(size(dst, 2)) - 1f0)
    x = u + 0.02f0 * CUDAnative.sin(30v)
    y = v + 0.03f0 * CUDAnative.sin(20u)
    @inbounds dst[i,j] = texture[x,y]
    return nothing
end

# Create a 500x1000 CuArray for the output (warped) image
outimg_d = CuArray{eltype(img)}(undef, 500, 1000)

# Execute the kernel
@cuda threads = (size(outimg_d, 1), 1) blocks = (1, size(outimg_d, 2)) warp(outimg_d, texture)

# Get the output image into host memory and save it to a file
outimg = Array(outimg_d)
save("imgwarp.png", outimg)
```

- Input image:

   ![](https://testimages.juliaimages.org/images/lighthouse.png)

- Warped image:

   ![](examples/imgwarp.png)

