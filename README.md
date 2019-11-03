# CuTextures

**CUDA textures (CUDA Arrays) interface for native Julia**

CUDA Textures are handled though two main types, `CuTextureArray` and `CuTexture`.

`CuTextureArray` is a type to handle CUDA arrays: opaque device memory buffers optimized for texture fetching.
The only way to initialize the content of these objects is by copying from host or device arrays using the constructor or `copyto!` calls.

`CuTexture` is a type to handle CUDA texture objects. These objects do not hold data by themselves,
but instead are bound either to `CuTextureArray`s (CUDA arrays) or to `CuArray`s (device linear memory). `CuArray`s must have the memory well aligned (good pitch) for correct wrapping.

`CuTexture` objects are meant to be used to do texture fetching inside *CUDAnative.jl* kernels.
When passed to *CUDAnative.jl* kernels, `CuTexture` objects are transformed into lightweight `CuDeviceTexture` objects.
Fetching (sampling) to textures from within the kernels can then be done through indexing operations on the `CuTexture`/`CuDeviceTexture` objects, like `interpolatedval = sometexture2d[0.2f0, 0.9f0]`.



### To do

- Assert good alignment when wrapping `CuArray`s.
- Deal with CUDA contexts.
- Add support to choose texture normalized or non-normalized coordinate access.
- Add support to choose texture nearest-neighbor or linear interpolation.
- Add support to choose texture address mode: clamp, border, etc.
- Improve code using wrapped CUDA drive API C structures.
- Check potential performance optimizations
- Check potential better (more optimized) ways to wrap fetch intrinsics (currently relying on `llvm.nvvm`)