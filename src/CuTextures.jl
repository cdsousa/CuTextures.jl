module CuTextures

export CuTextureArray, CuTexture, CuDeviceTexture, cuda_texture_alias_type

import CUDAdrv
import CUDAnative # TODO: handle CUDA context creation. This is just to create a context already (?)
import CuArrays
import Adapt

include("formats.jl")
include("texturearray.jl")
include("texture.jl")
include("native.jl")

end # module
