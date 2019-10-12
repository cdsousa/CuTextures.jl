module CuTextures

export CuTextureMemory, cutexture_eltype_alias

import CUDAnative # TODO: handle CUDA context creation. This is just to create a context already (?)
import CUDAdrv
import CUDAdrv: cuArray3DCreate, CUarray, CUarray_format, CUDA_ARRAY3D_DESCRIPTOR, cuArrayDestroy


const _type_to_cuarrayformat_dict = Dict{DataType,CUarray_format}(
    UInt8 => CUDAdrv.CU_AD_FORMAT_UNSIGNED_INT8,
    UInt16 => CUDAdrv.CU_AD_FORMAT_UNSIGNED_INT16,
    UInt32 => CUDAdrv.CU_AD_FORMAT_UNSIGNED_INT32,
    Int8 => CUDAdrv.CU_AD_FORMAT_SIGNED_INT8,
    Int16 => CUDAdrv.CU_AD_FORMAT_SIGNED_INT16,
    Int32 => CUDAdrv.CU_AD_FORMAT_SIGNED_INT32,
    Float16 => CUDAdrv.CU_AD_FORMAT_HALF,
    Float32 => CUDAdrv.CU_AD_FORMAT_FLOAT,
)

_type_to_cuarrayformat(::Type{T}) where T = @error "Julia type `$T` does not maps to any \"CUarray_format\""
for (type, cuarrayformat) in _type_to_cuarrayformat_dict
    @eval @inline _type_to_cuarrayformat(::Type{$type}) = $cuarrayformat
end
cutexture_eltype_alias(::Type{T}) where T = @error "Julia type `$T` does not have an alias type that can be used as the CUDA texture element type"
for (type, _) in _type_to_cuarrayformat_dict
    @eval @inline cutexture_eltype_alias(::Type{$type}) = $type
end

mutable struct CuTextureMemory{T,N}
    handle::CUarray
    dims::Dims{N}
    
    function CuTextureMemory{T,N}(dims::Dims{N}) where {T, N}
        format = _type_to_cuarrayformat(cutexture_eltype_alias(T))
        num_channels = 1 # TODO enable 1 to 4 channels for NTuple{N,T}-like types
        if N == 2
            width, height = dims
            depth = 0
            @assert 1 <= width "CUDA 2D array (texture) width must be >= 1"
            # @assert witdh <= CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_WIDTH
            @assert 1 <= height "CUDA 2D array (texture) height must be >= 1"
            # @assert height <= CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_HEIGHT
        elseif N == 3
            width, height, depth = dims
            @assert 1 <= width "CUDA 3D array (texture) width must be >= 1"
            # @assert witdh <= CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_WIDTH
            @assert 1 <= height "CUDA 3D array (texture) height must be >= 1"
            # @assert height <= CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_HEIGHT
            @assert 1 <= depth "CUDA 3D array (texture) depth must be >= 1"
            # @assert depth <= CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_DEPTH
        elseif N == 1
            width = dims[1]
            height = depth = 0
            @assert 1 <= width "CUDA 1D array (texture) width must be >= 1"
            # @assert witdh <= CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_WIDTH
        else
            "CUDA arrays (texture memory) can only have 1, 2 or 3 dimensions"
        end
        
        allocateArray_ref = Ref(CUDAdrv.CUDA_ARRAY3D_DESCRIPTOR(width,height,depth,format,num_channels,0))
        handle_ref = Ref{CUarray}(C_NULL)
        cuArray3DCreate(handle_ref, allocateArray_ref)
        
        t = new{T,N}(handle_ref[], dims)
        finalizer(unsafe_free!, t)
        return t
    end
end

function unsafe_free!(t::CuTextureMemory)
    if t.handle != C_NULL
        CUDAdrv.cuArrayDestroy(t.handle)
        t.handle = C_NULL
    end
    return nothing
end
        
end # module
