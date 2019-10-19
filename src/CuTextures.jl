module CuTextures

export CuTextureMemory, cutexture_eltype_alias, CuTexture

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
        
        allocateArray_ref = Ref(CUDAdrv.CUDA_ARRAY3D_DESCRIPTOR(
            width, # Width::Csize_t
            height, # Height::Csize_t
            depth, # Depth::Csize_t
            format, # Format::CUarray_format
            num_channels, # NumChannels::UInt32
            0 # Flags::CU_AD_FORMAT_UNSIGNED_INT32
        ))

        handle_ref = Ref{CUarray}(C_NULL)
        cuArray3DCreate(handle_ref, allocateArray_ref)
        
        t = new{T,N}(handle_ref[], dims)
        finalizer(unsafe_free!, t)
        return t
    end
end

function unsafe_free!(t::CuTextureMemory)
    if t.handle != C_NULL
        cuArrayDestroy(t.handle)
        t.handle = C_NULL
    end
    return nothing
end


import CUDAdrv: CUtexObject, cuTexObjectCreate, cuTexObjectDestroy, 
                CUDA_RESOURCE_DESC, CUDA_TEXTURE_DESC, CUDA_RESOURCE_VIEW_DESC,
                CU_RESOURCE_TYPE_ARRAY, CU_TR_ADDRESS_MODE_BORDER, CU_TR_FILTER_MODE_LINEAR,
                CU_TRSF_NORMALIZED_COORDINATES

mutable struct CuTexture{T,N}
    mem::CuTextureMemory{T,N}
    handle::CUtexObject

    function CuTexture{T,N}(texmem::CuTextureMemory{T,N}) where {T,N}
    
        # #### TODO: use CUDAdrv wrapped struct when its padding becomes fixed
        # res = Ref{CUDAdrv.ANONYMOUS1_res}()
        # unsafe_store!(Ptr{CUarray}(pointer_from_objref(res)), texmem.handle)
        # resDesc_ref = Ref(CUDA_RESOURCE_DESC(
        #     CU_RESOURCE_TYPE_ARRAY, # resType::CUresourcetype
        #     res[], # res::ANONYMOUS1_res
        #     0 # flags::UInt32
        # ))
        resDesc_ref = Ref((
            CU_RESOURCE_TYPE_ARRAY, # resType::CUresourcetype
            texmem.handle, # 1 x UInt64
            ntuple(_->Int64(0), 15), # 15 x UInt64
            UInt32(0) # flags::UInt32
        ))
        resDesc_ref = pointer_from_objref(resDesc_ref)
        
        texDesc_ref = Ref(CUDA_TEXTURE_DESC(
            ntuple(_->CU_TR_ADDRESS_MODE_BORDER, 3), # addressMode::NTuple{3, CUaddress_mode}
            CU_TR_FILTER_MODE_LINEAR, # filterMode::CUfilter_mode
            CU_TRSF_NORMALIZED_COORDINATES, # flags::UInt32
            1, # maxAnisotropy::UInt32
            CU_TR_FILTER_MODE_LINEAR, # mipmapFilterMode::CUfilter_mode
            0, # mipmapLevelBias::Cfloat
            0, # minMipmapLevelClamp::Cfloat
            0, # maxMipmapLevelClamp::Cfloat
            ntuple(_->Cfloat(zero(T)), 4), # borderColor::NTuple{4, Cfloat}
            ntuple(_->Cint(0), 12) # reserved::NTuple{12, Cint}
        ))
            
        texObject_ref = Ref{CUtexObject}(0)
        cuTexObjectCreate(texObject_ref, resDesc_ref, texDesc_ref, C_NULL)

        t = new{T,N}(texmem, texObject_ref[])
        finalizer(unsafe_free!, t)
        return t
    end
end

function unsafe_free!(t::CuTexture)
    if t.handle != C_NULL
        cuTexObjectDestroy(t.handle)
        t.handle = C_NULL
    end
    return nothing
end

end # module
