
import CUDAdrv: CUtexObject, cuTexObjectCreate, cuTexObjectDestroy, 
CUDA_RESOURCE_DESC, CUDA_TEXTURE_DESC, CUDA_RESOURCE_VIEW_DESC,
CU_RESOURCE_TYPE_ARRAY, CU_RESOURCE_TYPE_LINEAR, CU_RESOURCE_TYPE_PITCH2D,
CU_TR_ADDRESS_MODE_BORDER, CU_TR_ADDRESS_MODE_CLAMP, CU_TR_FILTER_MODE_LINEAR,
CU_TRSF_NORMALIZED_COORDINATES, CU_TRSF_READ_AS_INTEGER

import CuArrays: CuArray

"""
Type to handle CUDA texture objects. These objects do not hold data by themselves,
but instead are bound either to `CuTextureArray`s (CUDA arrays) or to `CuArray`s.
(Note: For correct wrapping `CuArray`s it is necessary the their memory is well aligned and strided (good pitch).
Currently, that is not being enforced.)

Theses objects are meant to be used to do texture fetchts inside CUDAnative.jl kernels.
When passed to CUDAnative.jl kernels, `CuTexture` objects are transformed into `CuDeviceTexture`s objects.
"""
mutable struct CuTexture{T,N,Mem}
    mem::Mem
    handle::CUtexObject

    function CuTexture{T,N,Mem}(texmemory::Mem) where {T,N,Mem}
        # TODO: add support to choose normalized or non-normalized coordinate access
        # TODO: add support to choose nearest-neighbor or linear interpolation 
        # TODO: add support to choose address mode: clamp, border, etc.

        Ta = cuda_texture_alias_type(T)
        _assert_alias_size(T, Ta)
        nchan, format, Te = _alias_type_to_nchan_and_format(Ta)

        resDesc_ref = _construct_CUDA_RESOURCE_DESC(texmemory)
        resDesc_ref = pointer_from_objref(resDesc_ref)

        addressMode = CU_TR_ADDRESS_MODE_CLAMP  # N > 1 ? CU_TR_ADDRESS_MODE_BORDER : CU_TR_ADDRESS_MODE_CLAMP
        flags = CU_TRSF_NORMALIZED_COORDINATES
        flags = flags | (Te <: Integer ? CU_TRSF_READ_AS_INTEGER : zero(CU_TRSF_READ_AS_INTEGER))

        texDesc_ref = Ref(CUDA_TEXTURE_DESC(ntuple(_->addressMode, 3), # addressMode::NTuple{3, CUaddress_mode}
                                            CU_TR_FILTER_MODE_LINEAR, # filterMode::CUfilter_mode
                                            flags, # flags::UInt32
                                            1, # maxAnisotropy::UInt32
                                            CU_TR_FILTER_MODE_LINEAR, # mipmapFilterMode::CUfilter_mode
                                            0, # mipmapLevelBias::Cfloat
                                            0, # minMipmapLevelClamp::Cfloat
                                            0, # maxMipmapLevelClamp::Cfloat
                                            ntuple(_->Cfloat(zero(Te)), 4), # borderColor::NTuple{4, Cfloat}
                                            ntuple(_->Cint(0), 12)))

        texObject_ref = Ref{CUtexObject}(0)
        cuTexObjectCreate(texObject_ref, resDesc_ref, texDesc_ref, C_NULL)

        t = new{T,N,Mem}(texmemory, texObject_ref[])
        finalizer(unsafe_free!, t)
        return t
    end
end

function _construct_CUDA_RESOURCE_DESC(texarr::CuTextureArray{T,N}) where {T,N}
    # #### TODO: use CUDAdrv wrapped struct when its padding becomes fixed
    # res = Ref{CUDAdrv.ANONYMOUS1_res}()
    # unsafe_store!(Ptr{CUarray}(pointer_from_objref(res)), texarr.handle)
    # resDesc_ref = Ref(CUDA_RESOURCE_DESC(
    #     CU_RESOURCE_TYPE_ARRAY, # resType::CUresourcetype
    #     res[], # res::ANONYMOUS1_res
    #     0 # flags::UInt32
    # ))
    resDesc_ref = Ref((CU_RESOURCE_TYPE_ARRAY, # resType::CUresourcetype
                        texarr.handle, # 1 x UInt64
                        ntuple(_->Int64(0), 15), # 15 x UInt64
                        UInt32(0)))
    return resDesc_ref
end

function _construct_CUDA_RESOURCE_DESC(arr::CuArray{T,N}) where {T,N}
# TODO: take care of allowed pitches
    @assert 1 <= N <= 2 "Only 1D or 2D (dimension) CuArray objects can be wrapped in a texture"
    
    Ta = cuda_texture_alias_type(T)
    _assert_alias_size(T, Ta)
    nchan, format, Te = _alias_type_to_nchan_and_format(Ta)

# #### TODO: use CUDAdrv wrapped struct when its padding becomes fixed
    resDesc_ref = Ref(((N == 1 ? CU_RESOURCE_TYPE_LINEAR : CU_RESOURCE_TYPE_PITCH2D), # resType::CUresourcetype
                        arr.buf.ptr, # 1 x UInt64 (CUdeviceptr)
                        format, # 1/2 x UInt64 (CUarray_format)
                        UInt32(nchan), # 1/2 x UInt64
                        (N == 2 ? size(arr, 1) : size(arr, 1) * sizeof(T)), # 1 x UInt64 nx
                        (N == 2 ? size(arr, 2) : 0), # 1 x UInt64 ny
                        (N == 2 ? size(arr, 1) * sizeof(T) : 0), # 1 x UInt64 pitch
                        ntuple(_->Int64(0), 11), # 11 x UInt64
                        UInt32(0)))
    return resDesc_ref
end

function unsafe_free!(t::CuTexture)
    if t.handle != C_NULL
        cuTexObjectDestroy(t.handle)
        t.handle = C_NULL
    end
    return nothing
end


CuTexture(texarr::CuTextureArray{T,N}) where {T,N} = CuTexture{T,N,CuTextureArray{T,N}}(texarr)
CuTexture{T}(n::Int) where {T} = CuTexture(CuTextureArray{T,1}((n,)))
CuTexture{T}(nx::Int, ny::Int) where {T} = CuTexture(CuTextureArray{T,2}((nx, ny)))
CuTexture{T}(nx::Int, ny::Int, nz::Int) where {T} = CuTexture(CuTextureArray{T,3}((nx, ny, nz)))
CuTexture(cuarr::CuArray{T,N}) where {T,N} = CuTexture{T,N,CuArray{T,N}}(cuarr)


Base.eltype(tm::CuTexture{T,N}) where {T,N} = T
Base.size(tm::CuTexture) = size(tm.mem)