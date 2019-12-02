


import CUDAdrv: cuArray3DCreate, CUarray, CUarray_format, CUDA_ARRAY3D_DESCRIPTOR, cuArrayDestroy
import CUDAdrv: cuMemcpyDtoA, cuMemcpyHtoA, cuMemcpy2D, CUDA_MEMCPY2D, cuMemcpy3D, CUDA_MEMCPY3D
import CuArrays: CuArray

"""
Type to handle CUDA arrays which are opaque device memory buffers optimized for texture fetching.
The only way to initialize the content of this objects is by copying from host or device arrays using the constructor or `copyto!` calls.
"""
mutable struct CuTextureArray{T,N}
    handle::CUarray
    dims::Dims{N}
    # TODO: hold the CUDA context here so that it is not finalized before this object
    
    function CuTextureArray{T,N}(dims::Dims{N}) where {T,N}
        Ta = cuda_texture_alias_type(T)
        _assert_alias_size(T, Ta)
        nchan, format = _alias_type_to_nchan_and_format(Ta)

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
        
        allocateArray_ref = Ref(CUDA_ARRAY3D_DESCRIPTOR(width, # Width::Csize_t
                                                        height, # Height::Csize_t
                                                        depth, # Depth::Csize_t
                                                        format, # Format::CUarray_format
                                                        UInt32(nchan), # NumChannels::UInt32
                                                        0))

        handle_ref = Ref{CUarray}(C_NULL)
        cuArray3DCreate(handle_ref, allocateArray_ref)
        
        t = new{T,N}(handle_ref[], dims)
        finalizer(unsafe_free!, t)
        return t
    end
end

function unsafe_free!(t::CuTextureArray)
    if t.handle != C_NULL
        cuArrayDestroy(t.handle)
        t.handle = C_NULL
    end
    return nothing
end

CuTextureArray{T}(n::Int) where {T} = CuTextureArray{T,1}((n,))
CuTextureArray{T}(nx::Int, ny::Int) where {T} = CuTextureArray{T,2}((nx, ny))
CuTextureArray{T}(nx::Int, ny::Int, nz::Int) where {T} = CuTextureArray{T,3}((nx, ny, nz))


Base.eltype(tm::CuTextureArray{T,N}) where {T,N} = T
Base.size(tm::CuTextureArray) = tm.dims


### Memory transfer

function Base.copyto!(dst::CuTextureArray{T,1}, src::Union{Array{T,1}, CuArray{T,1}}) where {T}
    @assert dst.dims == size(src) "CuTextureArray and CuArray sizes must match"
    if isa(src, CuArray)
        cuMemcpyDtoA(dst.handle, 0, src.ptr, dst.dims[1] * sizeof(T))
    else
        cuMemcpyHtoA(dst.handle, 0, pointer(src), dst.dims[1] * sizeof(T))
    end
    return dst
end

function Base.copyto!(dst::CuTextureArray{T,2}, src::Union{Array{T,2}, CuArray{T,2}}) where {T}
    @assert dst.dims == size(src) "CuTextureArray and source array sizes must match"
    isdevmem = isa(src, CuArray)
    copy_ref = Ref(CUDA_MEMCPY2D(0, # srcXInBytes::Csize_t
        0, # srcY::Csize_t
        isdevmem ? CUDAdrv.CU_MEMORYTYPE_DEVICE : CUDAdrv.CU_MEMORYTYPE_HOST, # srcMemoryType::CUmemorytype
        isdevmem ? 0 : pointer(src), # srcHost::Ptr{Cvoid}
        isdevmem ? pointer(src) : 0, # srcDevice::CUdeviceptr
        0, # srcArray::CUarray
        0, # srcPitch::Csize_t ### TODO: check why this cannot be `size(src.dims, 1) * sizeof(T)` as it should
        0, # dstXInBytes::Csize_t
        0, # dstY::Csize_t
        CUDAdrv.CU_MEMORYTYPE_ARRAY, # dstMemoryType::CUmemorytype
        0, # dstHost::Ptr{Cvoid}
        0, # dstDevice::CUdeviceptr
        dst.handle, # dstArray::CUarray
        0, # dstPitch::Csize_t
        dst.dims[1] * sizeof(T), # WidthInBytes::Csize_t
        dst.dims[2], # Height::Csize_t
    ))
    cuMemcpy2D(copy_ref)
    return dst
end

function Base.copyto!(dst::CuTextureArray{T,3}, src::Union{Array{T,3}, CuArray{T,3}}) where {T}
    @assert dst.dims == size(src) "CuTextureArray and source array sizes must match"
    isdevmem = isa(src, CuArray)
    copy_ref = Ref(CUDA_MEMCPY3D(0, # srcXInBytes::Csize_t
        0, # srcY::Csize_t
        0, # srcZ::Csize_t
        0, # srcLOD::Csize_t
        isdevmem ? CUDAdrv.CU_MEMORYTYPE_DEVICE : CUDAdrv.CU_MEMORYTYPE_HOST, # srcMemoryType::CUmemorytype
        isdevmem ? 0 : pointer(src), # srcHost::Ptr{Cvoid}
        isdevmem ? pointer(src) : 0, # srcDevice::CUdeviceptr
        0, # srcArray::CUarray
        0, # reserved0::Ptr{Cvoid}
        size(src, 1) * sizeof(T), # srcPitch::Csize_t
        size(src, 2), # srcHeight::Csize_t
        0, # dstXInBytes::Csize_t
        0, # dstY::Csize_t
        0, # dstZ::Csize_t
        0, # dstLOD::Csize_t
        CUDAdrv.CU_MEMORYTYPE_ARRAY, # dstMemoryType::CUmemorytype
        0, # dstHost::Ptr{Cvoid}
        0, # dstDevice::CUdeviceptr
        dst.handle, # dstArray::CUarray
        0, # reserved1::Ptr{Cvoid}
        0, # dstPitch::Csize_t
        0, # dstHeight::Csize_t
        dst.dims[1] * sizeof(T), # WidthInBytes::Csize_t
        dst.dims[2], # Height::Csize_t
        dst.dims[3], # Depth::Csize_t
    ))
    cuMemcpy3D(copy_ref)
    return dst
end


function CuTextureArray(a::Union{Array{T,N}, CuArray{T,N}}) where {T,N}
    t = CuTextureArray{T}(size(a)...)
    copyto!(t, a)
    return t
end
