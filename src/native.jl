"""
Lightweight type to handle CUDA texture objects inside CUDAnative.jl kernels.
Textures are fetched through indexing operations on `CuTexture`/`CuDeviceTexture` objects, e.g., `cutexture2d[0.2f0, 0.2f0]`.
"""
struct CuDeviceTexture{T,N,Ta}
    handle::CUtexObject
end
CuDeviceTexture{T,N}(handle::CUtexObject) where {T,N} = CuDeviceTexture{T,N,cuda_texture_alias_type(T)}(handle)

Adapt.adapt_storage(::CUDAnative.Adaptor, t::CuTexture{T,N}) where {T,N} = CuDeviceTexture{T,N}(t.handle)


@inline function tex1d(texObject::Int64, x::Float32)::Tuple{Int32,Int32,Int32,Int32}
    Base.llvmcall(("declare [4 x i32] @llvm.nvvm.tex.unified.1d.v4s32.f32(i64, float)",
        "%3 =  call [4 x i32] @llvm.nvvm.tex.unified.1d.v4s32.f32(i64 %0, float %1)\nret [4 x i32] %3"),
        Tuple{Int32,Int32,Int32,Int32},
        Tuple{Int64,Float32}, texObject, x)
end
@inline function tex2d(texObject::Int64, x::Float32, y::Float32)::Tuple{Int32,Int32,Int32,Int32}
    Base.llvmcall(("declare [4 x i32] @llvm.nvvm.tex.unified.2d.v4s32.f32(i64, float, float)",
        "%4 =  call [4 x i32] @llvm.nvvm.tex.unified.2d.v4s32.f32(i64 %0, float %1, float %2)\nret [4 x i32] %4"),
        Tuple{Int32,Int32,Int32,Int32},
        Tuple{Int64,Float32,Float32}, texObject, x, y)
end
@inline function tex3d(texObject::Int64, x::Float32, y::Float32, z::Float32)::Tuple{Int32,Int32,Int32,Int32}
    Base.llvmcall(("declare [4 x i32] @llvm.nvvm.tex.unified.3d.v4s32.f32(i64, float, float, float)",
        "%5 =  call [4 x i32] @llvm.nvvm.tex.unified.3d.v4s32.f32(i64 %0, float %1, float %2, float %3)\nret [4 x i32] %5"),
        Tuple{Int32,Int32,Int32,Int32},
        Tuple{Int64,Float32,Float32,Float32}, texObject, x, y, z)
end
@inline texXD(t::CuDeviceTexture{T,1} where T, x::Real)::Tuple{Int32,Int32,Int32,Int32} = tex1d(reinterpret(Int64, t.handle), convert(Float32, x))
@inline texXD(t::CuDeviceTexture{T,2} where T, x::Real, y::Real)::Tuple{Int32,Int32,Int32,Int32} = tex2d(reinterpret(Int64, t.handle), convert(Float32, x), convert(Float32, y))
@inline texXD(t::CuDeviceTexture{T,3} where T, x::Real, y::Real, z::Real)::Tuple{Int32,Int32,Int32,Int32} = tex3d(reinterpret(Int64, t.handle), convert(Float32, x), convert(Float32, y), convert(Float32, z))


@inline reconstruct(::Type{T}, x::Int32) where {T <: Union{Int32,UInt32,Int16,UInt16,Int8,UInt8}} = unsafe_trunc(T, x)
@inline reconstruct(::Type{Float32}, x::Int32) = reinterpret(Float32, x)
@inline reconstruct(::Type{Float16}, x::Int32) = convert(Float16, reinterpret(Float32, x))

@inline reconstruct(::Type{T}, i32_x4::NTuple{4,Int32}) where T = reconstruct(T, i32_x4[1])
@inline reconstruct(::Type{NTuple{2,T}}, i32_x4::NTuple{4,Int32}) where T = (reconstruct(T, i32_x4[1]),
                                                                             reconstruct(T, i32_x4[2]))
@inline reconstruct(::Type{NTuple{4,T}}, i32_x4::NTuple{4,Int32}) where T = (reconstruct(T, i32_x4[1]),
                                                                             reconstruct(T, i32_x4[2]),
                                                                             reconstruct(T, i32_x4[3]),
                                                                             reconstruct(T, i32_x4[4]))

@inline function cast(::Type{T}, x) where {T}
    @assert sizeof(T) == sizeof(x)
    return unsafe_load(Ptr{T}(pointer_from_objref(Ref(x))))
end

@inline function _getindex(t::CuDeviceTexture{T,N,Ta}, idcs::NTuple{Ni,Real}) where {T,N,Ta,Ni}
    i32_x4 = texXD(t, idcs...)
    cast(T, reconstruct(Ta, i32_x4))
end

@inline Base.getindex(t::CuDeviceTexture{T,1}, x::R) where {T,R <: Real} = _getindex(t, (x,))
@inline Base.getindex(t::CuDeviceTexture{T,2}, x::R, y::R) where {T,R <: Real} = _getindex(t, (x, y))
@inline Base.getindex(t::CuDeviceTexture{T,3}, x::R, y::R, z::R) where {T,R <: Real} = _getindex(t, (x, y, z))
