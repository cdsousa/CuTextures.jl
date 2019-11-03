
import CUDAdrv: CUarray_format

const _type_to_cuarrayformat_dict = Dict{DataType,CUarray_format}(UInt8 => CUDAdrv.CU_AD_FORMAT_UNSIGNED_INT8,
UInt16 => CUDAdrv.CU_AD_FORMAT_UNSIGNED_INT16,
UInt32 => CUDAdrv.CU_AD_FORMAT_UNSIGNED_INT32,
Int8 => CUDAdrv.CU_AD_FORMAT_SIGNED_INT8,
Int16 => CUDAdrv.CU_AD_FORMAT_SIGNED_INT16,
Int32 => CUDAdrv.CU_AD_FORMAT_SIGNED_INT32,
Float16 => CUDAdrv.CU_AD_FORMAT_HALF,
Float32 => CUDAdrv.CU_AD_FORMAT_FLOAT,
)

@inline _alias_type_to_nchan_and_eltype(::Type{NTuple{N,T}}) where {N,T} = @error "Julia type `$T` (from `NTuple{$N,$T}`) does not have an alias to a \"CUDA array\" (texture memory) format"
@inline _alias_type_to_nchan_and_eltype(::Type{T}) where {T} = @error "Julia type `$T` does not have an alias to a \"CUDA array\" (texture memory) format"
for (T, cuarrayformat) in _type_to_cuarrayformat_dict
    @eval @inline _alias_type_to_nchan_and_eltype(::Type{$T})  = 1, $T
    @eval @inline _alias_type_to_nchan_and_eltype(::Type{NTuple{2,$T}}) = 2, $T
    @eval @inline _alias_type_to_nchan_and_eltype(::Type{NTuple{4,$T}}) = 4, $T
    @eval @inline _alias_type_to_nchan_and_eltype(::Type{NTuple{N,$T}}) where {N} = @error "\"CUDA arrays\" (texture memory) can have only 1, 2 or 4 channels"
    @eval @inline _type_to_cuarrayformat(::Type{$T}) = $cuarrayformat
end


cuda_texture_alias_type(::Type{T}) where {T} = @error "Type `$T` does not have a defined alias to a \"CUDA array\" (texture memory) format"
for (T, _) in _type_to_cuarrayformat_dict
    @eval @inline cuda_texture_alias_type(::Type{$T}) = $T
    for N in (2, 4)
        @eval @inline cuda_texture_alias_type(::Type{NT}) where {NT <: NTuple{$N,$T}} = NT
    end
end

@inline function _cuda_texture_alias_type_with_asserted_size(::Type{T}) where {T}
    Ta = cuda_texture_alias_type(T)
    @assert sizeof(Ta) == sizeof(T) "Error in the alias of Julia type `$T` to the \"CUDA array\" (texture memory) format `$Ta`: sizes in bytes do not match"
    return Ta
end