
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

@inline _alias_type_to_nchan_and_format(::Type{T}) where {T} = @error "Type `$T` is not a valid alias type for a \"CUDA array\" (texture memory) format"
for (T, cuarrayformat) in _type_to_cuarrayformat_dict
    @eval @inline _alias_type_to_nchan_and_format(::Type{$T})  = 1, $cuarrayformat, $T
    @eval @inline _alias_type_to_nchan_and_format(::Type{NTuple{2,$T}}) = 2, $cuarrayformat, $T
    @eval @inline _alias_type_to_nchan_and_format(::Type{NTuple{4,$T}}) = 4, $cuarrayformat, $T
end

@inline function _assert_alias_size(::Type{T}, ::Type{Ta}) where {T,Ta}
    @assert sizeof(T) == sizeof(Ta) "Error: Julia type `$T` cannot be aliased to the \"CUDA array\" (texture memory) format `$Ta`: sizes in bytes do not match"
end

for (T, _) in _type_to_cuarrayformat_dict
    @eval @inline cuda_texture_alias_type(::Type{$T}) = $T
    for N in (2, 4)
        @eval @inline cuda_texture_alias_type(::Type{NT}) where {NT <: NTuple{$N,$T}} = NT
    end
end
    
@generated function cuda_texture_alias_type(t::Type{T}) where T
    err = "An alias from the type `$T` to a \"CUDA array\" (texture memory) format could not been inferred."
    isprimitivetype(T) && return :(@error $("Primitive type `$T` does not have a defined alias to a \"CUDA array\" (texture memory) format."))
    isbitstype(T) || return :(@error $(err * " Type in not `isbitstype`."))
    Te = nothing
    N = 0
    datatypes = DataType[T]
    while !isempty(datatypes)
        for Ti in fieldtypes(pop!(datatypes))
            if isprimitivetype(Ti)
                Ti = cuda_texture_alias_type(Ti)
                typeof(Ti) == DataType || return :(@error $(err * " Composed of primitive type with no alias."))
                if !isprimitivetype(Ti)
                    push!(datatypes, Ti)
                    break
                end
                if Te == nothing
                    Te = Ti
                    N = 1
                elseif Te == Ti
                    N += 1
                else
                    return :(@error $(err * " Incompatible elements."))
                end
            else
                push!(datatypes, Ti)
            end
        end
    end
    Ta = NTuple{N,Te}
    sizeof(T) == sizeof(Ta) || return :(@error $(err * " Inferred alias type and original type have different byte sizes."))
    in(N, (1, 2, 4)) || return :(@error $(err * " Incompatible number of elements ($N, but only 1, 2 or 4 supported)."))
    return Ta                                
end
