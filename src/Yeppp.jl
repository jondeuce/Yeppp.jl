module Yeppp

# @unix_only    const libyeppp = "libyeppp"
# @windows_only const libyeppp = "yeppp"

using Libdl # BinDeps injects a dependency on Libdl

if isfile(joinpath(dirname(@__FILE__),"..","deps","deps.jl"))
    include("../deps/deps.jl")
else
    error("Yeppp not properly installed. Please run Pkg.build(\"Yeppp\")")
end

function __init__()
    status = ccall( (:yepLibrary_Init, libyeppp), Cint, ())
    status != 0 && error("yepLibrary_Init: error: ", status)
end

function release()
    status = ccall( (:yepLibrary_Release, libyeppp), Cint, ())
    status != 0 && error("yepLibrary_Release: error: ", status)
end

_iscontiguous(A::DenseArray) = true
# Fallback
_iscontiguous(A::StridedArray) = false
# Cases we can verify
_iscontiguous(A::SubArray) = Base.iscontiguous(A)
_iscontiguous(A::StridedVector) = stride(A,1) == 1

macro yepppfunsAA_A(fname, libname, BT)
    errorname = libname * ": error: "
    quote
        global $(fname)
        function $(fname)(res::Union{DenseArray{$(BT)},StridedArray{$(BT)}}, x::Union{DenseArray{$(BT)},StridedArray{$(BT)}}, y::Union{DenseArray{$(BT)},StridedArray{$(BT)}})
            n = length(x)
            @assert(n == length(y) == length(res))
            @assert !Base.has_offset_axes(res, x, y)
            @assert _iscontiguous(res) && _iscontiguous(x) && _iscontiguous(y) "could not verify that arguments are contigous is memory"

            status = ccall( ($(libname), libyeppp), Cint, (Ptr{$(BT)}, Ptr{$(BT)}, Ptr{$(BT)}, Culong), x, y, res, n)
            status != 0 && error($(errorname), status)
            res
        end
    end
end

macro yepppfunsA_A(fname, libname, BT)
    errorname = libname * ": error: "
    quote
        global $(fname)
        function $(fname)(res::Union{DenseArray{$(BT)},StridedArray{$(BT)}}, x::Union{DenseArray{$(BT)},StridedArray{$(BT)}})
            n = length(x)
            @assert(n == length(res))
            @assert !Base.has_offset_axes(res, x)
            @assert _iscontiguous(res) && _iscontiguous(x) "could not verify that arguments are contigous is memory"

            status = ccall( ($(libname), libyeppp), Cint, (Ptr{$(BT)}, Ptr{$(BT)}, Culong), x, res, n)
            status != 0 && error($(errorname), status)
            res
        end
    end
end

macro yepppfunsA_S(fname, libname, BT)
    errorname = libname * ": error: "
    quote
        global $(fname)
        function $(fname)(x::Union{DenseArray{$(BT)},StridedArray{$(BT)}})
            n = length(x)
            @assert !Base.has_offset_axes(x)
            @assert _iscontiguous(x) "could not verify that arguments are contigous is memory"
            res = Ref{$BT}()
            status = ccall( ($(libname), libyeppp), Cint, (Ptr{$(BT)}, Ptr{$(BT)}, Culong), x, res, n)
            status != 0 && error($(errorname), status)
            res[]
        end
    end
end

## == Float64 == ##
@yepppfunsAA_A add! "yepCore_Add_V64fV64f_V64f" Float64
@yepppfunsAA_A multiply! "yepCore_Multiply_V64fV64f_V64f" Float64
@yepppfunsAA_A subtract! "yepCore_Subtract_V64fV64f_V64f" Float64
@yepppfunsAA_A max! "yepCore_Max_V64fV64f_V64f" Float64
@yepppfunsAA_A min! "yepCore_Min_V64fV64f_V64f" Float64

@yepppfunsA_A log! "yepMath_Log_V64f_V64f" Float64
@yepppfunsA_A exp! "yepMath_Exp_V64f_V64f" Float64
@yepppfunsA_A sin! "yepMath_Sin_V64f_V64f" Float64
@yepppfunsA_A cos! "yepMath_Cos_V64f_V64f" Float64
@yepppfunsA_A tan! "yepMath_Tan_V64f_V64f" Float64

@yepppfunsA_S sum "yepCore_Sum_V64f_S64f" Float64
@yepppfunsA_S sumabs "yepCore_SumAbs_V64f_S64f" Float64
@yepppfunsA_S sumabs2 "yepCore_SumSquares_V64f_S64f" Float64

## == Float32 == ##
@yepppfunsAA_A add! "yepCore_Add_V32fV32f_V32f" Float32
@yepppfunsAA_A multiply! "yepCore_Multiply_V32fV32f_V32f" Float32
@yepppfunsAA_A subtract! "yepCore_Subtract_V32fV32f_V32f" Float32
@yepppfunsAA_A max! "yepCore_Max_V32fV32f_V32f" Float32
@yepppfunsAA_A min! "yepCore_Min_V32fV32f_V32f" Float32

@yepppfunsA_S sum "yepCore_Sum_V32f_S32f" Float32
@yepppfunsA_S sumabs "yepCore_SumAbs_V32f_S32f" Float32
@yepppfunsA_S sumabs2 "yepCore_SumSquares_V32f_S32f" Float32

"""
    dot(x::Union{DenseVector{Float64},StridedVector{Float64}}, y::Union{DenseVector{Float64},StridedVector{Float64}})

Compute the dot product of x and y.
"""
function dot(x::Union{DenseVector{Float64},StridedVector{Float64}}, y::Union{DenseVector{Float64},StridedVector{Float64}})
    n = length(x)
    @assert(n == length(y))
    @assert !Base.has_offset_axes(x, y)
    @assert _iscontiguous(x) && _iscontiguous(y)

    dotproduct = Ref{Float64}()
    status = ccall( (:yepCore_DotProduct_V64fV64f_S64f, libyeppp), Cint, (Ptr{Float64}, Ptr{Float64}, Ptr{Float64}, Culong), x, y, dotproduct, n)
    status != 0 && error("yepCore_DotProduct_V64fV64f_S64f: error: ", status)
    dotproduct[]
end

"""
    dot(x::Union{DenseVector{Float32},StridedVector{Float32}}, y::Union{DenseVector{Float32},StridedVector{Float32}})

Compute the dot product of x and y.
"""
function dot(x::Union{DenseVector{Float32},StridedVector{Float32}}, y::Union{DenseVector{Float32},StridedVector{Float32}})
    n = length(x)
    @assert(n == length(y))
    @assert !Base.has_offset_axes(x, y)
    @assert _iscontiguous(x) && _iscontiguous(y)

    dotproduct = Ref{Float32}()
    status = ccall( (:yepCore_DotProduct_V32fV32f_S32f, libyeppp), Cint, (Ptr{Float32}, Ptr{Float32}, Ptr{Float32}, Culong), x, y, dotproduct, n)
    status != 0 && error("yepCore_DotProduct_V32fV32f_S32f: error: ", status)
    dotproduct[]
end

"""
    max(x, y)

Compares the vectors `x` and `y` and return the element wise maxima.
"""
max(x, y) = max!(similar(x), x, y)

"""
    max(x, y)

Compares the vectors `x` and `y` and return the element wise minima.
"""
min(x, y) = min!(similar(x), x, y)

function evalpoly!(res::Union{DenseArray{Float64},StridedVector{Float64}}, coef::Union{DenseArray{Float64},StridedVector{Float64}}, x::Union{DenseArray{Float64},StridedVector{Float64}})
    n = length(coef)
    arraysize = length(x)
    @assert(arraysize == length(res))
    @assert !Base.has_offset_axes(res, coef, x)
    @assert _iscontiguous(res) && _iscontiguous(coef) && _iscontiguous(x)

    status = ccall( (:yepMath_EvaluatePolynomial_V64fV64f_V64f, libyeppp), Cint, (Ptr{Float64}, Ptr{Float64}, Ptr{Float64}, Culong, Culong), coef, x, res, n, arraysize)
    status != 0 && error("yepMath_EvaluatePolynomial_V64fV64f_V64f: error: ", status)
    res
end

"""
    evalpoly(coeff, x)

Evaluates polynomial with double precision (64-bit) floating-point coefficients `coeff` on an array of double precision (64-bit) floating-point elements `x`.
"""
evalpoly(coef, x) = evalpoly!(similar(x), coef, x)

"""
    add(x, y)

Perform element wise addition of the two array `x` and `y`.
"""
add(x, y) = add!(similar(x), x, y)

"""
    subtract(x, y)

Perform element wise subtraction of the two array `x` and `y`.
"""
subtract(x, y) = subtract!(similar(x), x, y)

"""
    multiply(x, y)

Perform element wise multiplication of the two array `x` and `y`.
"""
multiply(x, y) = multiply!(similar(x), x, y)

"""
    log(x)

Returns the element wise natural logarithm of `x`.
"""
log(x) = log!(similar(x), x)

"""
    log!(x)

Computes the element wise natural logarithm of `x` inplace.
"""
log!(x) = log!(x, x)

"""
    exp(x)

Returns the element wise exponential of `x`.
"""
exp(x) = exp!(similar(x), x)

"""
    exp(x)

Computes the element wise exponential of `x` inplace.
"""
exp!(x) = exp!(x, x)

"""
    sin(x)

Returns element wise `sin` of `x`.
"""
sin(x) = sin!(similar(x), x)

"""
    sin(x)

Computes element wise `sin` of `x` inplace.
"""
sin!(x) = sin!(x, x)

"""
    cos(x)

Returns element wise `cos` of `x`.
"""
cos(x) = cos!(similar(x), x)

"""
    cos(x)

Returns element wise `cos` of `x`.
"""
cos!(x) = cos!(x, x)

"""
    tan(x)

Returns element wise `tan` of `x`.
"""
tan(x) = tan!(similar(x), x)

"""
    tan(x)

Returns element wise `tan` of `x`.
"""
tan!(x) = tan!(x, x)
end # module
