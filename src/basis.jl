"""
    AbstractBasis{N}
abstract supertype for a function basis with `N` functions
"""
abstract type AbstractBasis{N} end

"""
    AbstractBasis1D{N}
abstract supertype for a 1D function basis with `N` functions
"""
abstract type AbstractBasis1D{N} <: AbstractBasis{N} end

"""
    LagrangePolynomialBasis{N} <: AbstractBasis1D{N}
A basis of `N` Lagrange polynomials. The polynomial order is `N + 1`.
# Fields
    - `funcs::PolynomialSystem{N,1}` - a system of static polynomials
    - `points::StaticVector{N}` - a static vector of support points
"""
struct LagrangePolynomialBasis{NFuncs} <: AbstractBasis1D{NFuncs}
    funcs::SP.PolynomialSystem{NFuncs,1}
    points::SVector{NFuncs}
    function LagrangePolynomialBasis(funcs::AbstractVector{DP.Polynomial{C,T1}},
        points::AbstractVector{T2}) where {C,T1,T2}

        NFuncs = length(funcs)
        order = NFuncs - 1
        npoints = length(points)
        if NFuncs < 2
            msg = "Number of functions must be greater than 1"
            throw(ArgumentError(msg))
        end
        if NFuncs != length(funcs)
            length_funcs = length(funcs)
            msg = "Expected $NFuncs functions, got $length_funcs"
            throw(ArgumentError(msg))
        end
        if NFuncs != npoints
            msg = "Number of functions must be equal to number of points"
            throw(ArgumentError(msg))
        end
        if !isapprox(sum(funcs),1.0)
            val = sum(funcs)
            msg = "Polynomial basis must sum to 1.0, got $val"
        end
        vars = funcs[1].x.vars[1]
        for f in funcs
            if length(f.x.vars) != 1
                nvars = length(f.x.vars)
                msg = "Require univariate polynomials, got $nvars variables"
                throw(ArgumentError(msg))
            end
            if !(vars in f.x.vars)
                msg = "Polynomial functions must have the same variables"
                throw(ArgumentError(msg))
            end
            ord = maximum(maximum.(f.x.Z))
            if ord != order
                msg = "Polynomial functions must have the same order"
                throw(ArgumentError(msg))
            end
        end
        polysystem = SP.PolynomialSystem(funcs)
        static_points = SVector{NFuncs,T2}(points)
        new{NFuncs}(polysystem,static_points)
    end
end

"""
    LagrangePolynomialBasis(x::DP.PolyVar, order::Int; start = -1.0, stop = 1.0)
construct a polynomial basis with variable `x` of order `order` with
equally spaced points between `start` and `stop`.
"""
function LagrangePolynomialBasis(x::DP.PolyVar, order::Int;
    start = -1.0, stop = 1.0)

    NFuncs = order + 1
    roots = range(start, stop = stop, length = NFuncs)
    basis = lagrange_polynomials(x,roots)
    return LagrangePolynomialBasis(basis,roots)
end

"""
    TensorProductBasis{N,NFuncs,T<:AbstractBasis1D} <: AbstractBasis{NFuncs}
construct an `N` dimensional polynomial basis by tensor product of `T`
with a total of `NFuncs` functions.
Note that `NFuncs` can be inferred from `T{nfuncs}`.
# Fields
    - `basis::AbstractBasis1D` the underlying 1D basis
"""
struct TensorProductBasis{N,NFuncs,T<:AbstractBasis1D} <: AbstractBasis{NFuncs}
    basis::AbstractBasis1D
    function TensorProductBasis{N}(B::T) where {N,T<:AbstractBasis1D{nfuncs}} where {nfuncs}
        if !(N > 1)
            msg = "Tensor product construction requires dimension > 1, got $N"
            throw(ArgumentError(msg))
        end
        NFuncs = nfuncs^N
        new{N,NFuncs,T}(B)
    end
end

"""
    TensorProductBasis(x::DP.PolyVar, dim::Int, order::Int; start = -1.0, stop = 1.0)
construct a `dim` dimensional polynomial basis with variable `x` using a tensor
product of `order` order `LagrangePolynomialBasis`. The polynomials are
equispaced from `start` to `stop`.
"""
function TensorProductBasis(x::DP.PolyVar, dim::Int,order::Int; start = -1.0, stop = 1.0)
    basis_1d = LagrangePolynomialBasis(x, order, start = start, stop = stop)
    return TensorProductBasis{dim}(basis_1d)
end

"""
    (B::LagrangePolynomialBasis{NFuncs})(x::T) where {NFuncs,T<:Number}
evaluate the basis `B` at the point `x`
"""
function (B::LagrangePolynomialBasis)(x::Number)
    return SP.evaluate(B.funcs, @SVector [x])
 end

 """
     (B::TensorProductBasis{2,NFuncs,T})(x::Number,y::Number) where {T,NFuncs}
 evaluate a 2-D tensor product basis at the point `(x,y)`
 """
 function (B::TensorProductBasis{2,NFuncs,T})(x::Number,y::Number) where {T,NFuncs}
     return kron(B.basis(x), B.basis(y))
 end

 """
     (B::TensorProductBasis{3,NFuncs,T})(x::Number,y::Number,z::Number) where {T,NFuncs}
 evaluate a 3-D tensor product basis at the point `(x,y,z)`
 """
 function (B::TensorProductBasis{3,NFuncs,T})(x::Number,y::Number,z::Number) where {T,NFuncs}
     return kron(B.basis(x), B.basis(y), B.basis(z))
 end

 """
    (B::TensorProductBasis{N,NFuncs,T})(x::AbstractVector) where {N,NFuncs,T}
evaluate a tensor product basis on a vector of points
 """
 function (B::TensorProductBasis{N,NFuncs,T})(x::AbstractVector) where {N,NFuncs,T}
     @assert length(x) == N
     if N == 2
         return B(x[1],x[2])
     elseif N == 3
         return B(x[1],x[2],x[3])
     else
         msg = "Evaluation of tensor product basis currently supported for 2D and 3D points only"
         throw(ArgumentError(msg))
     end
 end
