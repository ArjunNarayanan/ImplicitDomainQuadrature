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
    LagrangePolynomialBasis(order::Int; start = -1.0, stop = 1.0)
construct a polynomial basis with variable `x` of order `order` with
equally spaced points between `start` and `stop`.
"""
function LagrangePolynomialBasis(order::Int; start = -1.0, stop = 1.0)

    DP.@polyvar x
    NFuncs = order + 1
    roots = range(start, stop = stop, length = NFuncs)
    basis = lagrange_polynomials(x,roots)
    return LagrangePolynomialBasis(basis,roots)
end

"""
    TensorProductBasis{N,NFuncs,T<:AbstractBasis1D} <: AbstractBasis{NFuncs}
construct an `N` dimensional polynomial basis by tensor product of `T`
with a total of `NFuncs` functions.
Note that `NFuncs` can be inferred from `T{nfuncs}` as
`NFuncs = nfuncs^N`.
# Fields
    - `basis::AbstractBasis1D` the underlying 1D basis
"""
struct TensorProductBasis{N,NFuncs,T<:AbstractBasis1D} <: AbstractBasis{NFuncs}
    basis::AbstractBasis1D
    function TensorProductBasis(N::Int, B::T) where {T<:AbstractBasis1D{nfuncs}} where {nfuncs}
        if (N < 1) || (N > 3)
            msg = "Require 1 <= N <= 3, got N = $N"
            throw(ArgumentError(msg))
        end
        NFuncs = nfuncs^N
        new{N,NFuncs,T}(B)
    end
end

"""
    TensorProductBasis(dim::Int, order::Int; start = -1.0, stop = 1.0)
construct a `dim` dimensional polynomial basis with variable `x` using a tensor
product of `order` order `LagrangePolynomialBasis`. The polynomials are
equispaced from `start` to `stop`.
"""
function TensorProductBasis(dim::Int, order::Int; start = -1.0, stop = 1.0)
    basis_1d = LagrangePolynomialBasis(order, start = start, stop = stop)
    return TensorProductBasis(dim, basis_1d)
end

"""
    (B::LagrangePolynomialBasis{NFuncs})(x::T) where {NFuncs,T<:Number}
evaluate the basis `B` at the point `x`
"""
function (B::LagrangePolynomialBasis)(x::Number)
    return SP.evaluate(B.funcs, @SVector [x])
 end

"""
    derivative(B::LagrangePolynomialBasis, x::Number)
evaluate the derivative of basis `B` at the point `x`
"""
function derivative(B::LagrangePolynomialBasis, x::Number)
    return SP.jacobian(B.funcs, @SVector [x])
end

"""
    (B::TensorProductBasis{1})(x::Number)
evaluate a 1-D tensor product basis at the point `x`
"""
function (B::TensorProductBasis{1})(x::Number)
    return B.basis(x)
end

"""
    gradient(B::TensorProductBasis{1}, x::Number)
evaluate the derivative of the 1-D tensor product basis at `x`
"""
function gradient(B::TensorProductBasis{1}, x::Number)
    return derivative(B.basis, x)
end

"""
 (B::TensorProductBasis{2})(x::Number,y::Number)
evaluate a 2-D tensor product basis at the point `(x,y)`
"""
function (B::TensorProductBasis{2})(x::Number,y::Number)
    return kron(B.basis(x), B.basis(y))
end

"""
    gradient(B::TensorProductBasis{2}, dir::Int, x::Number, y::Number)
return the gradient of `B` at the point `x` along direction `dir`
"""
function gradient(B::TensorProductBasis{2}, dir::Int, x::Number, y::Number)
    if dir == 1
        dNx = derivative(B.basis, x)
        Ny = B.basis(y)
        return kron(dNx,Ny)
    elseif dir == 2
        Nx = B.basis(x)
        dNy = derivative(B.basis, y)
        return kron(Nx, dNy)
    else
        throw(BoundsError(B,dir))
    end
end

"""
 (B::TensorProductBasis{3})(x::Number,y::Number,z::Number)
evaluate a 3-D tensor product basis at the point `(x,y,z)`
"""
function (B::TensorProductBasis{3})(x::Number,y::Number,z::Number)
    return kron(B.basis(x), B.basis(y), B.basis(z))
end

"""
    gradient(B::TensorProductBasis{3}, dir::Int, x::Number, y::Number, z::Number)
evaluate the gradient of `B` at `(x,y,z)` along direction `dir`.
"""
function gradient(B::TensorProductBasis{3}, dir::Int, x::Number, y::Number, z::Number)
    if dir == 1
        dNx = derivative(B.basis, x)
        Ny = B.basis(y)
        Nz = B.basis(z)
        return kron(dNx, Ny, Nz)
    elseif dir == 2
        Nx = B.basis(x)
        dNy = derivative(B.basis, y)
        Nz = B.basis(z)
        return kron(Nx, dNy, Nz)
    elseif dir == 3
        Nx = B.basis(x)
        Ny = B.basis(y)
        dNz = derivative(B.basis, z)
        return kron(Nx, Ny, dNz)
    else
        throw(BoundsError(B,dir))
    end
end

"""
(B::TensorProductBasis{N})(x::AbstractVector) where {N}
evaluate a tensor product basis on a vector of points
"""
function (B::TensorProductBasis{N})(x::AbstractVector) where {N}
    @assert length(x) == N
    if N == 1
        return B(x[1])
    elseif N == 2
        return B(x[1],x[2])
    elseif N == 3
        return B(x[1],x[2],x[3])
    else
        msg = "Evaluation of tensor product basis currently supported for 1 <= dimension <= 3 only"
        throw(ArgumentError(msg))
    end
end

"""
    gradient(B::TensorProductBasis{N}, dir::Int, x::AbstractVector) where {N}
evaluate the gradient of `B` along direction `dir` at the point vector `x`
"""
function gradient(B::TensorProductBasis{N}, dir::Int, x::AbstractVector) where {N}
    @assert length(x) == N
    if N == 1
        return gradient(B, x[1])
    elseif N == 2
        return gradient(B, dir, x[1], x[2])
    elseif N == 3
        return gradient(B, dir, x[1], x[2], x[3])
    else
        msg = "Evaluation of tensor product basis currently supported for 2D and 3D points only"
        throw(ArgumentError(msg))
    end
end

"""
    interpolation_points(B::TensorProductBasis{N,NFuncs,T}) where {NFuncs,T}
return the interpolation points of the tensor product basis
"""
function interpolation_points(B::TensorProductBasis{1,NFuncs,T}) where {NFuncs,T}
    return B.basis.points
end

function interpolation_points(B::TensorProductBasis{2,F,T}) where {T<:AbstractBasis1D{NFuncs}} where {F,NFuncs}
    npoints = NFuncs^2
    points = zeros(2,npoints)
    count = 1
    for i = 1:NFuncs
        for j = 1:NFuncs
            points[count] = B.basis.points[i]
            count += 1
            points[count] = B.basis.points[j]
            count += 1
        end
    end
    return SMatrix{2,npoints}(points)
end

function interpolation_points(B::TensorProductBasis{3,F,T}) where {T<:AbstractBasis1D{NFuncs}} where {F,NFuncs}
    npoints = NFuncs^3
    points = zeros(3,npoints)
    count = 1
    for i = 1:NFuncs
        for j = 1:NFuncs
            for k = 1:NFuncs
                points[count] = B.basis.points[i]
                count += 1
                points[count] = B.basis.points[j]
                count += 1
                points[count] = B.basis.points[k]
                count += 1
            end
        end
    end
    return SMatrix{3,npoints}(points)
end
