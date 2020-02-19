import Base: ==

"""
    AbstractBasis{N}
abstract supertype for a function basis with `N` functions.
"""
abstract type AbstractBasis{N} end

"""
    AbstractBasis1D{N}
abstract supertype for a univariate function basis with `N` functions.
"""
abstract type AbstractBasis1D{N} <: AbstractBasis{N} end

"""
    LagrangePolynomialBasis{N} <: AbstractBasis1D{N}
A basis of `N` Lagrange polynomials where each polynomial function is of order `N - 1`.
"""
struct LagrangePolynomialBasis{NFuncs} <: AbstractBasis1D{NFuncs}
    funcs::SP.PolynomialSystem{NFuncs,1}
    points::SMatrix{1,NFuncs}
    function LagrangePolynomialBasis(funcs::AbstractVector{DP.Polynomial{C,T1}},
        points::AbstractVector{T2}) where {C,T1<:Real,T2<:Real}

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
        if !isapprox(sum(funcs),one(T1))
            val = sum(funcs)
            msg = "Polynomial basis must sum to 1.0, got $val"
            throw(ArgumentError(msg))
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
    LagrangePolynomialBasis(order::Int; start::T = -1.0, stop::T = 1.0) where {T<:Real}
construct a polynomial basis with variable `x` of order `order` with
equally spaced points between `start` and `stop`.
    LagrangePolynomialBasis(order::Int)
construct a polynomial basis with variable `x` of order `order` with equally
spaced points of type `T` between `-1.0` and `1.0`
"""
function LagrangePolynomialBasis(order::Int; start::T = -1.0, stop::T = 1.0) where {T<:Real}
    DP.@polyvar x
    NF = order + 1
    roots = range(start, stop = stop, length = NF)
    basis = lagrange_polynomials(x,roots)
    return LagrangePolynomialBasis(basis,roots)
end

"""
    Base.isequal(b1::LagrangePolynomialBasis{NF},b2::LagrangePolynomialBasis{NF}) where {NF}
returns true if `b1` and `b2` have the same `funcs` and `points`.
"""
function Base.isequal(b1::LagrangePolynomialBasis{NF},b2::LagrangePolynomialBasis{NF}) where {NF}
    flag = true
    flag = flag && b1.funcs.polys == b2.funcs.polys
    flag = flag && b1.points == b2.points
    return flag
end

"""
    ==(b1::LagrangePolynomialBasis{NF},b2::LagrangePolynomialBasis{NF}) where {NF}
returns true if `b1` and `b2` have the same `funcs` and `points`.
"""
function ==(b1::LagrangePolynomialBasis{NF},b2::LagrangePolynomialBasis{NF}) where {NF}
    return isequal(b1,b2)
end

abstract type AbstractTensorProductBasis{D,T,NF} <: AbstractBasis{NF} end

"""
    TensorProductBasis{N,NFuncs,T<:AbstractBasis1D} <: AbstractBasis{NFuncs}
construct an `N` dimensional polynomial basis by tensor product of `T`
with a total of `NFuncs` functions.
Note that `NFuncs` can be inferred from `T{nfuncs}` as
`NFuncs = nfuncs^N`.
# Fields
    - `basis::T` the underlying 1D basis
    - `points::SMatrix{N,NFuncs}` a matrix of point vectors
"""
struct TensorProductBasis{D,T,NF} <: AbstractTensorProductBasis{D,T,NF}
    basis::T
    points::SMatrix{D,NF}
    function TensorProductBasis(D::Int, B::T) where {T<:AbstractBasis1D{N1D}} where {N1D}
        if (D < 1) || (D > 3)
            msg = "Require 1 <= D <= 3, got D = $D"
            throw(ArgumentError(msg))
        end
        NF = N1D^D
        points = interpolation_points(D,B)
        new{D,T,NF}(B,points)
    end
end

"""
    TensorProductBasis(dim::Int, order::Int; start = -1.0, stop = 1.0)
construct a `dim` dimensional polynomial basis with variable `x` using a tensor
product of `order` order `LagrangePolynomialBasis`. The polynomials are
equispaced from `start` to `stop`.
"""
function TensorProductBasis(dim::Int, order::Int; start::T = -1.0, stop::T = 1.0) where {T<:Real}
    basis_1d = LagrangePolynomialBasis(order, start = start, stop = stop)
    return TensorProductBasis(dim, basis_1d)
end

"""
    coefficient_number_type(B::LagrangePolynomialBasis)
    coefficient_number_type(B::TensorProductBasis)
return the type of the coefficient, i.e. `Float64, Float32`
"""
function coefficient_number_type(B::LagrangePolynomialBasis)
    return eltype(B.funcs.polys[1].coefficients)
end

function coefficient_number_type(B::TensorProductBasis)
    return eltype(B.basis.funcs.polys[1].coefficients)
end

"""
    Base.isequal(tp1::TensorProductBasis{D,T,NF}, tp2::TensorProductBasis{D,T,NF}) where {D,T,NF}
returns `true` if `tp1` and `tp2` have identical `basis` and `points`, `false` otherwise.
"""
function Base.isequal(tp1::TensorProductBasis{D,T,NF}, tp2::TensorProductBasis{D,T,NF}) where {D,T,NF}
    return tp1.basis == tp2.basis && tp1.points == tp2.points
end

"""
    ==(tp1::TensorProductBasis{D,T,NF}, tp2::TensorProductBasis{D,T,NF}) where {D,T,NF}
returns `true` if `tp1` and `tp2` have identical `basis` and `points`, `false` otherwise.
"""
function ==(tp1::TensorProductBasis{D,T,NF}, tp2::TensorProductBasis{D,T,NF}) where {D,T,NF}
    return isequal(tp1,tp2)
end

# """
#     (B::LagrangePolynomialBasis{NFuncs})(x::T) where {NFuncs,T<:Number}
# evaluate the basis `B` at the point `x`
# """
function (B::LagrangePolynomialBasis)(x::T) where {T<:Number}
    return SP.evaluate(B.funcs, @SVector [x])
 end

"""
    derivative(B::LagrangePolynomialBasis, x::Number)
evaluate the derivative of basis `B` at the point `x`
"""
function derivative(B::LagrangePolynomialBasis, x::T) where {T<:Number}
    return SP.jacobian(B.funcs, @SVector [x])
end

"""
    gradient(B::LagrangePolynomialBasis, x::Number)
alias for `derivative(B, x)`
"""
function gradient(B::LagrangePolynomialBasis, x::T) where {T<:Number}
    return derivative(B, x)
end

"""
    value_and_derivative(B::LagrangePolynomialBasis, x::Number)
return `(v,d)` where `v` is a vector of the polynomial basis values and `d` is
a vector of the derivative values of the polynomial basis at the point `x`.
"""
function value_and_derivative(B::LagrangePolynomialBasis, x::T) where {T<:Number}
    return SP.evaluate_and_jacobian(B.funcs, @SVector [x])
end

function (B::TensorProductBasis{1})(x::T) where {T<:Number}
    return B.basis(x)
end

function gradient(B::TensorProductBasis{1}, x::T) where {T<:Number}
    return derivative(B.basis, x)
end

function (B::TensorProductBasis{2})(x::T,y::T) where {T<:Number}
    return kron(B.basis(x), B.basis(y))
end

function gradient(B::TensorProductBasis{2}, dir::Int, x::T, y::T) where {T<:Number}
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

function gradient(B::TensorProductBasis{2}, x::T, y::T) where {T<:Number}
    col1 = gradient(B,1,x,y)
    col2 = gradient(B,2,x,y)
    return hcat(col1,col2)
end

# """
#     (B::TensorProductBasis{3})(x::Number,y::Number,z::Number)
# evaluate a 3-D tensor product basis at the point `(x,y,z)`
#     (B::TensorProductBasis{N})(x::AbstractVector) where {N}
# evaluate a tensor product basis on a vector of points
# """
function (B::TensorProductBasis{3})(x::T,y::T,z::T) where {T<:Number}
    return kron(B.basis(x), B.basis(y), B.basis(z))
end

function gradient(B::TensorProductBasis{3}, dir::Int, x::T, y::T, z::T) where {T<:Number}
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

function gradient(B::TensorProductBasis{3}, x::T, y::T, z::T) where {T<:Number}
    Nx = B.basis(x)
    Ny = B.basis(y)
    Nz = B.basis(z)

    dNx = derivative(B.basis, x)
    dNy = derivative(B.basis, y)
    dNz = derivative(B.basis, z)

    return hcat(kron(dNx,Ny,Nz), kron(Nx,dNy,Nz), kron(Nx,Ny,dNz))
end

function (B::TensorProductBasis{N})(x::AbstractVector) where {N}
    @assert length(x) == N
    if N == 1
        return B(x[1])
    elseif N == 2
        return B(x[1],x[2])
    elseif N == 3
        return B(x[1],x[2],x[3])
    # else
    #     msg = "Evaluation of tensor product basis currently supported for 1 <= dimension <= 3 only"
    #     throw(ArgumentError(msg))
    end
end

"""
    gradient(B::TensorProductBasis{N}, dir::Int, x::AbstractVector) where {N}
evaluate the gradient of `B` along direction `dir` at the point vector `x`
    gradient(B::TensorProductBasis{dim}, x::AbstractVector) where {dim}
return an `(N,dim)` matrix such that the `I`th row is the gradient of basis `I`.
Here `N` is the total number of basis functions.
    gradient(B::TensorProductBasis{3}, dir::Int, x::Number, y::Number, z::Number)
evaluate the gradient of `B` at `(x,y,z)` along direction `dir`.
    gradient(B::TensorProductBasis{3}, x::Number, y::Number, z::Number)
returns an `(N,3)` matrix, where row `I` is the (2D) gradient vector of the `I`th basis
function. Here `N` is the total number of basis functions.
    gradient(B::TensorProductBasis{2}, x::Number, y::Number)
returns an `(N,2)` matrix, where row `I` is the (2D) gradient vector of the `I`th basis
function. Here `N` is the total number of basis functions.
"""
function gradient(B::TensorProductBasis{N}, dir::Int, x::AbstractVector) where {N}
    @assert length(x) == N
    if N == 1
        return gradient(B, x[1])
    elseif N == 2
        return gradient(B, dir, x[1], x[2])
    elseif N == 3
        return gradient(B, dir, x[1], x[2], x[3])
    # else
    #     msg = "Evaluation of tensor product basis currently supported for 1 <= dim <= 3, got $N"
    #     throw(ArgumentError(msg))
    end
end

function gradient(B::TensorProductBasis{dim}, x::AbstractVector) where {dim}
    @assert length(x) == dim
    if dim == 1
        return gradient(B, x[1])
    elseif dim == 2
        return gradient(B, x[1], x[2])
    elseif dim == 3
        return gradient(B, x[1], x[2], x[3])
    # else
    #     msg = "Evaluation of tensor product basis gradient currently supported for 1 <= dim <= 3, got $dim"
    #     throw(ArgumentError(msg))
    end
end

"""
    interpolation_points(B::TensorProductBasis{N,NFuncs,T}) where {NFuncs,T}
return the interpolation points of the tensor product basis
"""
function interpolation_points(N::Int, B::AbstractBasis1D{NF}) where {NF}
    if N == 1
        return B.points
    elseif N == 2
        npoints = NF^2
        points = zeros(2,npoints)
        count = 1
        for i = 1:NF
            for j = 1:NF
                points[count] = B.points[i]
                count += 1
                points[count] = B.points[j]
                count += 1
            end
        end
        return SMatrix{2,npoints}(points)
    elseif N == 3
        npoints = NF^3
        points = zeros(3,npoints)
        count = 1
        for i = 1:NF
            for j = 1:NF
                for k = 1:NF
                    points[count] = B.points[i]
                    count += 1
                    points[count] = B.points[j]
                    count += 1
                    points[count] = B.points[k]
                    count += 1
                end
            end
        end
        return SMatrix{3,npoints}(points)
    end
end
