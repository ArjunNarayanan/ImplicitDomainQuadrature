using FastGaussQuadrature, StaticArrays


abstract type AbstractQuadratureRule end

struct QuadratureRule1D <: AbstractQuadratureRule
    points::Matrix{T} where {T<:Real}
    weights::Vector{S} where {S<:Real}
    N::Int
    function QuadratureRule1D(N::Int)
        if N < 1
            throw(ArgumentError("Require N > 0, got $N"))
        end
        points, weights = gausslegendre(N)
        new(Matrix(points'), weights,N)
    end
end

"""
`N` pairs of points and weights.
points are `dim` dimensional
"""
mutable struct QuadratureRule <: AbstractQuadratureRule
    points::Matrix{T} where {T<:Real}
    weights::Vector{S} where {S<:Real}
    N::Int
    dim::Int
    function QuadratureRule(points::AbstractMatrix, weights::AbstractVector)
        dim, npoints = size(points)
        if dim > 3
            msg = "Dimension of points must be 1 <= dim <= 3, got dim = $dim"
            throw(ArgumentError(msg))
        end
        nweights = length(weights)
        if npoints != nweights
            msg = "Require number of points = number of weights, got $npoints, $nweights"
            throw(ArgumentError(msg))
        end
        new(points,weights,npoints,dim)
    end
end

function QuadratureRule(T::Type{<:Real}, S::Type{<:Real}, dim::Int)
    points = Matrix{T}(undef, dim, 0)
    weights = Vector{S}(undef, 0)
    return QuadratureRule(points, weights)
end

function QuadratureRule(dim::Int)
    return QuadratureRule(Float64, Float64, dim)
end

function tensorProductPoints(p1::Matrix, p2::Matrix)
    n1 = size(p1)[2]
    n2 = size(p2)[2]
    return vcat(repeat(p1,inner=(1,n2)), repeat(p2,outer=(1,n1)))
end

struct TensorProductQuadratureRule <: AbstractQuadratureRule
    points::Matrix{T} where {T<:Real}
    weights::Vector{S} where {S<:Real}
    N::Int
    dim::Int
    N1D::Int
    function TensorProductQuadratureRule(dim::Int, quad::QuadratureRule1D)
        if dim < 1 || dim > 3
            msg = "Dimension of points must be 1 <= dim <= 3, got dim = $dim"
            throw(ArgumentError(msg))
        end
        N1D = quad.N
        N = (N1D)^dim
        if dim == 1
            return new(copy(quad.points), copy(quad.weights), N, dim, N1D)
        elseif dim == 2
            p2 = tensorProductPoints(quad.points, quad.points)
            w2 = kron(quad.weights, quad.weights)
            return new(p2, w2, N, dim, N1D)
        else
            p2 = tensorProductPoints(quad.points, quad.points)
            w2 = kron(quad.weights, quad.weights)
            p3 = tensorProductPoints(quad.points, p2)
            w3 = kron(quad.weights, w2)
            return new(p3, w3, N, dim, N1D)
        end
    end
end

function Base.iterate(quad::T, state=1) where {T<:AbstractQuadratureRule}
    if state > quad.N
        return nothing
    else
        return ((view(quad.points,:,state), quad.weights[state]), state+1)
    end
end

function Base.getindex(quad::T, i::Int) where {T<:AbstractQuadratureRule}
    1 <= i <= quad.N || BoundsError(quad, i)
    return (view(quad.points,:,i), quad.weights[i])
end

function update!(quad::QuadratureRule, points::Matrix, weights::Vector)
    dim, npoints = size(points)
    nweights = length(weights)
    if npoints != nweights
        msg = "Require number of points = number of weights, got $npoints, $nweights"
        throw(ArgumentError(msg))
    end
    if dim != quad.dim
        msg = "Require $(quad.dim) dimensional points for update, got $dim"
        throw(DimensionMismatch(msg))
    end
    quad.points = hcat(quad.points, points)
    append!(quad.weights, weights)
    quad.N = size(quad.points)[2]
end

quad1d = QuadratureRule1D(5)
quad = QuadratureRule(1)
update!(quad, quad1d.points, quad1d.weights)
# points2d = vcat(quad.points,ones(1,length(quad.points)))
#
# quad2d = QuadratureRule(copy(points2d), copy(quad.weights))
# new_points = [1. 2 3
#               1  2 3]
# new_weights = [0.5, 0.5, 0.5]
# update!(quad2d, new_points, new_weights)
#
# tq2 = TensorProductQuadratureRule(2,quad)
# tq3 = TensorProductQuadratureRule(3,quad)
