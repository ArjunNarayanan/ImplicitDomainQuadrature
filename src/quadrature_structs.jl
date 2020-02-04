"""
    AbstractQuadratureRule{D,T,S}
a quadrature rule is defined via an iterator that returns a
sequence of `D` dimensional quadrature point vectors `p::AbstractVector{T}`,
and quadrature weights `w::S`
"""
abstract type AbstractQuadratureRule{D,T,S} end

"""
    ReferenceQuadratureRule{N,T,S} <: AbstractQuadratureRule{T,S}
a one-dimensional quadrature rule defined on the reference domain `[-1,1]`.
# Inner Constructors:
    ReferenceQuadratureRule(points::SMatrix{1,N,T}, weights::SVector{N,S}) where {T<:Real,S<:Real}
# Outer Constructors:
    ReferenceQuadratureRule(points::AbstractMatrix, weights::AbstractVector)
    ReferenceQuadratureRule(points::AbstractVector, weights::AbstractVector)
    ReferenceQuadratureRule(N::Int)
return an `N` point gauss legendre quadrature rule on `[-1,1]`
"""
struct ReferenceQuadratureRule{N,T,S} <: AbstractQuadratureRule{1,T,S}
    points::SMatrix{1,N,T}
    weights::SVector{N,S}
    function ReferenceQuadratureRule(points::SMatrix{1,N,T}, weights::SVector{N,S}) where {T<:Real,S<:Real} where {N}
        return new{N,T,S}(points,weights)
    end
end

function checkNumPointsWeights(NP::Int,NW::Int)
    if NP != NW
        throw(DimensionMismatch("Number of points and weights should match, got `points,weights = ` $npoints, $nweights"))
    end
end

function ReferenceQuadratureRule(points::AbstractMatrix, weights::AbstractVector)
    dim,npoints = size(points)
    nweights = length(weights)
    if dim > 1
        throw(ArgumentError("Require `(D = 1,N)` matrix of points, got D = $dim"))
    end
    checkNumPointsWeights(npoints,nweights)
    return ReferenceQuadratureRule(SMatrix{1,npoints}(points), SVector{npoints}(weights))
end

function ReferenceQuadratureRule(points::AbstractVector, weights::AbstractVector)
    npoints = length(points)
    nweights = length(weights)
    checkNumPointsWeights(npoints,nweights)
    return ReferenceQuadratureRule(SMatrix{1,npoints}(points), SVector{npoints}(weights))
end

function ReferenceQuadratureRule(N::Int)
    points, weights = gausslegendre(N)
    ReferenceQuadratureRule(points,weights)
end


"""
    QuadratureRule{D,T,S} <: AbstractQuadratureRule{T,S}
a quadrature rule in `D` spatial dimensions. This is a mutable type;
points and weights can be added via the `update!` method.
# Inner Constructor:
    QuadratureRule(points::Matrix{T}, weights::Vector{S}) where {T,S}
# Outer Constructors:
    QuadratureRule(T::Type{<:Real}, S::Type{<:Real}, dim::Int)
initializes `(dim,0)` matrix of points and `(0)` vector of weights
    QuadratureRule(dim::Int)
initializes `(dim,0)` matrix of points and `(0)` vector of weights of type `Float64`
"""
mutable struct QuadratureRule{D,T,S} <: AbstractQuadratureRule{D,T,S}
    points::Matrix{T}
    weights::Vector{S}
    N::Int
    function QuadratureRule(points::Matrix{T}, weights::Vector{S}) where {T,S}
        dim, npoints = size(points)
        if dim > 3
            msg = "Dimension of points must be 1 <= dim <= 3, got dim = $dim"
            throw(ArgumentError(msg))
        end
        nweights = length(weights)
        checkNumPointsWeights(npoints,nweights)
        new{dim,T,S}(points,weights,npoints)
    end
end

function QuadratureRule(T::Type{<:Real}, S::Type{<:Real}, dim::Int)
    points = Matrix{T}(undef, dim, 0)
    weights = Vector{S}(undef, 0)
    return QuadratureRule(points, weights)
end

function QuadratureRule(dim::Int)
    QuadratureRule(Float64, Float64, dim)
end

"""
    Base.iterate(quad::ReferenceQuadratureRule{N}, state=1)
return each point, weight pair as a tuple `(p,w)`
"""
function Base.iterate(quad::ReferenceQuadratureRule{N}, state=1) where {N}
    if state > N
        return nothing
    else
        return ((quad.points[state], quad.weights[state]), state+1)
    end
end

"""
    Base.iterate(quad::T, state=1) where {T<:AbstractQuadratureRule{D}} where {D}
return each point, weight pair as a tuple `(p,w)` where `p` is an appropriate view
into the point matrix `quad.points`
"""
function Base.iterate(quad::T, state=1) where {T<:AbstractQuadratureRule{D}} where {D}
    if state > quad.N
        return nothing
    else
        return ((view(quad.points,1:D,state), quad.weights[state]), state+1)
    end
end

"""
    Base.getindex(quad::ReferenceQuadratureRule{N}, i::Int) where {N}
    Base.getindex(quad::T, i::Int) where {T<:AbstractQuadratureRule{D}} where {D}
return the `i`th point weight pair `p,w`.
"""
function Base.getindex(quad::ReferenceQuadratureRule{N}, i::Int) where {N}
    1 <= i <= N || BoundsError(quad, i)
    return (quad.points[i], quad.weights[i])
end

function Base.getindex(quad::T, i::Int) where {T<:AbstractQuadratureRule{D}} where {D}
    1 <= i <= quad.N || BoundsError(quad, i)
    return (view(quad.points,1:D,i), quad.weights[i])
end

"""
    update!(quad::QuadratureRule{D,T,S}, points::Matrix{T}, weights::Vector{S}) where {D,T,S}
concatenates `points` and `weights` into `quad.points` and `quad.weights`
"""
function update!(quad::QuadratureRule{D,T,S}, points::Matrix{T}, weights::Vector{S}) where {D,T,S}
    dim, npoints = size(points)
    nweights = length(weights)
    checkNumPointsWeights(npoints, nweights)
    if dim != D
        msg = "Require $(quad.dim) dimensional points for update, got $dim"
        throw(DimensionMismatch(msg))
    end
    quad.points = hcat(quad.points, points)
    append!(quad.weights, weights)
    quad.N = size(quad.points)[2]
end

"""
    transform(quad::ReferenceQuadratureRule, a::T, b::T) where {T<:Real}
transform `quad` from the interval `[-1,1]` to the interval `[a,b]`
    transform(quad::ReferenceQuadratureRule, int::Interval)
transform `quad` from the interval `[-1,1]` to `int`
"""
function transform(quad::ReferenceQuadratureRule, a::T, b::T) where {T<:Real}
    if b <= a
        throw(ArgumentError("Require a < b, got ($a,$b)"))
    end
    L = 0.5*(b-a)
    mid = 0.5*(b+a)
    transformed_points = L*quad.points .+ mid
    transformed_weights = L*quad.weights
    return transformed_points, transformed_weights
end

function transform(quad::ReferenceQuadratureRule, int::Interval)
    return transform(quad, int.lo, int.hi)
end
