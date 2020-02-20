"""
    AbstractQuadratureRule{D,T}
Abstract supertype for all quadrature rules in `D` dimensions with type `T`
points and weights.
"""
abstract type AbstractQuadratureRule{D,T} end

"""
    ReferenceQuadratureRule{N,T}
a one-dimensional quadrature rule defined on the reference domain `[-1,1]`.
# Inner Constructors:
    ReferenceQuadratureRule(points::SMatrix{1,N,T}, weights::SVector{N,T}) where {T<:Real}
# Outer Constructors:
    ReferenceQuadratureRule(points::AbstractMatrix, weights::AbstractVector)
    ReferenceQuadratureRule(points::AbstractVector, weights::AbstractVector)
    ReferenceQuadratureRule(N::Int)
return an `N` point gauss legendre quadrature rule on `[-1,1]`
"""
struct ReferenceQuadratureRule{N,T} <: AbstractQuadratureRule{1,T}
    points::SMatrix{1,N,T}
    weights::SVector{N,T}
    function ReferenceQuadratureRule(points::SMatrix{1,N,T}, weights::SVector{N,T}) where {T<:Real,N}
        w = sum(weights)
        if !(w ≈ 2)
            msg = "Quadrature rule over reference domain [-1,1] must have weights that sum to 2.0, got sum = $w"
            throw(ArgumentError(msg))
        end
        for i in 1:N
            p = points[i]
            -1.0 <= p <= 1.0 || throw(DomainError("Quadrature points must be contained within reference domain [-1,1], got $p"))
        end
        return new{N,T}(points,weights)
    end
end

function checkNumPointsWeights(NP::Int,NW::Int)
    if NP != NW
        throw(DimensionMismatch("Number of points and weights should match, got num. points, num. weights = ` $NP,$NW"))
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
    Base.getindex(quad::ReferenceQuadratureRule{N}, i::Int) where {N}
return the `i`th point weight pair `p,w`.
"""
function Base.getindex(quad::ReferenceQuadratureRule{N}, i::Int) where {N}
    1 <= i <= N || throw(BoundsError(quad, i))
    return (quad.points[i], quad.weights[i])
end

"""
    quadrature_transformers(lo, hi)
return `0.5(hi - lo), 0.5*(hi + lo)` which are the factors used to
transform a quadrature rule from `(-1,1)` to `(lo,hi)`
"""
function quadrature_transformers(lo, hi)
    if hi < lo
        throw(ArgumentError("Require lo <= hi, got ($lo,$hi)"))
    end
    if lo == hi
        @warn "Transforming quadrature rule to a null domain"
    end
    scale = 0.5(hi - lo)
    mid = 0.5*(hi + lo)
    return scale, mid
end

"""
    transform(quad::ReferenceQuadratureRule{N,T}, lo::T, hi::T) where {N,T}
transform `quad` from the interval `[-1,1]` to the interval `[a,b]`
    transform(quad::ReferenceQuadratureRule, int::Interval)
transform `quad` from the interval `[-1,1]` to `int`
"""
function transform(quad::ReferenceQuadratureRule{N,T}, lo::T, hi::T) where {N,T}
    scale, mid = quadrature_transformers(lo, hi)
    transformed_points = scale*quad.points .+ mid
    transformed_weights = scale*quad.weights
    return transformed_points, transformed_weights
end

function transform(quad::ReferenceQuadratureRule{N,T}, int::Interval{T}) where {N,T}
    return transform(quad, int.lo, int.hi)
end

"""
    QuadratureRule{D,T}
a quadrature rule in `D` spatial dimensions. This is a mutable type;
points and weights can be added via the `update!` method.
# Inner Constructor:
    QuadratureRule(points::Matrix{T}, weights::Vector{T}) where {T<:Real}
# Outer Constructors:
    QuadratureRule(T::Type{<:Real}, dim::Int)
initializes `(dim,0)` matrix of points and `(0)` vector of weights. Both are of type `T`
    QuadratureRule(dim::Int)
initializes `(dim,0)` matrix of points and `(0)` vector of weights of type `Float64`
"""
mutable struct QuadratureRule{D,T} <: AbstractQuadratureRule{D,T}
    points::Matrix{T}
    weights::Vector{T}
    N::Int
    function QuadratureRule(points::AbstractMatrix{T}, weights::AbstractVector{T}) where {T<:AbstractFloat}
        dim, npoints = size(points)
        if dim > 3
            msg = "Dimension of points must be 1 <= dim <= 3, got dim = $dim"
            throw(ArgumentError(msg))
        end
        nweights = length(weights)
        checkNumPointsWeights(npoints,nweights)
        new{dim,T}(points,weights,npoints)
    end
end

function QuadratureRule(T::Type{<:AbstractFloat}, dim::Int)
    points = Matrix{T}(undef, dim, 0)
    weights = Vector{T}(undef, 0)
    return QuadratureRule(points, weights)
end

function QuadratureRule(dim::Int)
    QuadratureRule(Float64, dim)
end

"""
    Base.iterate(quad::QuadratureRule{D}, state=1) where {D}
return each point, weight pair as a tuple `(p,w)` where `p` is an appropriate view
into the point matrix `quad.points`
"""
function Base.iterate(quad::QuadratureRule{D}, state=1) where {D}
    if state > quad.N
        return nothing
    else
        return ((view(quad.points,1:D,state), quad.weights[state]), state+1)
    end
end

"""
    Base.getindex(quad::QuadratureRule{D}, i::Int) where {D}
return `(p,w)` where `p` is an appropriate view into the `i`th quadrature
point, and `w` is the `i`th quadrature weight.
"""
function Base.getindex(quad::QuadratureRule{D}, i::Int) where {D}
    1 <= i <= quad.N || throw(BoundsError(quad, i))
    return (view(quad.points,1:D,i), quad.weights[i])
end

"""
    update!(quad::QuadratureRule{D,T}, points::AbstractMatrix{T}, weights::AbstractVector{T}) where {D,T}
concatenates `points` and `weights` into `quad.points` and `quad.weights` and
increments `quad.N` appropriately.
    update!(quad::QuadratureRule{D,T}, point::AbstractVector{T}, weight::T) where {D,T}
adds a single `point,weight` pair to `quad`
"""
function update!(quad::QuadratureRule{D,T}, points::AbstractMatrix{T}, weights::AbstractVector{T}) where {D,T}
    dim, npoints = size(points)
    nweights = length(weights)
    checkNumPointsWeights(npoints, nweights)
    if dim != D
        msg = "Require $D dimensional points for update, got $dim"
        throw(DimensionMismatch(msg))
    end
    quad.points = hcat(quad.points, points)
    append!(quad.weights, weights)
    quad.N = length(quad.weights)
end

function update!(quad::QuadratureRule{D,T}, point::AbstractVector{T}, weight::T) where {D,T}
    dim = length(point)
    if dim != D
        msg = "Require $D dimensional points for update, got $dim"
        throw(DimensionMismatch(msg))
    end
    quad.points = hcat(quad.points, point)
    append!(quad.weights, weight)
    quad.N = quad.N + 1
end

"""
    tensorProductPoints(p1::Matrix, p2::Matrix)
returns a matrix of points representing the "tensor product" (or kroneker product)
of `p1` and `p2`
"""
function tensorProductPoints(p1::AbstractMatrix, p2::AbstractMatrix)
    n1 = size(p1)[2]
    n2 = size(p2)[2]
    return vcat(repeat(p1,inner=(1,n2)), repeat(p2,outer=(1,n1)))
end

"""
    tensorProduct(quad1d::ReferenceQuadratureRule, box::IntervalBox{2})
returns a quadrature rule representing the "tensor product" or "kroneker product"
of `quad1d` that is appropriately transformed into `box`
"""
function tensorProduct(quad1d::ReferenceQuadratureRule, box::IntervalBox{2})
    p1, w1 = transform(quad1d, box[1])
    p2, w2 = transform(quad1d, box[2])
    points = tensorProductPoints(p1, p2)
    weights = kron(w1,w2)
    return QuadratureRule(points,weights)
end

struct TensorProductQuadratureRule{D,R,N,T} <: AbstractQuadratureRule{D,T}
    points::SMatrix{D,N,T}
    weights::SVector{N,T}
    function TensorProductQuadratureRule(D::Int, quad1d::R) where {R<:ReferenceQuadratureRule{N1,T}} where {N1,T}
        1 <= D <= 2 || throw(ArgumentError("Expected D ∈ {1,2}, got D = $D"))
        p = tensorProductPoints(quad1d.points, quad1d.points)
        w = kron(quad1d.weights, quad1d.weights)
        N = N1^D
        points = SMatrix{D,N}(p)
        weights = SVector{N}(w)
        new{D,R,N,T}(points,weights)
    end
end

function TensorProductQuadratureRule(D::Int, order::Int)
    order > 0 || throw(ArgumentError("Expected order > 0, got order = $order"))
    quad1d = ReferenceQuadratureRule(order)
    return TensorProductQuadratureRule(D,quad1d)
end

function Base.iterate(quad::TensorProductQuadratureRule{D,R,N}, state=1) where {D,R,N}
    if state > N
        return nothing
    else
        return ((view(quad.points,1:D,state), quad.weights[state]), state+1)
    end
end

function Base.getindex(quad::TensorProductQuadratureRule{D,R,N}, i::Int) where {D,R,N}
    1 <= i <= N || throw(BoundsError(quad, i))
    return (view(quad.points,1:D,i), quad.weights[i])
end
