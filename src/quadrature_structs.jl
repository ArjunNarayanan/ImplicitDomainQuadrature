abstract type AbstractQuadratureRule{dim,NQ} end

struct ReferenceQuadratureRule{NQ,T} <: AbstractQuadratureRule{1,NQ}
    points::SMatrix{1,NQ,T}
    weights::SVector{NQ,T}
    lo::T
    hi::T
    function ReferenceQuadratureRule(points::SMatrix{1,NQ,T},
        weights::SVector{NQ,T},lo::T,hi::T) where {T<:Real,NQ}

        @assert issorted(points)
        @assert lo < hi
        interval_length = hi-lo
        @assert sum(weights) ≈ interval_length

        return new{NQ,T}(points,weights,lo,hi)

    end
end

function ReferenceQuadratureRule(points::M,weights::V,
        lo,hi) where {M<:AbstractMatrix} where {V<:AbstractVector}

    dim,npoints = size(points)
    nweights = length(weights)
    @assert dim == 1
    @assert npoints == nweights
    sp = SMatrix{1,npoints}(points)
    sw = SVector{npoints}(weights)
    return ReferenceQuadratureRule(sp,sw,lo,hi)
end

function ReferenceQuadratureRule(points::V1,weights::V2,
        lo,hi) where {V1<:AbstractVector,V2<:AbstractVector}

    return ReferenceQuadratureRule(points',weights,lo,hi)
end

function ReferenceQuadratureRule(N::Z) where {Z<:Integer}
    @assert N > 0
    points,weights = gausslegendre(N)
    return ReferenceQuadratureRule(points,weights,-1.0,1.0)
end

function Base.iterate(quad::Q,
        state=1) where {Q<:AbstractQuadratureRule{dim,NQ}} where {dim,NQ}

    if state > NQ
        return nothing
    else
        return ((view(quad.points,1:dim,state),quad.weights[state]),state+1)
    end
end

function Base.getindex(quad::Q,
        i::Z) where
        {Q<:AbstractQuadratureRule{dim,NQ}} where {dim,NQ,Z<:Integer}

    1 <= i <= NQ || throw(BoundsError(quad,i))
    return (view(quad.points,1:dim,i),quad.weights[i])
end

function affine_map(xi,xiL,xiR,xL,xR)
    return xL + (xR-xL)/(xiR-xiL)*(xi-xiL)
end

function affine_map_derivative(dxi,xiL,xiR,xL,xR)
    return (xR-xL)/(xiR-xiL)*dxi
end

function transform(points,weights,xiL,xiR,xL,xR)
    p = affine_map.(points,xiL,xiR,xL,xR)
    w = affine_map_derivative.(weights,xiL,xiR,xL,xR)
    return p,w
end

function transform(quad::ReferenceQuadratureRule,xL,xR)
    return transform(quad.points,quad.weights,quad.lo,quad.hi,xL,xR)
end

function transform(quad::ReferenceQuadratureRule,int::Interval)
    return transform(quad,int.lo,int.hi)
end

struct QuadratureRule{dim,NQ,T} <: AbstractQuadratureRule{dim,NQ}
    points::SMatrix{dim,NQ,T}
    weights::SVector{NQ,T}
    function QuadratureRule(points::SMatrix{dim,NQ,T},
            weights::SVector{NQ,T}) where {dim,NQ,T}

        @assert 1 <= dim <= 3
        new{dim,NQ,T}(points,weights)
    end
end

function QuadratureRule(points::M,
        weights::V) where {M<:AbstractMatrix,V<:AbstractVector}

    dim,np = size(points)
    nw = length(weights)
    @assert np == nw
    @assert 1 <= dim <= 3
    sp = SMatrix{dim,np}(points)
    sw = SVector{np}(weights)
    return QuadratureRule(sp,sw)
end

function tensor_product_points(p1::M,p2::M) where {M<:AbstractMatrix}
    n1 = size(p1)[2]
    n2 = size(p2)[2]
    return vcat(repeat(p1,inner=(1,n2)),repeat(p2,outer=(1,n1)))
end

function tensor_product(quad::ReferenceQuadratureRule,box::IntervalBox{1})
    return QuadratureRule(quad.points,quad.weights)
end

function tensor_product(quad::ReferenceQuadratureRule,box::IntervalBox{2})
    p1,w1 = transform(quad,box[1])
    p2,w2 = transform(quad,box[2])
    points = tensor_product_points(p1,p2)
    weights = kron(w1,w2)
    return QuadratureRule(points,weights)
end

function tensor_product_quadrature(D::Z,NQ::Z) where {Z<:Integer}
    quad1d = ReferenceQuadratureRule(NQ)
    box = IntervalBox(-1.0..1.0,D)
    return tensor_product(quad1d,box)
end

mutable struct TemporaryQuadrature{T}
    points::Matrix{T}
    weights::Vector{T}
    function TemporaryQuadrature(
        p::M,
        w::V,
    ) where {M<:AbstractMatrix{T},V<:AbstractVector} where {T}
        d, np = size(p)
        nw = length(w)
        @assert np == nw
        new{T}(p, w)
    end
end

function TemporaryQuadrature(T, D::Z) where {Z<:Integer}
    @assert 1 <= D <= 3
    p = Matrix{T}(undef, D, 0)
    w = Vector{T}(undef, 0)
    return TemporaryQuadrature(p, w)
end

function QuadratureRule(quad::TemporaryQuadrature)
    return QuadratureRule(quad.points, quad.weights)
end

function temporary_tensor_product(
    quad::ReferenceQuadratureRule,
    box::IntervalBox{2},
)

    p1, w1 = transform(quad, box[1])
    p2, w2 = transform(quad, box[2])
    points = tensor_product_points(p1, p2)
    weights = kron(w1, w2)
    return TemporaryQuadrature(points, weights)
end

function update!(quad::TemporaryQuadrature, p::V, w) where {V<:AbstractVector}
    d = length(p)
    nw = length(w)
    dq, nqp = size(quad.points)
    @assert dq == d
    @assert nw == 1
    quad.points = hcat(quad.points, p)
    append!(quad.weights, w)
end

function update!(quad::TemporaryQuadrature, p::M, w) where {M<:AbstractMatrix}
    d, np = size(p)
    nw = length(w)
    dq, nqp = size(quad.points)
    @assert dq == d
    @assert nw == np
    quad.points = hcat(quad.points, p)
    append!(quad.weights, w)
end
