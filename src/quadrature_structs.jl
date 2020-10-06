abstract type AbstractQuadratureRule{dim} end

struct ReferenceQuadratureRule{T} <: AbstractQuadratureRule{1}
    points::Matrix{T}
    weights::Vector{T}
    lo::T
    hi::T
    numqp::Int
    function ReferenceQuadratureRule(
        points::M,
        weights::V,
        lo,
        hi,
    ) where {M<:AbstractMatrix{T},V<:AbstractVector{T}} where {T}

        dim, numqp = size(points)
        @assert dim == 1
        @assert length(weights) == numqp
        @assert issorted(points)
        @assert lo < hi

        interval_length = hi - lo
        @assert sum(weights) ≈ interval_length

        return new{T}(points, weights, lo, hi, numqp)
    end
end

function ReferenceQuadratureRule(
    points::V,
    weights::V,
    lo,
    hi,
) where {V<:AbstractVector{T}} where {T}

    return ReferenceQuadratureRule(points', weights, lo, hi)
end

function ReferenceQuadratureRule(N)
    @assert N > 0
    points, weights = gausslegendre(N)
    return ReferenceQuadratureRule(points, weights, -1.0, 1.0)
end

function Base.iterate(
    quad::Q,
    state = 1,
) where {Q<:AbstractQuadratureRule{dim}} where {dim}

    if state > quad.numqp
        return nothing
    else
        return (
            (view(quad.points, 1:dim, state), quad.weights[state]),
            state + 1,
        )
    end
end

function Base.getindex(
    quad::Q,
    i::Z,
) where {Q<:AbstractQuadratureRule{dim}} where {dim,Z<:Integer}

    1 <= i <= quad.numqp || throw(BoundsError(quad, i))
    return (view(quad.points, 1:dim, i), quad.weights[i])
end

function Base.show(
    io::IO,
    quad::Q,
) where {Q<:AbstractQuadratureRule{dim}} where {dim}

    numqp = quad.numqp
    T = typeof(quad)
    print(io, "$T\n\tDimension: $dim\n\tLength   : $numqp")
end

function Base.length(quad::Q) where {Q<:AbstractQuadratureRule}
    return quad.numqp
end

function dimension(quad::Q) where {Q<:AbstractQuadratureRule{dim}} where {dim}
    return dim
end

function affine_map(xi, xiL, xiR, xL, xR)
    return xL + (xR - xL) / (xiR - xiL) * (xi - xiL)
end

function affine_map_derivative(dxi, xiL, xiR, xL, xR)
    return (xR - xL) / (xiR - xiL) * dxi
end

function transform(points, weights, xiL, xiR, xL, xR)
    p = affine_map.(points, xiL, xiR, xL, xR)
    w = affine_map_derivative.(weights, xiL, xiR, xL, xR)
    return p, w
end

function transform(quad::ReferenceQuadratureRule, xL, xR)
    return transform(quad.points, quad.weights, quad.lo, quad.hi, xL, xR)
end

function transform(quad::ReferenceQuadratureRule, int::Interval)
    return transform(quad, int.lo, int.hi)
end

struct QuadratureRule{dim,T} <: AbstractQuadratureRule{dim}
    points::Matrix{T}
    weights::Vector{T}
    numqp::Int
    function QuadratureRule(
        points::M,
        weights::V,
    ) where {M<:AbstractMatrix{T},V<:AbstractVector{T}} where {T}

        dim, numqp = size(points)
        @assert 1 <= dim <= 3
        @assert length(weights) == numqp

        new{dim,T}(points, weights, numqp)
    end
end

function tensor_product_points(p1, p2)
    n1 = size(p1)[2]
    n2 = size(p2)[2]
    return vcat(repeat(p1, inner = (1, n2)), repeat(p2, outer = (1, n1)))
end

function tensor_product(quad, box::IntervalBox{1})
    return QuadratureRule(quad.points, quad.weights)
end

function tensor_product(quad, box::IntervalBox{2})
    p1, w1 = transform(quad, box[1])
    p2, w2 = transform(quad, box[2])
    points = tensor_product_points(p1, p2)
    weights = kron(w1, w2)
    return QuadratureRule(points, weights)
end

function tensor_product_quadrature(D, NQ)
    quad1d = ReferenceQuadratureRule(NQ)
    box = IntervalBox(-1.0..1.0, D)
    return tensor_product(quad1d, box)
end

mutable struct TemporaryQuadrature{T}
    points::Matrix{T}
    weights::Vector{T}
    function TemporaryQuadrature(
        p::M,
        w::V,
    ) where {M<:AbstractMatrix{T},V<:AbstractVector{T}} where {T}
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

function temporary_tensor_product(quad, box::IntervalBox{2})

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
