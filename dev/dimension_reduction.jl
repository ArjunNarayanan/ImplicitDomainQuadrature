using FastGaussQuadrature, Roots, IntervalArithmetic, IntervalRootFinding
using LinearAlgebra, BenchmarkTools
using Plots, Printf
using Revise
using ImplicitDomainQuadrature

"""
points of type `T`
weights of type `S`
"""
abstract type AbstractQuadratureRule{T,S} end

struct QuadratureRule1D{T,S} <: AbstractQuadratureRule{T,S}
    points::Matrix{T}
    weights::Vector{S}
    N::Int
    function QuadratureRule1D(N::Int)
        if N < 1
            throw(ArgumentError("Require N > 0, got $N"))
        end
        points, weights = gausslegendre(N)
        T = eltype(points)
        S = eltype(weights)
        new{T,S}(Matrix(points'), weights,N)
    end
end

"""
`N` pairs of points and weights.
points are `dim` dimensional
"""
mutable struct QuadratureRule{T,S} <: AbstractQuadratureRule{T,S}
    points::Matrix{T}
    weights::Vector{S}
    N::Int
    dim::Int
    function QuadratureRule(points::Matrix{T}, weights::Vector{S}) where {T,S}
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
        new{T,S}(points,weights,npoints,dim)
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

function update!(quad::QuadratureRule{T,S}, points::Matrix{T}, weights::Vector{S}) where {T,S}
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

function update!(quad::QuadratureRule{T,S}, point::Vector{T}, weight::S) where {T,S}
    dim = length(point)
    if dim != quad.dim
        msg = "Require $(quad.dim) dimensional points for update, got $dim"
        throw(DimensionMismatch(msg))
    end
    quad.points = hcat(quad.points, point)
    append!(quad.weights, weight)
    quad.N = size(quad.points)[2]
end

"""
    transform_quadrature(quad::QuadratureRule1D, a::Real, b::Real)
transform the given quadrature rule to the interval `[a,b]`
"""
function transform_quadrature(quad::QuadratureRule1D, a::Real, b::Real)
    if b <= a
        throw(ArgumentError("Require a < b, got ($a,$b)"))
    end
    L = 0.5*(b-a)
    mid = 0.5*(b+a)
    transformed_points = L*quad.points .+ mid
    transformed_weights = L*quad.weights
    return transformed_points, transformed_weights
end

function tensorProductPoints(p1::Matrix, p2::Matrix)
    n1 = size(p1)[2]
    n2 = size(p2)[2]
    return vcat(repeat(p1,inner=(1,n2)), repeat(p2,outer=(1,n1)))
end

function tensorProduct(box::IntervalBox{2}, quad1d::QuadratureRule1D)
    p1, w1 = transform_quadrature(quad1d, box[1].lo, box[1].hi)
    p2, w2 = transform_quadrature(quad1d, box[2].lo, box[2].hi)
    points = tensorProductPoints(p1, p2)
    weights = kron(w1,w2)
    return QuadratureRule(points,weights)
end

"""
    unique_root_intervals(f, x1::Real, x2::Real)
returns sub-intervals in `[x1,x2]` which contain unique roots of `f`
"""
function unique_root_intervals(f, x1::Real, x2::Real)
    all_roots = roots(f, Interval(x1, x2))
    for r in all_roots
        if r.status != :unique
            msg = "Intervals with possibly non-unique roots were found.\nAdjusting the discretization may fix this."
            throw(ErrorException(msg))
        end
    end
    return [r.interval for r in all_roots]
end

"""
    unique_roots(f, x1::T, x2::T) where {T<:Real}
returns the unique roots of `f` in the interval `(x1, x2)`
"""
function unique_roots(f, x1::T, x2::T) where {T<:Real}
    root_intervals = unique_root_intervals(f, x1, x2)
    _roots = zeros(T, length(root_intervals))
    for (idx,int) in enumerate(root_intervals)
        _roots[idx] = find_zero(f, (int.lo, int.hi), Order1())
    end
    return _roots
end

"""
    roots_and_ends(F::AbstractVector, x1::Real, x2::Real)
return a sorted array containing `[x1, <roots of each f in F>, x2]`
"""
function roots_and_ends(F::AbstractVector, x1::Real, x2::Real)
    r = [x1,x2]
    for f in F
        roots = unique_roots(f, x1, x2)
        append!(r,roots)
    end
    return sort!(r)
end

"""
    root_and_ends(F::AbstractVector, x1::Real, x2::Real)
return a sorted array containing `[x1, <root of each f in F>, x2]`,
this function assumes that each `f` has only one root in `(x1,x2)`, hence
runs faster.
"""
function root_and_ends(F::AbstractVector, x1::Real, x2::Real)
    r = [x1,x2]
    for f in F
        roots = find_zero(f, (x1, x2), Order1())
        append!(r,roots)
    end
    return sort!(r)
end

"""
    quadrature(F::AbstractVector, sign_conditions::AbstractVector, int::Interval, quad::QuadratureRule1D)
return a 1D quadrature rule that can be used to integrate in the domain where
each `f in F` has sign specified in `sign_conditions`
"""
function quadrature(F::AbstractVector, sign_conditions::AbstractVector,
    int::Interval, quad1d::QuadratureRule1D{T,S}) where {T,S}

    @assert length(F) == length(sign_conditions)
    quad = QuadratureRule(T,S,1)

    roots = roots_and_ends(F,int.lo,int.hi)
    for j in 1:length(roots)-1
        xc = 0.5*(roots[j] + roots[j+1])
        if all(i -> F[i](xc)*sign_conditions[i] >= 0.0, 1:length(F))
            p,w = transform_quadrature(quad1d, roots[j], roots[j+1])
            update!(quad, p, w)
        end
    end
    return quad
end

function surface_quadrature(F, height_dir::Int, int::Interval, x0::AbstractVector, w0::Real)

    extended_func(x) = F(extend(x0,height_dir,x))
    root = int.lo
    try
        root = find_zero(extended_func, (int.lo, int.hi), Order1())
    catch err
        msg = "Failed to find a root along given height direction. Check if the given function is bounded by its zero level-set along $height_dir from "*string(x0)
        error(msg)
    end
    p = extend(x0,height_dir,root)
    gradF = gradient(F, p)
    jac = norm(gradF)/(abs(gradF[height_dir]))
    w = w0*jac
    return p, w
end

function surface_quadrature(F, height_dir::Int, int::Interval, x0::Matrix{T}, w0::Vector{S}) where {T,S}

    lower_dim, npoints = size(x0)
    dim = lower_dim + 1
    quad = QuadratureRule(T,S,dim)

    for i in 1:npoints
        x0p = view(x0, :, i)
        w0p = w0[i]
        qp, qw = surface_quadrature(F, height_dir, int, x0p, w0p)
        if !(isnothing(qp))
            update!(quad, qp, qw)
        end
    end
    return quad
end

"""
    extend(x0::Number, k::Int, x::Number)
extend `x0` into 2D space by inserting `x` in direction `k`
"""
function extend(x0::Number, k::Int, x::Number)
    if k == 1
        return [x, x0]
    elseif k == 2
        return [x0, x]
    else
        throw(ArgumentError("Expected k ∈ {1,2}, got $k"))
    end
end

"""
    extend(x0::Number, k::Int, x::AbstractMatrix)
extend the point `x0` along the `k`th direction by using the values `x`
"""
function extend(x0::Number, k::Int, x::AbstractMatrix)
    dim, npoints = size(x)
    old_row = repeat([x0], inner = (1,npoints))

    if dim == 1 && k == 1
        return vcat(x, old_row)
    elseif dim == 1 && k == 2
        return vcat(old_row, x)
    else
        throw(ArgumentError("Not implemented"))
    end
end

function extend(x0::AbstractVector, k::Int, x::Number)
    dim = length(x0)
    if dim == 1 && k == 1
        return [x, x0[1]]
    elseif dim == 1 && k == 2
        return [x0[1], x]
    else
        throw(ArgumentError("Not implemented"))
    end
end

"""
    extend(x0::AbstractVector, k::Int, x::AbstractMatrix)
extend the point vector `x0` along direction `k` treating `x` as the new
coordinate values
"""
function extend(x0::AbstractVector, k::Int, x::AbstractMatrix)
    old_dim = length(x0)
    dim, npoints = size(x)
    @assert dim == 1

    old_row = repeat(x0, inner = (1,npoints))
    if old_dim == 1 && k == 1
        return vcat(x, old_row)
    elseif old_dim == 1 && k == 2
        return vcat(old_row, x)
    else
        throw(ArgumentError("Not implemented"))
    end
end

"""
    extend(x0::AbstractMatrix, k::Int, x::Number)
extend the matrix of points `x0` (treating each column as a point vector)
along direction `k` using the value `x`
"""
function extend(x0::AbstractMatrix, k::Int, x::Number)
    dim, npoints = size(x0)
    new_row = repeat([x], inner = (1,npoints))

    if dim == 1 && k == 1
        return vcat(new_row, x0)
    elseif dim == 1 && k == 2
        return vcat(x0, new_row)
    else
        throw(ArgumentError("Not implemented"))
    end
end

function quadrature(F::AbstractVector, sign_conditions::AbstractVector,
    height_dir::Int, int::Interval, x0::AbstractVector, w0::Real,
    quad1d::QuadratureRule1D{T,S}) where {T,S}

    @assert length(F) == length(sign_conditions)

    lower_dim = length(x0)
    dim = lower_dim + 1
    points = Matrix{T}(undef, dim, 0)
    weights = Vector{T}(undef, 0)

    extended_funcs = [x -> f(extend(x0,height_dir,x)) for f in F]
    roots = roots_and_ends(extended_funcs,int.lo,int.hi)
    for j in 1:length(roots)-1
        xc = 0.5*(roots[j] + roots[j+1])
        if all(i -> extended_funcs[i](xc)*sign_conditions[i] >= 0.0, 1:length(extended_funcs))
            p,w = transform_quadrature(quad1d, roots[j], roots[j+1])
            extended_points = extend(x0,height_dir,p)
            points = hcat(points, extended_points)
            append!(weights, w0*w)
        end
    end
    return points, weights
end

function quadrature(F::AbstractVector, sign_conditions::AbstractVector,
    height_dir::Int, int::Interval, x0::AbstractMatrix, w0::AbstractVector,
    quad1d::QuadratureRule1D{T,S}) where {T,S}

    lower_dim, npoints = size(x0)
    dim = lower_dim + 1
    quad = QuadratureRule(T, S, dim)

    for i in 1:npoints
        x0p = view(x0, :, i)
        w0p = w0[i]
        qp, qw = quadrature(F, sign_conditions, height_dir, int, x0p, w0p, quad1d)
        update!(quad, qp, qw)
    end
    return quad
end

function delete(box::IntervalBox{N,T}, k::Int) where {N,T}
    if k < 1 || k > N
        throw(BoundsError(box, k))
    else
        reduced_intervals = Vector{Interval{T}}(undef, N-1)
        count = 1
        for i in 1:N
            if i != k
                reduced_intervals[count] = box[i]
                count += 1
            end
        end
        new_box = IntervalBox(reduced_intervals)
        return new_box
    end
end

"""
    prune!(F::AbstractVector, sign_conditions::Vector{Int}, box::IntervalBox)
returns a `flag` such that `flag = true` implies that the domain of integration
is non-empty, and `flag = false` implies that the domain of integration is empty
and therefore the algorithm should terminate.
"""
function prune!(F::AbstractVector, sign_conditions::Vector{Int}, box::IntervalBox)
    idx_for_deletion = Int[]
    flag = true
    for (idx,f) in enumerate(F)
        s = sign(f,box)
        if s!= 0
            if s*sign_conditions[idx] >= 0
                push!(idx_for_deletion, idx)
            else
                flag = false
            end
        end
    end
    deleteat!(F, idx_for_deletion)
    deleteat!(sign_conditions, idx_for_deletion)
    return flag
end




function height_direction(F::AbstractVector, box::IntervalBox)
    if length(F) == 0
        throw(ArgumentError("Expected non-empty array"))
    end
    g = abs.(gradient(F[1], mid(box)))
    k = LinearIndices(g)[argmax(g)]
    return k
end

function height_direction(P::InterpolatingPolynomial, box::IntervalBox)
    g = abs.(gradient(P, mid(box)))
    k = LinearIndices(g)[argmax(g)]
    return k
end

function isSuitable(height_dir::Int, F::AbstractVector, box::IntervalBox)
    flag = true
    gradFk = [(x...) -> gradient(f,height_dir,x...) for f in F]
    s = [sign(gk,box) for gk in gradFk]
    if 0 in s
        flag = false
    end
    return flag, s
end

function Base.sign(m::Int,s::Int,S::Bool,sigma::Int)
    if S || m == sigma*s
        return sigma*m
    else
        return 0
    end
end

"""
quadrature rule in 2D on implicitly defined domain
"""
function quadrature(P::InterpolatingPolynomial{N,NF,B}, sign_condition::Int, surface::Bool,
    box::IntervalBox{2}, quad1d::QuadratureRule1D) where {B<:TensorProductBasis{2}} where {N,NF}

    s = sign(P,box)
    if s*sign_condition < 0
        return QuadratureRule(2)
    elseif s*sign_condition > 0
        return tensorProduct(box, quad1d)
    else
        height_dir = height_direction(P, box)
        gradient_sign = sign((x...) -> gradient(P, height_dir, x...), box)
        if gradient_sign == 0
            error("Subdivision not implemented yet")
        else
            lower(x) = P(extend(x,height_dir,box[height_dir].lo))
            upper(x) = P(extend(x,height_dir,box[height_dir].hi))
            lower_sign = sign(gradient_sign,sign_condition,surface,-1)
            upper_sign = sign(gradient_sign,sign_condition,surface,+1)
            lower_box = height_dir == 1 ? box[2] : box[1]
            quad = quadrature([lower, upper], [lower_sign, upper_sign], lower_box, quad1d)
            if surface
                return surface_quadrature(P, height_dir, box[height_dir], quad.points, quad.weights)
            else
                return quadrature([P], [sign_condition], height_dir, box[height_dir], quad.points, quad.weights, quad1d)
            end
        end
    end
end

function circleshape(xc, r, theta_start, theta_end)
    theta = LinRange(theta_start, theta_end, 1000)
    return (xc[1] .+ r*cos.(theta)), (xc[2] .+ r*sin.(theta))
end

function circleArea(xc, r, theta_start)
    theta_center = 2*(pi - theta_start)
    A = (1.0 - (xc[1] + r*cos(theta_start)))*2 + r^2/2*(theta_center - sin(theta_center))
end

function plotcircle(x,r,theta_start,theta_end)
    plot!(circleshape(x,r,theta_start,theta_end), lw = 2.0,
        legend = false, aspect_ratio = 1.0, color = :blue)
end

function distance(p::AbstractVector,xc,r)
    return norm(p - xc) - r
end

function distance(p::AbstractMatrix,xc,r)
    ndim, npts = size(p)
    return [distance(p[:,i],xc,r) for i in 1:npts]
end

function makePlot(P; nquad = 5, poly_order = 2, box = IntervalBox(-1..1,2), filename = "test_quadrature.png")
    quad1d = QuadratureRule1D(nquad)
    pquad = quadrature(P, 1, false, box, quad1d)
    nquad = quadrature(P, -1, false, box, quad1d)
    zquad = quadrature(P, 0, true, box, quad1d)

    xrange = box[1].lo:1e-3:box[1].hi
    yrange = box[2].lo:1e-3:box[2].hi
    contour(xrange, xrange, (x,y) -> P(x,y), lw = 10, levels = [0.0], aspect_ratio = 1.0, size = (1280,1280), legend = false)
    plot!(box)
    scatter!(pquad.points[1,:], pquad.points[2,:], marker = (:cross, 12, :blue), label = "Positive", size = (1280,1280))
    scatter!(nquad.points[1,:], nquad.points[2,:], marker = (:xcross, 12, :green), label = "Negative")
    scatter!(zquad.points[1,:], zquad.points[2,:], marker = (:rect, 12, :red), label = "Surface")
    savefig(filename)
end

circle_center = [-1.5,1.0]
circle_radius = 1.5
poly_order = 5
nquad = 20

P = InterpolatingPolynomial(1,2,poly_order)
points = P.basis.points
coeffs = distance(points,circle_center,circle_radius)
ImplicitDomainQuadrature.update!(P,coeffs)
quad1d = QuadratureRule1D(nquad)

# makePlot(P, nquad = nquad, filename = "high_order.png")

box = IntervalBox(-1 .. 1,2)

pquad = quadrature(P, +1, false, box, quad1d)
pquad = quadrature(P, +1, false, box, quad1d)
zquad = quadrature(P, 0, true, box, quad1d)
surface_points = quadrature(P, sign_condition, true, box, quad1d)
# #
# points2d = quad2d.points
# # surface_points = surface_quad.points
# #
# xrange = -1:1e-3:1
# contour(xrange, xrange, (x,y) -> P(x,y), lw = 10, levels = [0.0], aspect_ratio = 1.0, size = (1280,1280))
# plot!(box, size = (800,800), label = ""))
# scatter!(points2d[1,:], points2d[2,:], marker = (:hexagon, 12), label = "Surface")
# savefig("test_quadrature.png")
# scatter!(surface_points[1,:], surface_points[2,:], markersize = 6)
# plotcircle(circle_center,circle_radius,0,2pi)
