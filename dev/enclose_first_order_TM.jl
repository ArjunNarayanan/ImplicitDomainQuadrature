using TaylorModels, IntervalArithmetic, StaticArrays, LinearAlgebra
using BenchmarkTools
using ImplicitDomainQuadrature

function boxCenters(box::IntervalBox{N}) where {N}
    return [IntervalBox(i..i,N) for i in mid(box)]
end

function taylorN(order::Int, box::IntervalBox{N,T}) where {N,T}
    set_variables(T, "x", numvars = N)
    centers = boxCenters(box)
    return [TaylorModelN(i,order,centers[i],box) for i in 1:N]
end

function corners(box::IntervalBox{1})
    b = box[1]
    return @SMatrix [b.lo b.hi]
end

function corners(box::IntervalBox{2})
    b1 = box[1]
    b2 = box[2]
    return @SMatrix [b1.lo  b1.lo  b1.hi  b1.hi
                     b2.lo  b2.hi  b2.lo  b2.hi]
end

function corners(box::IntervalBox{3})
    b1 = box[1]
    b2 = box[2]
    b3 = box[3]
    return @SMatrix [b1.lo  b1.lo  b1.lo  b1.lo  b1.hi  b1.hi  b1.hi  b1.hi
                     b2.lo  b2.lo  b2.hi  b2.hi  b2.lo  b2.lo  b2.hi  b2.hi
                     b3.lo  b3.hi  b3.lo  b3.hi  b3.lo  b3.hi  b3.lo  b3.hi]
end

function TaylorSeries.evaluate(pol::TaylorN, points::AbstractMatrix)
    num_points = size(points)[2]
    return [pol(view(points, :, i)) for i = 1:num_points]
end

function bound(f, box)
    tm = taylorN(1,box)
    ftm = f(tm...)
    points = corners(box)
    vals = evaluate(ftm.pol, points)
    max_val = maximum(sup.(vals))
    min_val = minimum(inf.(vals))
    return union(max_val + ftm.rem, min_val + ftm.rem)
end

function circleShape(xc,r,start,stop)
    theta = LinRange(start,stop,1000)
    return xc[1] .+ r*cos.(theta), xc[2] .+ r*sin.(theta)
end

function plotcircle(xc,r,start,stop)
    plot!(circleShape(xc,r,start,stop), lw = 4.0, c = :blue, legend = false, aspect_ratio = 1.0)
end

function distance(xc,r,p)
    return norm(xc - p) - r
end





xc = [1.0,0.0]
r = 2.25
const TMORDER = 1
const POLYORDER = 3

P = InterpolatingPolynomial(1,2,POLYORDER)
points = P.basis.points
num_pts = size(points)[2]
coeffs = [distance(xc,r,points[:,i]) for i = 1:num_pts]
ImplicitDomainQuadrature.update!(P, coeffs)
box = IntervalBox(-1..1,2)
dPx(x,y) = gradient(P, 1, x, y)

box = IntervalBox(-1..1,2)
b = @btime bound(P,box)
