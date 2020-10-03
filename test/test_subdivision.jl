using Test
using Plots
using LinearAlgebra
using PolynomialBasis
using IntervalArithmetic
using Revise
using ImplicitDomainQuadrature

IDQ = ImplicitDomainQuadrature

function allapprox(v1, v2; tol = 1e-14)
    @assert length(v1) == length(v2)
    flags = [isapprox(v1[i], v2[i], atol = tol) for i = 1:length(v1)]
    return all(flags)
end

function circle_distance_function(coords,center,radius)
    npts = size(coords)[2]
    return [norm(coords[:,i] - center) - radius for i = 1:npts]
end

function plane_distance_function(coords,normal,x0)
    return (coords .- x0)'*normal
end

function plot_zero_levelset(poly)
    x = -1:1e-2:1
    contour(x,x,(x,y) -> poly(x,y), levels = [0.])
end

radius = 1.2
circcenter = [radius,0.0]
poly = InterpolatingPolynomial(1,2,2)
# coeffs = circle_distance_function(poly.basis.points,circcenter,radius)
x0 = [-1.,0.]
coeffs = plane_distance_function(poly.basis.points,[1.,0.],x0)
update!(poly,coeffs)

box = IntervalBox(-1..1,2)
# IDQ.is_suitable(1,poly,box)

quad1d = IDQ.ReferenceQuadratureRule(2)
quad = IDQ.area_quadrature(poly,+1,box,quad1d)


plot_zero_levelset(poly)
scatter!(quad.points[1,:],quad.points[2,:],legend=false)

# testp = [0.,0.]
# testw = [4.0]
# @test allapprox(quad.points,testp)
# @test allapprox(quad.weights,testw)
#
# quad = IDQ.area_quadrature(poly,+1,box,quad1d)
# p,w = IDQ.transform(quad1d, 0.0, 1.0)
# p2 = hcat([IDQ.extend([quad1d.points[i]], 2, p) for i in 1:5]...)
# w2 = vcat([quad1d.weights[i]*w for i in 1:5]...)
# @test allapprox(quad.points,p2)
# @test allapprox(quad.weights,w2)
