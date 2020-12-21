using Test
using LinearAlgebra
using IntervalArithmetic
using PolynomialBasis
# using Revise
using ImplicitDomainQuadrature

IDQ = ImplicitDomainQuadrature

function allapprox(v1, v2; tol = 1e-14)
    @assert length(v1) == length(v2)
    flags = [isapprox(v1[i], v2[i], atol = tol) for i = 1:length(v1)]
    return all(flags)
end

function circle_distance_function(coords, center, radius)
    npts = size(coords)[2]
    return [norm(coords[:, i] - center) - radius for i = 1:npts]
end

function plane_distance_function(coords, normal, x0)
    return (coords .- x0)' * normal
end

function plot_zero_levelset(poly)
    x = -1:1e-2:1
    contour(x, x, (x, y) -> poly(x, y), levels = [0.0])
end

struct FunctionAndGradient
    func::Any
    grad::Any
    perturbation::Any
end

function (f::FunctionAndGradient)(x)
    return f.func(x) + perturbation
end

function PolynomialBasis.gradient(f::FunctionAndGradient, x)
    return f.grad(x)
end

poly = InterpolatingPolynomial(1, 2, 2)
x0 = [1.0, 0.0]
normal = [1.0, 0.0]
coeffs = plane_distance_function(poly.basis.points, normal, x0)
update!(poly, coeffs)

xL, xR = [-1.0, -1.0], [1.0, 1.0]
numqp = 2

quad = area_quadrature(poly,-1,xL,xR,numqp)
@test isapprox(sum(quad.weights), 4.0)

quad = area_quadrature(poly,+1,xL,xR,numqp)
@test isapprox(sum(quad.weights), 0.0, atol = 0.25)

x0 = [1.0, 1.0]
normal = [0.0, 1.0]
coeffs = plane_distance_function(poly.basis.points, normal, x0)
update!(poly, coeffs)

quad = IDQ.area_quadrature(poly,+1, xL, xR, numqp)
@test isapprox(sum(quad.weights), 0.0, atol = 0.25)

quad = IDQ.area_quadrature(poly,-1, xL, xR, numqp)
@test isapprox(sum(quad.weights), 4.0)

x0 = [-1.0, 1.0]
normal = [1.0, 0.0]
coeffs = plane_distance_function(poly.basis.points, normal, x0)
update!(poly, coeffs)

quad = IDQ.area_quadrature(poly, +1, xL, xR, numqp)
@test isapprox(sum(quad.weights), 4.0)

quad = IDQ.area_quadrature(poly,-1, xL, xR, numqp)
@test isapprox(sum(quad.weights), 0.0, atol = 0.25)

x0 = [1.0, -1.0]
normal = [0.0, 1.0]
coeffs = plane_distance_function(poly.basis.points, normal, x0)
update!(poly, coeffs)

quad = IDQ.area_quadrature(poly,+1, xL, xR, numqp)
@test isapprox(sum(quad.weights), 4.0)

quad = IDQ.area_quadrature(poly, -1, xL, xR, numqp)
@test isapprox(sum(quad.weights), 0.0, atol = 0.25)



#######################################################################
# Test a curved interface and subdivision
radius = 0.5
center = [0.0, 0.0]
poly = InterpolatingPolynomial(1, 2, 3)
coeffs = circle_distance_function(poly.basis.points, center, radius)
update!(poly, coeffs)
testarea = 4.0 - pi * 0.5^2

numqp = 5
quad = IDQ.area_quadrature(poly, +1, xL,xR, numqp)
@test isapprox(sum(quad.weights), testarea, atol = 1e-1)

squad = IDQ.surface_quadrature(poly, xL,xR, numqp)
@test isapprox(sum(quad.weights), pi, atol = 1e-1)
