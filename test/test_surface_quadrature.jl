using Test
using LinearAlgebra
using PolynomialBasis
using Roots
using IntervalArithmetic
# using Revise
using ImplicitDomainQuadrature

IDQ = ImplicitDomainQuadrature

function allapprox(v1, v2; tol = 1e-14)
    @assert length(v1) == length(v2)
    flags = [isapprox(v1[i], v2[i], atol = tol) for i = 1:length(v1)]
    return all(flags)
end

f2(x) = x[1] - 0.5
P = InterpolatingPolynomial(1, 2, 2)
coeffs = [f2(P.basis.points[:, i]) for i = 1:size(P.basis.points)[2]]
update!(P, coeffs)
x0 = [1.0]
w0 = 3.0
p, w = IDQ.extend_edge_quadrature_to_surface_quadrature(
    P,
    x -> gradient(P, x),
    1,
    -1.0,
    1.0,
    x0,
    w0,
)
@test allapprox(p, [0.5, 1.0])
@test w ≈ w0

f2(x) = x[1]
P = InterpolatingPolynomial(1, 2, 2)
coeffs = [f2(P.basis.points[:, i]) for i = 1:size(P.basis.points)[2]]
update!(P, coeffs)
x0 = [-0.5 0.0 0.5]
w0 = [2.0, 3.0, 4.0]
@test_throws AssertionError IDQ.extend_edge_quadrature_to_surface_quadrature(
    P,
    x -> gradient(P, x),
    1,
    -1.0,
    1.0,
    x0,
    [2.0, 3.0, 4.0, 5.0],
)
surf_quad = IDQ.extend_edge_quadrature_to_surface_quadrature(
    P,
    x -> gradient(P, x),
    1,
    -1.0,
    1.0,
    x0,
    w0,
)
p = hcat([IDQ.extend(x0[i], 1, 0.0) for i = 1:3]...)
@test allapprox(surf_quad.points, p)
@test allapprox(surf_quad.weights, w0)

edgequad = IDQ.TemporaryQuadrature(x0, w0)
surf_quad = IDQ.extend_edge_quadrature_to_surface_quadrature(
    P,
    x -> gradient(P, x),
    1,
    Interval(-1.0, 1.0),
    edgequad,
)
@test allapprox(surf_quad.points, p)
@test allapprox(surf_quad.weights, w0)


f2(x) = x[2]
numqp = 5
quad1d = IDQ.ReferenceQuadratureRule(numqp)
xL, xR = [-1.0, -1.0], [1.0, 1.0]
P = InterpolatingPolynomial(1, 2, 2)
coeffs = [f2(P.basis.points[:, i]) for i = 1:size(P.basis.points)[2]]
update!(P, coeffs)
quad = IDQ.surface_quadrature(P, xL, xR, numqp)
p = IDQ.extend([0.0], 1, quad1d.points)
@test allapprox(p, quad.points)
@test allapprox(quad1d.weights, quad.weights)

f2(x) = (x[2] - 0.5) * (x[2] + 0.75)
numqp = 5
quad1d = IDQ.ReferenceQuadratureRule(numqp)
xL, xR = [-1.0, -1.0], [1.0, 1.0]
P = InterpolatingPolynomial(1, 2, 2)
coeffs = [f2(P.basis.points[:, i]) for i = 1:size(P.basis.points)[2]]
update!(P, coeffs)
quad = IDQ.surface_quadrature(P, xL, xR, numqp)

function integrate(f, quad)
    s = 0.0
    for (p, w) in quad
        s += f(p) * w
    end
    return s
end

f(x) = (x[2] ≈ -0.75 || x[2] ≈ 0.5) ? 1.0 : 0.0
s = integrate(f, quad)
@test s ≈ 4.0

function circle_distance_function(coords, center, radius)
    npts = size(coords)[2]
    return [norm(coords[:, i] - center) - radius for i = 1:npts]
end

radius = 0.5
center = [0.0, 0.0]
poly = InterpolatingPolynomial(1, 2, 3)
numqp = 5
xL, xR = [-1.0, -1.0], [1.0, 1.0]
coeffs = circle_distance_function(poly.basis.points, center, radius)
update!(poly, coeffs)
squad = IDQ.surface_quadrature(poly, xL,xR, numqp)

f(x) = isapprox(norm(x),radius,atol=1e-1) ? 1.0 : 0.0
s = integrate(f,squad)
@test isapprox(s,sum(squad.weights))
@test isapprox(s,pi,atol=1e-1)
