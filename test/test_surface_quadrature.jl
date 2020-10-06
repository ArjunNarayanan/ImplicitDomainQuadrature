using Test
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
quad1d = IDQ.ReferenceQuadratureRule(5)
box = IntervalBox(-1..1, 2)
P = InterpolatingPolynomial(1, 2, 2)
coeffs = [f2(P.basis.points[:, i]) for i = 1:size(P.basis.points)[2]]
update!(P, coeffs)
quad = IDQ.surface_quadrature(P, box, quad1d)
p = IDQ.extend([0.0], 1, quad1d.points)
@test allapprox(p, quad.points)
@test allapprox(quad1d.weights, quad.weights)

f2(x) = (x[2] - 0.5) * (x[2] + 0.75)
quad1d = IDQ.ReferenceQuadratureRule(5)
box = IntervalBox(-1..1, 2)
P = InterpolatingPolynomial(1, 2, 2)
coeffs = [f2(P.basis.points[:, i]) for i = 1:size(P.basis.points)[2]]
update!(P, coeffs)
quad = IDQ.surface_quadrature(P, box, quad1d)

function integrate(f,quad)
    s = 0.0
    for (p,w) in quad
        s += f(p)*w
    end
    return s
end

f(x) = (x[2] ≈ -0.75 || x[2] ≈ 0.5) ? 1.0 : 0.0
s = integrate(f,quad)
@test s ≈ 4.0

f2(x) = (x[2] - 0.5) * (x[2] + 0.5)
quad1d = IDQ.ReferenceQuadratureRule(5)
box = IntervalBox(-1..1, 2)
P = InterpolatingPolynomial(1, 2, 2)
coeffs = [f2(P.basis.points[:, i]) for i = 1:size(P.basis.points)[2]]
update!(P, coeffs)
quad = IDQ.surface_quadrature(P, box, quad1d)

f(x) =
    (isapprox(x[2], 0.5, atol = 2e-2) || isapprox(x[2], -0.5, atol = 2e-2)) ?
    1.0 : 0.0
s = integrate(f,quad)
@test s ≈ 4.0
