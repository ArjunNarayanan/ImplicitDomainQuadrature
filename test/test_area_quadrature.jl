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

f(x) = (x - 1.0)*(x - 2.0)*(x - 3.0)
g(x) = (x - 0.1)*(x - 0.2)*(x - 2.5)

quad1d = IDQ.ReferenceQuadratureRule(3)
f2(x) = f(x[1])
g2(x) = g(x[1])
w0 = 2.0
@test_throws AssertionError IDQ.extend_edge_quadrature_to_area_quadrature(
    [f2, g2],
    [+1],
    1,
    0.0,
    3.5,
    [1.0],
    w0,
    quad1d,
)
quad1 = IDQ.extend_edge_quadrature_to_area_quadrature(
    [f2, g2],
    [+1, -1],
    1,
    0.0,
    3.5,
    [1.0],
    w0,
    quad1d,
)
quad2 = IDQ.one_dimensional_quadrature([f, g], [+1, -1], 0.0, 3.5, quad1d)
@test allapprox(quad1.points, IDQ.extend([1.0], 1, quad2.points))
@test allapprox(quad1.weights, w0 * quad2.weights)


edgequad = IDQ.TemporaryQuadrature(reshape([1.0], 1, 1), [w0])
quad1 = IDQ.extend_edge_quadrature_to_area_quadrature(
    [f2, g2],
    [+1, -1],
    1,
    Interval(0.0, 3.5),
    edgequad,
    quad1d,
)
quad2 =
    IDQ.one_dimensional_quadrature([f, g], [+1, -1], Interval(0.0, 3.5), quad1d)
@test allapprox(quad1.points, IDQ.extend([1.0], 1, quad2.points))
@test allapprox(quad1.weights, w0 * quad2.weights)


x0 = [1.0 2.0 3.0]
w0 = [3.0, 6.0, 9.0, 10.0]
@test_throws AssertionError IDQ.extend_edge_quadrature_to_area_quadrature(
    [f2, g2],
    [+1, -1, +1],
    1,
    0.0,
    3.5,
    x0,
    w0,
    quad1d,
)
@test_throws AssertionError IDQ.extend_edge_quadrature_to_area_quadrature(
    [f2, g2],
    [+1, -1],
    1,
    0.0,
    3.5,
    x0,
    w0,
    quad1d,
)


w0 = [3.0, 6.0, 9.0]
quad_ext = IDQ.extend_edge_quadrature_to_area_quadrature(
    [f2, g2],
    [+1, -1],
    1,
    0.0,
    3.5,
    x0,
    w0,
    quad1d,
)
quad = IDQ.one_dimensional_quadrature([f, g], [+1, -1], 0.0, 3.5, quad1d)
testp = hcat([IDQ.extend([x0[i]], 1, quad.points) for i = 1:3]...)
testw = vcat([quad.weights * w0[i] for i = 1:3]...)
@test allapprox(quad_ext.points, testp)
@test allapprox(quad_ext.weights, testw)

edgequad = IDQ.TemporaryQuadrature(x0, w0)
quad_ext = IDQ.extend_edge_quadrature_to_area_quadrature(
    [f2, g2],
    [+1, -1],
    1,
    Interval(0.0, 3.5),
    edgequad,
    quad1d,
)
@test allapprox(quad_ext.points, testp)
@test allapprox(quad_ext.weights, testw)


f2(x) = x[2]
quad1d = IDQ.ReferenceQuadratureRule(5)
box = IntervalBox(-1..1, 2)
P = InterpolatingPolynomial(1, 2, 2)
coeffs = [f2(P.basis.points[:, i]) for i = 1:size(P.basis.points)[2]]
update!(P, coeffs)
quad = IDQ.area_quadrature(P, +1, box, quad1d)
p, w = IDQ.transform(quad1d, 0.0, 1.0)
p2 = hcat([IDQ.extend([quad1d.points[i]], 2, p) for i = 1:5]...)
w2 = vcat([quad1d.weights[i] * w for i = 1:5]...)
@test allapprox(quad.points, p2)
@test allapprox(quad.weights, w2)


f2(x) = x[2] + 1.5
quad1d = IDQ.ReferenceQuadratureRule(5)
box = IntervalBox(-1..1, 2)
P = InterpolatingPolynomial(1, 2, 2)
coeffs = [f2(P.basis.points[:, i]) for i = 1:size(P.basis.points)[2]]
update!(P, coeffs)
quad = IDQ.area_quadrature(P, -1, box, quad1d)
@test size(quad.points) == (2, 0)
@test length(quad.weights) == 0

f2(x) = x[2] + 1.5
quad1d = IDQ.ReferenceQuadratureRule(5)
box = IntervalBox(-1..1, 2)
P = InterpolatingPolynomial(1, 2, 2)
coeffs = [f2(P.basis.points[:, i]) for i = 1:size(P.basis.points)[2]]
update!(P, coeffs)
quad = IDQ.area_quadrature(P, +1, box, quad1d)
p = hcat([IDQ.extend([quad1d.points[i]], 2, quad1d.points) for i = 1:5]...)
w = vcat([quad1d.weights[i] * quad1d.weights for i = 1:5]...)
@test allapprox(quad.points, p)
@test allapprox(quad.weights, w)
