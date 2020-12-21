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

f(x) = (x - 1.0) * (x - 2.0) * (x - 3.0)
g(x) = (x - 0.1) * (x - 0.2) * (x - 2.5)
@test IDQ.sign_conditions_satisfied([f, g], 0.5, [-1, -1])
@test IDQ.sign_conditions_satisfied([f, g], 1.5, [+1, -1])

quad1d = IDQ.ReferenceQuadratureRule(3)
@test_throws AssertionError IDQ.one_dimensional_quadrature(
    [f, g],
    [+1, -1, +1],
    0.0,
    3.5,
    quad1d,
)

quad = IDQ.one_dimensional_quadrature([f, g], [+1, -1], 0.0, 3.5, quad1d)
p, w = IDQ.transform(quad1d, 1.0, 2.0)
@test allapprox(quad.points, p)
@test allapprox(quad.weights, w)

quad =
    IDQ.one_dimensional_quadrature([f, g], [+1, +1], Interval(0.0, 3.5), quad1d)
p, w = IDQ.transform(quad1d, 3.0, 3.5)
@test allapprox(quad.points, p)
@test allapprox(quad.weights, w)
