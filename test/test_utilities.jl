using Test
using PolynomialBasis
using IntervalArithmetic
# using Revise
using ImplicitDomainQuadrature

IDQ = ImplicitDomainQuadrature

function allapprox(v1, v2; tol = 1e-14)
    @assert length(v1) == length(v2)
    flags = [isapprox(v1[i], v2[i], atol = tol) for i = 1:length(v1)]
    return all(flags)
end

r = sort!(IDQ.unique_root_intervals(sin, pi / 2, 5pi / 2))
@test length(r) == 2
@test pi in r[1]
@test 2pi in r[2]

r = sort!(IDQ.unique_roots(cos, 0.0, 2pi))
@test length(r) == 2
@test pi / 2 ≈ r[1]
@test 3pi / 2 ≈ r[2]

f(x) = (x - 1.0) * (x - 1.5)
g(x) = (x - 0.5) * (x - 1.25)
r = IDQ.roots_and_ends([f, g], 0.0, 2.0)
testr = [0.0, 0.5, 1.0, 1.25, 1.5, 2.0]
@test allapprox(r, testr)

@test IDQ.extend(1.0, 1, 2.0) ≈ [2.0, 1.0]
@test IDQ.extend(1.0, 2, 2.0) ≈ [1.0, 2.0]
@test_throws ArgumentError IDQ.extend(1.0, 3, 2.0)

@test allapprox(IDQ.extend([1.0], 1, 2.0), [2.0, 1.0])
@test_throws ArgumentError IDQ.extend([1.0, 2.0], 1, 2.0)

x0 = [1.0]
x = [
    2.0 3.0 4.0 5.0
    6.0 7.0 8.0 9.0
]
@test_throws ArgumentError IDQ.extend(x0, 1, x)

x0 = [1.0]
x = [2.0 3.0 4.0 5.0]
testx = [
    2.0 3.0 4.0 5.0
    1.0 1.0 1.0 1.0
]
@test allapprox(IDQ.extend(x0, 1, x), testx)

x0 = [1.0]
x = [2.0 3.0 4.0 5.0]
testx = [
    1.0 1.0 1.0 1.0
    2.0 3.0 4.0 5.0
]
@test allapprox(IDQ.extend(x0, 2, x), testx)

x0 = [1.0]
x = [2.0 3.0 4.0 5.0]
@test_throws ArgumentError IDQ.extend(x0, 3, x)


f2(x) = x[2]
P = InterpolatingPolynomial(1, 2, 2)
coeffs = [f2(P.basis.points[:, i]) for i = 1:size(P.basis.points)[2]]
update!(P, coeffs)
@test IDQ.height_direction(x -> gradient(P, x), [0.0, 0.0]) == 2
box = IntervalBox(-1..1, 2)
@test IDQ.height_direction(x -> gradient(P, x), box) == 2

flag, s = IDQ.is_suitable(2, x -> gradient(P, x), box)
@test flag == true
@test s == 1

f2(x) = (x[2] + 0.5) * (x[2] - 0.5)
P = InterpolatingPolynomial(1, 2, 2)
coeffs = [f2(P.basis.points[:, i]) for i = 1:size(P.basis.points)[2]]
update!(P, coeffs)
flag, s = IDQ.is_suitable(2, x -> gradient(P, x), box)
@test flag == false

@test IDQ.sign(1, 1, true, -1) == -1
@test IDQ.sign(1, -1, false, -1) == -1
@test IDQ.sign(1, -1, false, 1) == 0
