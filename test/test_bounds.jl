using Test
using PolynomialBasis
using IntervalArithmetic
# using Revise
using ImplicitDomainQuadrature
IDQ = ImplicitDomainQuadrature

box = IntervalBox([1.0, 2], [3.0, 5.0])
xL, xR = IDQ.corners(box)
@test all(xL .≈ [1.0, 2.0])
@test all(xR .≈ [3.0, 5.0])

b1, b2, b3, b4 = IDQ.split_box(box)
testb1 = IntervalBox([1.0, 2], [2, 3.5])
testb2 = IntervalBox([2.0, 2], [3, 3.5])
testb3 = IntervalBox([2.0, 3.5], [3.0, 5])
testb4 = IntervalBox([1.0, 3.5], [2, 5.0])
@test b1 == testb1
@test b2 == testb2
@test b3 == testb3
@test b4 == testb4

box = IntervalBox(0..1, 0..2, 0..5)
@test IDQ.min_diam(box) ≈ 1.0

f(x) = x[1] + 1
box = IntervalBox(-1..1, 1)
s = IDQ.interval_arithmetic_sign_search(f, box, 1e-2,0.0)
@test !(s == 0 || s == 1 || s == -1)

f(x) = x[1] + 2
box = IntervalBox(-1..1, 1)
@test IDQ.interval_arithmetic_sign_search(f, box, 1e-2,0.0) == 1

f(x) = x[1] - 2
box = IntervalBox(-1..1, 1)
@test IDQ.interval_arithmetic_sign_search(f, box, 1e-2,0.0) == -1

f(x) = x[1]
box = IntervalBox(-1..1, 1)
@test IDQ.interval_arithmetic_sign_search(f, box, 1e-2,0.0) == 0
@test sign(f, [-1.], [1.], tol = 1e-2) == 0

f(x) = x[1] * (x[1] - 0.1) * (x[1] - 1.0)
@test sign(f, [-1.], [1.], tol = 1e-2) == 0

f(x) = (x[1] - 0.5) * (x[1] + 0.5) + 10
@test sign(f, [-1.], [1.], tol = 1e-2) == 1

f(x) = (x[1] - 0.5) * (x[1] + 0.5) - 3
@test sign(f, [-1.], [1.], tol = 1e-2) == -1

f(x) = x[1] * (x[1] - 1e-4)
s = sign(f, [-1.], [1.], tol = 1e-2)
@test !(s == 0 || s == +1 || s == -1)

box = IntervalBox(2..3, 1)
g(x) = (x[1] - 2.5)*(x[1] - 2.6)*(x[1] - 2.9)
@test sign(g, [2.], [3.], tol = 1e-2) == 0

r = 0.5
f2(x) = x[1]^2 + x[2]^2 - r^2
@test sign(f2, [-1.,-1.], [1.,1.], tol = 1e-2) == 0

g2(x) = sin(2pi * x[1]) * cos(2pi * x[2]) + 2
@test sign(g2, [-1.,-1.], [1.,1.], tol = 1e-2) == 1

P = InterpolatingPolynomial(1, 2, 2)
coeffs = 1:9
PolynomialBasis.update!(P, coeffs)
xL = [-1.,-1.]
xR = [1.,1.]
max_coeff, min_coeff = IDQ.extremal_coeffs_in_box(P, xL,xR)
@test max_coeff ≈ 9.0
@test min_coeff ≈ 1.0

xL = [0.5,0.5]
xR = [1.,1.]
max_coeff, min_coeff = IDQ.extremal_coeffs_in_box(P, xL,xR)
@test max_coeff ≈ 9.0
@test min_coeff ≈ 9.0

xL = [0.6,0.6]
xR = [0.7,0.7]
max_coeff, min_coeff = IDQ.extremal_coeffs_in_box(P, xL,xR)
@test max_coeff == -Inf
@test min_coeff == Inf

P = InterpolatingPolynomial(1, 2, 3)
coeffs = (1:16) .- 5.5
PolynomialBasis.update!(P, coeffs)
xL = [-1.,-1.]
xR = [1.,1.]
s = sign(P, xL, xR)
@test s == 0

P = InterpolatingPolynomial(1, 2, 3)
coeffs = 1.0:16.0
PolynomialBasis.update!(P, coeffs)
s = sign(P, xL,xR)
@test s == 1
