using Test
using PolynomialBasis
using IntervalArithmetic
using TaylorModels
# using Revise
using ImplicitDomainQuadrature
IDQ = ImplicitDomainQuadrature

box = IntervalBox([1.,2],[3.,5.])
xL,xR = IDQ.corners(box)
@test all(xL .≈ [1.,2.])
@test all(xR .≈ [3.,5.])

b1,b2,b3,b4 = IDQ.split_box(box)
testb1 = IntervalBox([1.,2],[2,3.5])
testb2 = IntervalBox([2.,2],[3,3.5])
testb3 = IntervalBox([2.,3.5],[3.,5])
testb4 = IntervalBox([1.,3.5],[2,5.])
@test b1 == testb1
@test b2 == testb2
@test b3 == testb3
@test b4 == testb4

box = IntervalBox(0..1,0..2,0..5)
@test IDQ.min_diam(box) ≈ 1.0

@test IDQ.zeroBox(2) == IntervalBox(0..0,2)
@test IDQ.symBox(3) == IntervalBox(-1..1,3)

set_variables(Float64, "x", order = 4, numvars = 1)
f(x) = x + 1
box = IntervalBox(-1..1,1)
@test IDQ.bound(f, box, 1) == 0..2
@test IDQ.bound(f, box, 2) == 0..2
box = IntervalBox(0.5 .. 0.6, 1)
@test IDQ.bound(f, box, 1) == f(box)[1]

box = IntervalBox(-1..1,1)
@test_throws ErrorException IDQ.taylor_models_sign_search(f,box,1,1e-2)

f(x) = x + 2
box = IntervalBox(-1..1,1)
@test IDQ.taylor_models_sign_search(f, box, 1, 1e-2) == 1

f(x) = x - 2
box = IntervalBox(-1..1,1)
@test IDQ.taylor_models_sign_search(f, box, 1, 1e-2) == -1

f(x) = x
box = IntervalBox(-1..1,1)
@test IDQ.taylor_models_sign_search(f,box,1,1e-2) == 0

box = IntervalBox(-1..1,1)
@test sign(f, box, order = 1, tol = 1e-2) == 0
f(x) = x*(x - 0.1)*(x - 1.0)
@test sign(f, box, order = 1, tol = 1e-2) == 0

f(x) = (x - 0.5)*(x+0.5) + 10
box = IntervalBox(-1..1,1)
@test sign(f, box, order = 1, tol = 1e-2) == 1

f(x) = (x - 0.5)*(x+0.5) - 3
box = IntervalBox(-1..1,1)
@test sign(f, box, order = 1, tol = 1e-2) == -1

f(x) = x*(x - 1e-4)
box = IntervalBox(-1..1,1)
@test_throws ErrorException sign(f, box, order = 1, tol = 1e-2)

box = IntervalBox(2 .. 3,1)
g(x) = (x - 2.5)*(x - 2.6)*(x - 2.9)
orders = 1:10
s = [sign(g,box, order = i, tol = 1e-2) for i in orders]
@test all(i -> s[i] == 0, 1:length(orders))

r = 0.5
f2(x,y) = x^2 + y^2 - r^2
box = IntervalBox(-1..1,2)
s = [sign(f2,box, order = i, tol = 1e-2) for i in orders]
@test all(i -> s[i] == 0, 1:length(orders))

g2(x,y) = sin(2pi*x)*cos(2pi*y) + 2
box = IntervalBox(-1..1,2)
s = [sign(g2,box, order = i, tol = 1e-2) for i in orders]
@test all(i -> s[i] == 1, 1:length(orders))

P = InterpolatingPolynomial(1,2,2)
coeffs = 1:9
PolynomialBasis.update!(P,coeffs)
box = IntervalBox(-1..1,2)
max_coeff, min_coeff = IDQ.extremal_coeffs_in_box(P,box)
@test max_coeff ≈ 9.0
@test min_coeff ≈ 1.0
box = IntervalBox(0.5 .. 1.0,2)
max_coeff, min_coeff = IDQ.extremal_coeffs_in_box(P,box)
@test max_coeff ≈ 9.0
@test min_coeff ≈ 9.0
box = IntervalBox(0.6..0.7,2)
max_coeff, min_coeff = IDQ.extremal_coeffs_in_box(P,box)
@test max_coeff == -Inf
@test min_coeff == Inf

P = InterpolatingPolynomial(1,2,3)
coeffs = (1:16) .- 5.5
PolynomialBasis.update!(P,coeffs)
box = IntervalBox(-1..1,2)
s = sign(P,box)
@test s == 0

P = InterpolatingPolynomial(1,2,3)
coeffs = 1.0:16.0
PolynomialBasis.update!(P,coeffs)
box = IntervalBox(-1..1,2)
s = sign(P,box)
@test s == 1
