using ImplicitDomainQuadrature
using Test
using StaticArrays
using IntervalArithmetic
using TaylorModels
using BranchAndPrune

IDQ = ImplicitDomainQuadrature

f(x) = x*(0.1 - x)*(1.0 - x)
box = IntervalBox(-1..1,1)
@test_throws ArgumentError IDQ.SignSearch(f, box, +1, -1e-2)
@test_throws ArgumentError IDQ.SignSearch(f, box, -1, 1e-3)

box = IntervalBox(-1..1,3)
search = IDQ.SignSearch(f, box, +5, 1e-2)
@test length(get_variables()) == 3
@test get_order() == 10

box = IntervalBox(-1..1,1)
search = IDQ.SignSearch(f, box, 1, 1e-2)
@test search.tol ≈ 2*1e-2
@test search.initial == box
@test search.found_positive == false
@test search.found_negative == false
@test search.breached_tolerance == false
@test search.order == 1

box = IntervalBox(0..1,0..2,0..5)
@test IDQ.min_diam(box) ≈ 1.0

set_variables(Float64, "x", order = 2, numvars = 2)
box = IntervalBox(-1..1,2)
x0 = IntervalBox(mid(box))
tm = TaylorModelN(1,1,x0,box)
a = 5.0
b = 2.0
@test muladd(tm,a,b) == a*tm + b
@test muladd(a,tm,b) == a*tm + b

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
search = IDQ.SignSearch(f, box, 1, 1e-2)
search.found_positive = true
search.found_negative = true
@test BranchAndPrune.process(search, box) == (:discard, box)

search = IDQ.SignSearch(f, box, 1, 1e-2)
search.breached_tolerance = true
@test BranchAndPrune.process(search,box) == (:discard, box)

f(x) = x + 2
box = IntervalBox(-1..1,1)
search = IDQ.SignSearch(f, box, 1, 1e-2)
@test BranchAndPrune.process(search,box) == (:store, box)
@test search.found_positive == true
@test search.found_negative == false
@test search.breached_tolerance == false

f(x) = x - 2
box = IntervalBox(-1..1,1)
search = IDQ.SignSearch(f, box, 1, 1e-2)
@test BranchAndPrune.process(search,box) == (:store, box)
@test search.found_positive == false
@test search.found_negative == true
@test search.breached_tolerance == false

f(x) = x
box = IntervalBox(-1..1,1)
search = IDQ.SignSearch(f, box, 1, 1e-2)
sbox = IntervalBox(0..1e-3,1)
@test BranchAndPrune.process(search,sbox) == (:discard, sbox)
@test search.found_positive == false
@test search.found_negative == false
@test search.breached_tolerance == true

f(x) = x
box = IntervalBox(-1..1,1)
search = IDQ.SignSearch(f, box, 1, 1e-2)
@test BranchAndPrune.process(search,box) == (:bisect, box)
@test search.found_positive == false
@test search.found_negative == false
@test search.breached_tolerance == false

box = IntervalBox(0..1,0..2)
b1,b2 = BranchAndPrune.bisect(search, box)
@test all(i -> b1[i] ≈ IntervalBox(0..1,0..1)[i], 1:2)
@test all(i -> b2[i] ≈ IntervalBox(0..1,1..2)[i], 1:2)

box = IntervalBox(-1..1,1)
f(x) = x
tree, search = IDQ.run_search(f, box, 1, 1e-2)
boxes = data(tree)
@test IntervalBox(0.5 .. 1,1) in boxes
@test IntervalBox(-1 .. -0.5,1) in boxes
@test search.found_positive == true
@test search.found_negative == true
@test search.breached_tolerance == false

@test sign(f, box, 1, 1e-2) == 0
f(x) = x*(x - 0.1)*(x - 1.0)
@test sign(f, box, 1, 1e-2) == 0

f(x) = (x - 0.5)*(x+0.5) + 10
@test sign(f, box, 1, 1e-2) == 1

box = IntervalBox(2 .. 3,1)
g(x) = (x - 2.5)*(x - 2.6)*(x - 2.9)
orders = 1:10
s = [sign(g,box,i,1e-2) for i in orders]
@test all(i -> s[i] == 0, 1:length(orders))

r = 0.5
f2(x,y) = x^2 + y^2 - r^2
box = IntervalBox(-1..1,2)
s = [sign(f2,box,i,1e-2) for i in orders]
@test all(i -> s[i] == 0, 1:length(orders))

g2(x,y) = sin(2pi*x)*cos(2pi*y) + 2
box = IntervalBox(-1..1,2)
s = [sign(g2,box,i,1e-2) for i in orders]
@test all(i -> s[i] == 1, 1:length(orders))

P = InterpolatingPolynomial(1,2,2)
coeffs = 1:9
IDQ.update!(P,coeffs)
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
IDQ.update!(P,coeffs)
box = IntervalBox(-1..1,2)
s = sign(P,box)
@test s == 0
