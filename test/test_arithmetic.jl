using Test
using IntervalArithmetic
using TaylorModels
# using Revise
using ImplicitDomainQuadrature

set_variables(Float64, "x", order = 2, numvars = 2)
box = IntervalBox(-1..1,2)
x0 = IntervalBox(mid(box))
tm = TaylorModelN(1,1,x0,box)
a = 5.0
b = 2.0
@test muladd(tm,a,b) == a*tm + b
@test muladd(a,tm,b) == a*tm + b
@test muladd(tm,tm,b) == tm*tm + b
@test muladd(a,b,tm) == a*b+tm
tm1 = TaylorModelN(1,1,x0,box)
tm2 = TaylorModelN(1,1,x0,box)
tm3 = TaylorModelN(1,1,x0,box)
@test muladd(tm1,a,tm2) == a*tm1 + tm2
@test muladd(tm1,tm2,tm3) == tm1*tm2 + tm3
