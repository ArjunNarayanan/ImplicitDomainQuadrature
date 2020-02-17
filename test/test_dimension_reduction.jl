using ImplicitDomainQuadrature
using Test
using IntervalRootFinding
using IntervalArithmetic

IDQ = ImplicitDomainQuadrature

r = roots(sin,0..2pi)
@test_throws ErrorException IDQ.checkUniqueRoots(r)

r = IDQ.unique_root_intervals(sin, pi/2, 5pi/2)
@test length(r) == 2
@test pi in r[1] || pi in r[2]
@test 2pi in r[1] || 2pi in r[2]

r = IDQ.unique_roots(cos,0.0,2pi)
@test length(r) == 2
@test pi/2 ≈ r[1] || pi/2 ≈ r[2]
@test 3pi/2 ≈ r[1] || 3pi/2 ≈ r[2]

f(x) = (x - 1.0)*(x - 1.5)
g(x) = (x - 0.5)*(x - 1.25)
r = IDQ.roots_and_ends([f,g], 0.0, 2.0)
@test r ≈ [0.0, 0.5, 1.0, 1.25, 1.5, 2.0]

@test IDQ.extend(1.0, 1, 2.0) ≈ [2.0,1.0]
@test IDQ.extend(1.0, 2, 2.0) ≈ [1.0,2.0]
@test_throws ArgumentError IDQ.extend(1.0,3,2.0)

@test IDQ.extend([1.0], 1, 2.0) ≈ [2.0,1.0]
@test_throws ArgumentError IDQ.extend([1.0,2.0], 1, 2.0)

x0 = [1.0]
x = [2.0 3.0 4.0 5.0
     6.0 7.0 8.0 9.0]
@test_throws ArgumentError IDQ.extend(x0,1,x)

x0 = [1.0]
x = [2.0 3.0 4.0 5.0]
@test IDQ.extend(x0, 1, x) ≈ [2.0 3.0 4.0 5.0
                              1.0 1.0 1.0 1.0]

x0 = [1.0]
x = [2.0 3.0 4.0 5.0]
@test IDQ.extend(x0, 2, x) ≈ [1.0 1.0 1.0 1.0
                              2.0 3.0 4.0 5.0]

x0 = [1.0]
x = [2.0 3.0 4.0 5.0]
@test_throws ArgumentError IDQ.extend(x0, 3, x)

f(x) = (x - 1.0)*(x - 2.0)*(x - 3.0)
g(x) = (x - 0.1)*(x - 0.2)*(x - 2.5)
@test IDQ.signConditionsSatisfied([f,g],0.5,[-1,-1])
@test IDQ.signConditionsSatisfied([f,g],1.5,[+1,-1])

quad1d = IDQ.ReferenceQuadratureRule(3)
@test_throws DimensionMismatch quadrature([f,g], [+1,-1,+1], 0.0, 3.5, quad1d)

quad = quadrature([f,g], [+1,-1], 0.0, 3.5, quad1d)
p,w = IDQ.transform(quad1d, 1.0, 2.0)
@test quad.points ≈ p
@test quad.weights ≈ w

quad = quadrature([f,g], [+1,+1], Interval(0.0,3.5), quad1d)
p,w = IDQ.transform(quad1d, 3.0, 3.5)
@test quad.points ≈ p
@test quad.weights ≈ w

f2(x) = f(x[1])
g2(x) = g(x[1])
p,w = quadrature([f2,g2], [+1,-1], 1, 0.0, 3.5, [1.0], 2.0, quad1d)
quad = quadrature([f,g], [+1,-1], 0.0, 3.5, quad1d)
@test p ≈ IDQ.extend([1.0],1,quad.points)
@test w ≈ 2quad.weights
