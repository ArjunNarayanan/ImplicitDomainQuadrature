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
w0 = 2.0
p,w = quadrature([f2,g2], [+1,-1], 1, 0.0, 3.5, [1.0], w0, quad1d)
quad = quadrature([f,g], [+1,-1], 0.0, 3.5, quad1d)
@test p ≈ IDQ.extend([1.0],1,quad.points)
@test w ≈ w0*quad.weights

p,w = quadrature([f2,g2], [+1,-1], 1, Interval(0.0,3.5), [1.0], w0, quad1d)
quad = quadrature([f,g], [+1,-1], Interval(0.0,3.5), quad1d)
@test p ≈ IDQ.extend([1.0],1,quad.points)
@test w ≈ w0*quad.weights

x0 = [1.0 2.0 3.0]
w0 = [3.0, 6.0, 9.0]
quad_ext = quadrature([f2,g2], [+1,-1], 1, 0.0, 3.5, x0, w0, quad1d)
quad = quadrature([f,g], [+1,-1], 0.0, 3.5, quad1d)
p = hcat([IDQ.extend([x0[i]], 1, quad.points) for i = 1:3]...)
w = vcat([quad.weights*w0[i] for i = 1:3]...)
@test p ≈ quad_ext.points
@test w ≈ quad_ext.weights

x0 = [1.0 2.0 3.0]
w0 = [3.0, 6.0, 9.0]
quad_ext = quadrature([f2,g2], [+1,-1], 1, Interval(0.0,3.5), x0, w0, quad1d)
quad = quadrature([f,g], [+1,-1], 0.0, 3.5, quad1d)
p = hcat([IDQ.extend([x0[i]], 1, quad.points) for i = 1:3]...)
w = vcat([quad.weights*w0[i] for i = 1:3]...)
@test p ≈ quad_ext.points
@test w ≈ quad_ext.weights

f2(x) = x[1]
P = InterpolatingPolynomial(1,2,2)
coeffs = [f2(P.basis.points[:,i]) for i in 1:size(P.basis.points)[2]]
update!(P,coeffs)
x0 = [1.0]
w0 = 3.0
p,w = IDQ.surface_quadrature(P,1,-1.0,1.0,x0,w0)
@test p ≈ [0.0,1.0]
@test w ≈ w0

p,w = IDQ.surface_quadrature(P,1,Interval(-1.0,1.0),x0,w0)
@test p ≈ [0.0,1.0]
@test w ≈ w0

f2(x) = 1.5 + x[1]
P = InterpolatingPolynomial(1,2,2)
coeffs = [f2(P.basis.points[:,i]) for i in 1:size(P.basis.points)[2]]
update!(P,coeffs)
x0 = [1.0]
w0 = 3.0
@test_throws ArgumentError IDQ.surface_quadrature(P,1,-1.0,1.0,x0,w0)

f2(x) = x[1]
P = InterpolatingPolynomial(1,2,2)
coeffs = [f2(P.basis.points[:,i]) for i in 1:size(P.basis.points)[2]]
update!(P,coeffs)
x0 = [-0.5 0.0 0.5]
w0 = [2.0,3.0,4.0]
surf_quad = IDQ.surface_quadrature(P,1,-1.0,1.0,x0,w0)
p = hcat([IDQ.extend(x0[i],1,0.0) for i in 1:3]...)
@test surf_quad.points ≈ p
@test surf_quad.weights ≈ w0

surf_quad = IDQ.surface_quadrature(P,1,Interval(-1.0,1.0),x0,w0)
@test surf_quad.points ≈ p
@test surf_quad.weights ≈ w0

f2(x) = x[2]
P = InterpolatingPolynomial(1,2,2)
coeffs = [f2(P.basis.points[:,i]) for i in 1:size(P.basis.points)[2]]
update!(P,coeffs)
box = IntervalBox(-1..1,2)
@test IDQ.height_direction(P,box) == 2

flag,s = IDQ.isSuitable(2,P,box)
@test flag == true
@test s == 1

f2(x) = (x[2] + 0.5)*(x[2] - 0.5)
P = InterpolatingPolynomial(1,2,2)
coeffs = [f2(P.basis.points[:,i]) for i in 1:size(P.basis.points)[2]]
update!(P,coeffs)
flag,s = IDQ.isSuitable(2,P,box)
@test flag == false

@test sign(1,1,true,-1) == -1
@test sign(1,-1,false,-1) == -1
@test sign(1,-1,false,1) == 0

f2(x) = x[2]
quad1d = IDQ.ReferenceQuadratureRule(5)
box = IntervalBox(-1..1,2)
P = InterpolatingPolynomial(1,2,2)
coeffs = [f2(P.basis.points[:,i]) for i in 1:size(P.basis.points)[2]]
update!(P,coeffs)
quad = quadrature(P,+1,false,box,quad1d)
p,w = IDQ.transform(quad1d, 0.0, 1.0)
p2 = hcat([IDQ.extend([quad1d.points[i]], 2, p) for i in 1:5]...)
w2 = vcat([quad1d.weights[i]*w for i in 1:5]...)
@test quad.points ≈ p2
@test quad.weights ≈ w2

f2(x) = x[2] + 1.5
quad1d = IDQ.ReferenceQuadratureRule(5)
box = IntervalBox(-1..1,2)
P = InterpolatingPolynomial(1,2,2)
coeffs = [f2(P.basis.points[:,i]) for i in 1:size(P.basis.points)[2]]
update!(P,coeffs)
quad = quadrature(P,-1,false,box,quad1d)
@test size(quad.points) == (2,0)
@test length(quad.weights) == 0

f2(x) = x[2] + 1.5
quad1d = IDQ.ReferenceQuadratureRule(5)
box = IntervalBox(-1..1,2)
P = InterpolatingPolynomial(1,2,2)
coeffs = [f2(P.basis.points[:,i]) for i in 1:size(P.basis.points)[2]]
update!(P,coeffs)
quad = quadrature(P,+1,false,box,quad1d)
p = hcat([IDQ.extend([quad1d.points[i]], 2, quad1d.points) for i = 1:5]...)
w = vcat([quad1d.weights[i]*quad1d.weights for i = 1:5]...)
@test quad.points ≈ p
@test quad.weights ≈ w

f2(x) = x[2]
quad1d = IDQ.ReferenceQuadratureRule(5)
box = IntervalBox(-1..1,2)
P = InterpolatingPolynomial(1,2,2)
coeffs = [f2(P.basis.points[:,i]) for i in 1:size(P.basis.points)[2]]
update!(P,coeffs)
quad = quadrature(P,+1,true,box,quad1d)
p = IDQ.extend([0.0], 1, quad1d.points)
@test p ≈ quad.points
@test quad1d.weights ≈ quad.weights

f2(x) = (x[2] - 0.5)*(x[2] + 0.75)
quad1d = IDQ.ReferenceQuadratureRule(5)
box = IntervalBox(-1..1,2)
P = InterpolatingPolynomial(1,2,2)
coeffs = [f2(P.basis.points[:,i]) for i in 1:size(P.basis.points)[2]]
update!(P,coeffs)
@test_throws MethodError quadrature(P,+1,true,box,quad1d)

f2(x) = (x[2] - 0.5)*(x[2] + 0.5)
quad1d = IDQ.ReferenceQuadratureRule(5)
box = IntervalBox(-1..1,2)
P = InterpolatingPolynomial(1,2,2)
coeffs = [f2(P.basis.points[:,i]) for i in 1:size(P.basis.points)[2]]
update!(P,coeffs)
@test_throws ArgumentError IDQ.height_direction(P,box)
