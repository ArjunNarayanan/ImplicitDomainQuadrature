using Test
using PolynomialBasis
using IntervalRootFinding
using IntervalArithmetic
# using Revise
using ImplicitDomainQuadrature

IDQ = ImplicitDomainQuadrature

function allapprox(v1,v2;tol=1e-14)
    @assert length(v1) == length(v2)
    flags = [isapprox(v1[i],v2[i],atol=tol) for i = 1:length(v1)]
    return all(flags)
end


rootint = IDQ.unique_root_intervals(sin,0.0,2pi)
r = IDQ.unique_roots(sin,0.,2pi)
# @test_throws AssertionError IDQ.unique_root_intervals(sin,0.0,2pi)

r = sort!(IDQ.unique_root_intervals(sin, pi/2, 5pi/2))
@test length(r) == 2
@test pi in r[1]
@test 2pi in r[2]

r = sort!(IDQ.unique_roots(cos,0.0,2pi))
@test length(r) == 2
@test pi/2 ≈ r[1]
@test 3pi/2 ≈ r[2]

f(x) = (x - 1.0)*(x - 1.5)
g(x) = (x - 0.5)*(x - 1.25)
r = IDQ.roots_and_ends([f,g], 0.0, 2.0)
testr = [0.0, 0.5, 1.0, 1.25, 1.5, 2.0]
@test allapprox(r,testr)

@test IDQ.extend(1.0, 1, 2.0) ≈ [2.0,1.0]
@test IDQ.extend(1.0, 2, 2.0) ≈ [1.0,2.0]
@test_throws ArgumentError IDQ.extend(1.0,3,2.0)

@test allapprox(IDQ.extend([1.0], 1, 2.0), [2.0,1.0])
@test_throws ArgumentError IDQ.extend([1.0,2.0], 1, 2.0)

x0 = [1.0]
x = [2.0 3.0 4.0 5.0
     6.0 7.0 8.0 9.0]
@test_throws ArgumentError IDQ.extend(x0,1,x)

x0 = [1.0]
x = [2.0 3.0 4.0 5.0]
testx = [2.0 3.0 4.0 5.0
         1.0 1.0 1.0 1.0]
@test allapprox(IDQ.extend(x0, 1, x),testx)

x0 = [1.0]
x = [2.0 3.0 4.0 5.0]
testx = [1.0 1.0 1.0 1.0
         2.0 3.0 4.0 5.0]
@test allapprox(IDQ.extend(x0, 2, x),testx)

x0 = [1.0]
x = [2.0 3.0 4.0 5.0]
@test_throws ArgumentError IDQ.extend(x0, 3, x)

f(x) = (x - 1.0)*(x - 2.0)*(x - 3.0)
g(x) = (x - 0.1)*(x - 0.2)*(x - 2.5)
@test IDQ.sign_conditions_satisfied([f,g],0.5,[-1,-1])
@test IDQ.sign_conditions_satisfied([f,g],1.5,[+1,-1])

p = [1. 2. 3.]
w = [1.,2.]
@test_throws AssertionError IDQ.TemporaryQuadrature(p,w)

float_type = typeof(1.0)
p = [-1. 1.]
w = [0.5,0.5]
quad = IDQ.TemporaryQuadrature(p,w)
@test typeof(quad) == IDQ.TemporaryQuadrature{float_type}
@test allapprox(quad.points,p)
@test allapprox(quad.weights,w)

@test_throws AssertionError IDQ.TemporaryQuadrature(float_type,0)
@test_throws AssertionError IDQ.TemporaryQuadrature(float_type,4)
quad = IDQ.TemporaryQuadrature(float_type,2)
@test size(quad.points) == (2,0)
@test length(quad.weights) == 0

p = [0.0,1.0]
w = [1.0,2.0]
@test_throws AssertionError IDQ.update!(quad,p,w)
p = [0.0]
w = [1.0]
@test_throws AssertionError IDQ.update!(quad,p,w)

p = [0.,1.]
w = [1.]
IDQ.update!(quad,p,w)
@test allapprox(quad.points,p)
@test allapprox(quad.weights,w)

quad1d = IDQ.ReferenceQuadratureRule(3)
@test_throws AssertionError IDQ.quadrature([f,g], [+1,-1,+1], 0.0, 3.5, quad1d)

quad = IDQ.quadrature([f,g], [+1,-1], 0.0, 3.5, quad1d)
p,w = IDQ.transform(quad1d, 1.0, 2.0)
@test allapprox(quad.points,p)
@test allapprox(quad.weights,w)

qr = IDQ.QuadratureRule(quad)
@test allapprox(qr.points,p)
@test allapprox(qr.weights,w)
@test typeof(qr) == IDQ.QuadratureRule{1,3,typeof(1.0)}

quad = IDQ.quadrature([f,g], [+1,+1], Interval(0.0,3.5), quad1d)
p,w = IDQ.transform(quad1d, 3.0, 3.5)
@test allapprox(quad.points, p)
@test allapprox(quad.weights, w)

f2(x) = f(x[1])
g2(x) = g(x[1])
w0 = 2.0
@test_throws AssertionError IDQ.quadrature([f2,g2], [+1], 1, 0.0, 3.5, [1.0], w0, quad1d)
quad1 = IDQ.quadrature([f2,g2], [+1,-1], 1, 0.0, 3.5, [1.0], w0, quad1d)
quad2 = IDQ.quadrature([f,g], [+1,-1], 0.0, 3.5, quad1d)
@test allapprox(quad1.points,IDQ.extend([1.0],1,quad2.points))
@test allapprox(quad1.weights,w0*quad2.weights)

quad1 = IDQ.quadrature([f2,g2], [+1,-1], 1, Interval(0.0,3.5), [1.0], w0, quad1d)
quad2 = IDQ.quadrature([f,g], [+1,-1], Interval(0.0,3.5), quad1d)
@test allapprox(quad1.points,IDQ.extend([1.0],1,quad2.points))
@test allapprox(quad1.weights, w0*quad2.weights)

x0 = [1.0 2.0 3.0]
w0 = [3.0, 6.0, 9.0, 10.0]
@test_throws AssertionError IDQ.quadrature([f2,g2],[+1,-1,+1],1,0.0,3.5,x0,w0,quad1d)
@test_throws AssertionError IDQ.quadrature([f2,g2],[+1,-1],1,0.0,3.5,x0,w0,quad1d)

w0 = [3.0, 6.0, 9.0]
quad_ext = IDQ.quadrature([f2,g2], [+1,-1], 1, 0.0, 3.5, x0, w0, quad1d)
quad = IDQ.quadrature([f,g], [+1,-1], 0.0, 3.5, quad1d)
testp = hcat([IDQ.extend([x0[i]], 1, quad.points) for i = 1:3]...)
testw = vcat([quad.weights*w0[i] for i = 1:3]...)
@test allapprox(quad_ext.points,testp)
@test allapprox(quad_ext.weights,testw)

quad_ext = IDQ.quadrature([f2,g2],[+1,-1],1,Interval(0.0,3.5),x0,w0,quad1d)
@test allapprox(quad_ext.points,testp)
@test allapprox(quad_ext.weights,testw)

f2(x) = 1.5 + x[1]
P = InterpolatingPolynomial(1,2,2)
coeffs = [f2(P.basis.points[:,i]) for i in 1:size(P.basis.points)[2]]
update!(P,coeffs)
x0 = [1.0]
w0 = 3.0
@test_throws ArgumentError IDQ.surface_quadrature(P,1,-1.0,1.0,x0,w0)

f2(x) = x[1]-0.5
P = InterpolatingPolynomial(1,2,2)
coeffs = [f2(P.basis.points[:,i]) for i in 1:size(P.basis.points)[2]]
update!(P,coeffs)
x0 = [1.0]
w0 = 3.0
p,w = IDQ.surface_quadrature(P,1,-1.0,1.0,x0,w0)
@test allapprox(p,[0.5,1.0])
@test w ≈ w0

f2(x) = x[1]
P = InterpolatingPolynomial(1,2,2)
coeffs = [f2(P.basis.points[:,i]) for i in 1:size(P.basis.points)[2]]
update!(P,coeffs)
x0 = [-0.5 0.0 0.5]
w0 = [2.0,3.0,4.0]
@test_throws AssertionError IDQ.surface_quadrature(P,1,-1.0,1.0,x0,[2.,3.,4.,5.])
surf_quad = IDQ.surface_quadrature(P,1,-1.0,1.0,x0,w0)
p = hcat([IDQ.extend(x0[i],1,0.0) for i in 1:3]...)
@test allapprox(surf_quad.points,p)
@test allapprox(surf_quad.weights,w0)

surf_quad = IDQ.surface_quadrature(P,1,Interval(-1.0,1.0),x0,w0)
@test allapprox(surf_quad.points,p)
@test allapprox(surf_quad.weights,w0)

f2(x) = x[2]
P = InterpolatingPolynomial(1,2,2)
coeffs = [f2(P.basis.points[:,i]) for i in 1:size(P.basis.points)[2]]
update!(P,coeffs)
@test IDQ.height_direction(P,[0.0,0.0]) == 2
box = IntervalBox(-1..1,2)
@test IDQ.height_direction(P,box) == 2

flag,s = IDQ.is_suitable(2,P,box)
@test flag == true
@test s == 1

f2(x) = (x[2] + 0.5)*(x[2] - 0.5)
P = InterpolatingPolynomial(1,2,2)
coeffs = [f2(P.basis.points[:,i]) for i in 1:size(P.basis.points)[2]]
update!(P,coeffs)
flag,s = IDQ.is_suitable(2,P,box)
@test flag == false

@test IDQ.sign(1,1,true,-1) == -1
@test IDQ.sign(1,-1,false,-1) == -1
@test IDQ.sign(1,-1,false,1) == 0

f2(x) = x[2]
quad1d = IDQ.ReferenceQuadratureRule(5)
box = IntervalBox(-1..1,2)
P = InterpolatingPolynomial(1,2,2)
coeffs = [f2(P.basis.points[:,i]) for i in 1:size(P.basis.points)[2]]
update!(P,coeffs)
quad = IDQ.quadrature(P,+1,false,box,quad1d)
p,w = IDQ.transform(quad1d, 0.0, 1.0)
p2 = hcat([IDQ.extend([quad1d.points[i]], 2, p) for i in 1:5]...)
w2 = vcat([quad1d.weights[i]*w for i in 1:5]...)
@test allapprox(quad.points,p2)
@test allapprox(quad.weights,w2)

f2(x) = x[2] + 1.5
quad1d = IDQ.ReferenceQuadratureRule(5)
box = IntervalBox(-1..1,2)
P = InterpolatingPolynomial(1,2,2)
coeffs = [f2(P.basis.points[:,i]) for i in 1:size(P.basis.points)[2]]
update!(P,coeffs)
quad = IDQ.quadrature(P,-1,false,box,quad1d)
@test size(quad.points) == (2,0)
@test length(quad.weights) == 0

f2(x) = x[2] + 1.5
quad1d = IDQ.ReferenceQuadratureRule(5)
box = IntervalBox(-1..1,2)
P = InterpolatingPolynomial(1,2,2)
coeffs = [f2(P.basis.points[:,i]) for i in 1:size(P.basis.points)[2]]
update!(P,coeffs)
quad = IDQ.quadrature(P,+1,false,box,quad1d)
p = hcat([IDQ.extend([quad1d.points[i]], 2, quad1d.points) for i = 1:5]...)
w = vcat([quad1d.weights[i]*quad1d.weights for i = 1:5]...)
@test allapprox(quad.points,p)
@test allapprox(quad.weights,w)

f2(x) = x[2]
quad1d = IDQ.ReferenceQuadratureRule(5)
box = IntervalBox(-1..1,2)
P = InterpolatingPolynomial(1,2,2)
coeffs = [f2(P.basis.points[:,i]) for i in 1:size(P.basis.points)[2]]
update!(P,coeffs)
quad = IDQ.quadrature(P,+1,true,box,quad1d)
p = IDQ.extend([0.0], 1, quad1d.points)
@test allapprox(p,quad.points)
@test allapprox(quad1d.weights,quad.weights)

f2(x) = (x[2] - 0.5)*(x[2] + 0.75)
quad1d = IDQ.ReferenceQuadratureRule(5)
box = IntervalBox(-1..1,2)
P = InterpolatingPolynomial(1,2,2)
coeffs = [f2(P.basis.points[:,i]) for i in 1:size(P.basis.points)[2]]
update!(P,coeffs)
@test_throws AssertionError IDQ.quadrature(P,+1,true,box,quad1d)

f2(x) = (x[2] - 0.5)*(x[2] + 0.5)
quad1d = IDQ.ReferenceQuadratureRule(5)
box = IntervalBox(-1..1,2)
P = InterpolatingPolynomial(1,2,2)
coeffs = [f2(P.basis.points[:,i]) for i in 1:size(P.basis.points)[2]]
update!(P,coeffs)
@test_throws AssertionError IDQ.height_direction(P,box)
