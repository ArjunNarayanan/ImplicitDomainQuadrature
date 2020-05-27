using Test
using StaticArrays
using FastGaussQuadrature
using IntervalArithmetic
# using Revise
using ImplicitDomainQuadrature

IDQ = ImplicitDomainQuadrature

function allequal(v1,v2)
    return all(v1 .≈ v2)
end

p = SMatrix{1,3}([+1.0 0.0 1.0])
w = SVector{3}([1/3,2/3,1/3])
@test_throws AssertionError IDQ.ReferenceQuadratureRule(p,w,-1.0,1.0)

p = SMatrix{1,3}([-1.0 0.0 1.0])
@test_throws AssertionError IDQ.ReferenceQuadratureRule(p,w,1.,1.)
@test_throws AssertionError IDQ.ReferenceQuadratureRule(p,w,-1.,1.)

float_type = typeof(1.0)
w = SVector{3}([0.5,1.0,0.5])
quad = IDQ.ReferenceQuadratureRule(p,w,-1.,1.)
@test typeof(quad) == IDQ.ReferenceQuadratureRule{3,float_type}
@test typeof(quad) <: IDQ.AbstractQuadratureRule{1,3}
@test allequal(quad.points,p)
@test allequal(quad.weights,w)
@test quad.lo ≈ -1.0
@test quad.hi ≈ +1.0

p = [1.0 0.0 1.0
     1.0 0.0 1.0]
w = [0.5,1.0,0.5]
@test_throws AssertionError IDQ.ReferenceQuadratureRule(p,w,-1.0,1.0)
p = [-1.0 0.0 1.0 2.0]
@test_throws AssertionError IDQ.ReferenceQuadratureRule(p,w,-1.0,1.0)
p = [-1.0 0.0 1.0]
quad = IDQ.ReferenceQuadratureRule(p,w,-1.0,1.0)
@test allequal(quad.points,p)
@test allequal(quad.weights,w)
@test quad.lo ≈ -1.0
@test quad.hi ≈ +1.0

p = [0.0,0.5,1.0]
w = [1/3,1/3,1/3]
quad = IDQ.ReferenceQuadratureRule(p,w,0.0,1.0)
@test allequal(quad.points,p')
@test allequal(quad.weights,w)
@test quad.lo ≈ 0.0
@test quad.hi ≈ 1.0

quad = IDQ.ReferenceQuadratureRule(4)
p,w = gausslegendre(4)
@test allequal(quad.points,p')
@test allequal(quad.weights,w)
@test quad.lo ≈ -1.0
@test quad.hi ≈ +1.0
@test typeof(quad) == IDQ.ReferenceQuadratureRule{4,float_type}
@test typeof(quad) <: IDQ.AbstractQuadratureRule{1,4}

function test_iteration(quad,points,weights)
    count = 1
    flag = true
    for (p,w) in quad
        flag = flag && allequal(p,points[count])
        flag = flag && w == weights[count]
        count += 1
    end
    return flag
end

@test test_iteration(quad,p,w)
qp,qw = quad[3]
@test allequal(qp,p[3])
@test qw == w[3]

@test IDQ.affine_map(0.0,-1.,1.,0.,1.) ≈ 0.5
@test IDQ.affine_map(0.75,0.,1.,5.,10.) ≈ 8.75
points = [-0.5 0.0 0.5]
@test allequal(IDQ.affine_map.(points,-1.,1.,0.,1.),[0.25 0.5 0.75])

@test IDQ.affine_map_derivative(1.0,-1.,1.,0.,1.) ≈ 0.5
@test IDQ.affine_map_derivative(1.0,-1.,1.,-5.,5.) ≈ 5.0
weights = [1.,2.,3.]
@test allequal(IDQ.affine_map_derivative.(weights,2.,3.,5.,10.),5weights)

points = [2. 3. 4.]
weights = [5.,7.,9.]
p,w = IDQ.transform(points,weights,1.,5.,-1.,1.)
@test allequal(p,[-0.5 0.0 0.5])
@test allequal(w,0.5weights)

quad = IDQ.ReferenceQuadratureRule(3)
p,w = IDQ.transform(quad,5.,10.)
@test allequal(p,5.0 .+ 2.5*(quad.points .+ 1))
@test allequal(w,2.5*quad.weights)

p,w = IDQ.transform(quad,3..4)
@test allequal(p,3 .+ 0.5*(quad.points .+ 1))
@test allequal(w,0.5*quad.weights)

p = SMatrix{4,3}(zeros(4,3))
w = SVector{3}([0.5,1.0,0.5])
@test_throws AssertionError IDQ.QuadratureRule(p,w)
p = SMatrix{2,4}([-1. -1 +1 +1
                  -1  +1 -1 +1])
w = SVector{5}([1.,1.,1.,1.,1.])
@test_throws AssertionError IDQ.QuadratureRule(p,w)

w = SVector{4}([1.,1.,1.,1.])
quad = IDQ.QuadratureRule(p,w)
@test allequal(quad.points,p)
@test allequal(quad.weights,w)

p = [-1. -1 +1 +1
     -1  +1 -1 +1
     -1  +1 -1 +1
     -1  +1 -1 +1]
w = [1.,1.,1.,1.]
@test_throws AssertionError IDQ.QuadratureRule(p,w)

p = [-1. -1 +1 +1
     -1  +1 -1 +1]
w = [1.,1.,1.,1.,1.]
@test_throws AssertionError IDQ.QuadratureRule(p,w)

w = [1.,1.,1.,1.]
quad = IDQ.QuadratureRule(p,w)
@test typeof(quad) == IDQ.QuadratureRule{2,4,float_type}
@test supertype(typeof(quad)) == IDQ.AbstractQuadratureRule{2,4}
@test allequal(quad.points,p)
@test allequal(quad.weights,w)

p1 = [1. 2.]
p2 = [3. 4.]
p = IDQ.tensor_product_points(p1,p2)
testp = [1. 1.  2.  2.
         3. 4.  3.  4.]
@test allequal(p,testp)

box = IntervalBox(-1..1,2)
quad = IDQ.ReferenceQuadratureRule(2)
tq = IDQ.tensor_product(quad,box)
p = quad.points
testp = [p[1]  p[1]  p[2]  p[2]
         p[1]  p[2]  p[1]  p[2]]
testw = [w[1]*w[1],w[1]*w[2],w[2]*w[1],w[2]*w[2]]
@test allequal(tq.points,testp)
@test allequal(tq.weights,testw)

box = IntervalBox(0..1,2..3)
tq = IDQ.tensor_product(quad,box)
p1 = 0.5*(quad.points .+ 1.)
p2 = 2. .+ 0.5*(quad.points .+ 1.)
w = 0.5*quad.weights
testp = [p1[1]  p1[1]  p1[2]  p1[2]
         p2[1]  p2[2]  p2[1]  p2[2]]
testw = kron(w,w)
@test allequal(tq.points,testp)
@test allequal(tq.weights,testw)

@test_throws AssertionError IDQ.tensor_product_quadrature(1,2)
@test_throws AssertionError IDQ.tensor_product_quadrature(3,2)
@test_throws AssertionError IDQ.tensor_product_quadrature(2,0)
@test_throws AssertionError IDQ.tensor_product_quadrature(2,-1)

tq = IDQ.tensor_product_quadrature(2,2)
quad = IDQ.ReferenceQuadratureRule(2)
p = quad.points
testp = [p[1]  p[1]  p[2]  p[2]
         p[1]  p[2]  p[1]  p[2]]
testw = kron(quad.weights,quad.weights)
@test allequal(tq.points,testp)
@test allequal(tq.weights,testw)
