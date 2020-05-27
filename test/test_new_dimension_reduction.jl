using Test
using IntervalRootFinding
using IntervalArithmetic
using Revise
using ImplicitDomainQuadrature

IDQ = ImplicitDomainQuadrature

function allequal(v1,v2)
    return all(v1 .≈ v2)
end

r = roots(sin,0..2pi)
@test_throws AssertionError IDQ.check_unique_roots(r)

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
@test allequal(r,[0.0, 0.5, 1.0, 1.25, 1.5, 2.0])

@test IDQ.extend(1.0, 1, 2.0) ≈ [2.0,1.0]
@test IDQ.extend(1.0, 2, 2.0) ≈ [1.0,2.0]
@test_throws ArgumentError IDQ.extend(1.0,3,2.0)

@test allequal(IDQ.extend([1.0], 1, 2.0), [2.0,1.0])
@test_throws ArgumentError IDQ.extend([1.0,2.0], 1, 2.0)

x0 = [1.0]
x = [2.0 3.0 4.0 5.0
     6.0 7.0 8.0 9.0]
@test_throws ArgumentError IDQ.extend(x0,1,x)

x0 = [1.0]
x = [2.0 3.0 4.0 5.0]
testx = [2.0 3.0 4.0 5.0
         1.0 1.0 1.0 1.0]
@test allequal(IDQ.extend(x0, 1, x),testx)

x0 = [1.0]
x = [2.0 3.0 4.0 5.0]
testx = [1.0 1.0 1.0 1.0
         2.0 3.0 4.0 5.0]
@test allequal(IDQ.extend(x0, 2, x),testx)

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
@test allequal(quad.points,p)
@test allequal(quad.weights,w)

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
@test allequal(quad.points,p)
@test allequal(quad.weights,w)
