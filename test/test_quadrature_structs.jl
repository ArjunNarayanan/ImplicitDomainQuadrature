using ImplicitDomainQuadrature
using Test
using StaticArrays
using FastGaussQuadrature
using IntervalArithmetic

IDQ = ImplicitDomainQuadrature

@test_throws DimensionMismatch IDQ.checkNumPointsWeights(2,3)

p = [1.0 3.0 5.0
     2.0 4.0 6.0]
w = [1.0,2.0,1.0]
@test_throws ArgumentError IDQ.ReferenceQuadratureRule(p,w)
p = [1.0,2.0,1.0]
w = [1.0,2.0,3.0,4.0]
@test_throws DimensionMismatch IDQ.ReferenceQuadratureRule(p,w)

p = [0.5 1.0 2.0]
w = [0.5,1.0,0.5]
@test_throws DomainError IDQ.ReferenceQuadratureRule(p,w)

p = [-0.5 0.0 0.5]
w = [0.5, 1.0, 0.5]
quad = IDQ.ReferenceQuadratureRule(p,w)
@test quad.points ≈ p
@test quad.weights ≈ w

p = [-0.5, 0.0, 0.5]
w = [0.5, 1.0, 0.5]
quad = IDQ.ReferenceQuadratureRule(p,w)
@test quad.points ≈ p'
@test quad.weights ≈ w

quad = IDQ.ReferenceQuadratureRule(5)
p,w = gausslegendre(5)
@test quad.points ≈ p'
@test quad.weights ≈ w

function test_iteration(quad,points,weights)
    flag = true
    count = 1
    for (p,w) in quad
        flag = flag && points[count] ≈ p
        flag = flag && weights[count] ≈ w
        count += 1
    end
    return flag
end

@test test_iteration(quad,p,w)

pq,wq = quad[3]
@test pq ≈ p[3]
@test wq ≈ w[3]

@test_throws ArgumentError IDQ.quadrature_transformers(2.0,1.0)
@test_logs (:warn, "Transforming quadrature rule to a null domain")
a = 1.0; b = 2.0
scale, middle = IDQ.quadrature_transformers(a,b)
@test scale ≈ 0.5(b - a)
@test middle ≈ 0.5*(b+a)

quad = IDQ.ReferenceQuadratureRule(3)
p,w = IDQ.transform(quad, 0.0, 1.0)
scale = 0.5
middle = 0.5
@test p ≈ quad.points*scale .+ middle
@test w ≈ quad.weights*scale

quad = IDQ.ReferenceQuadratureRule(3)
p,w = IDQ.transform(quad, 0..1)
scale = 0.5
middle = 0.5
@test p ≈ quad.points*scale .+ middle
@test w ≈ quad.weights*scale

points = Array(reshape(1.0:40.0,4,10))
weights = collect(1.0:10.0)
@test_throws ArgumentError QuadratureRule(points, weights)
weights = collect(1.0:20.0)
points = Array(reshape(1.0:30.0,3,10))
@test_throws DimensionMismatch QuadratureRule(points, weights)

quad = QuadratureRule(Float64, 2)
@test typeof(quad) == QuadratureRule{2,Float64}
@test typeof(quad.points) == Matrix{Float64}
@test typeof(quad.weights) == Vector{Float64}
@test size(quad.points) == (2,0)
@test length(quad.weights) == 0

quad = QuadratureRule(BigFloat, 2)
@test typeof(quad) == QuadratureRule{2,BigFloat}
@test typeof(quad.points) == Matrix{BigFloat}
@test typeof(quad.weights) == Vector{BigFloat}
@test size(quad.points) == (2,0)
@test length(quad.weights) == 0

quad = QuadratureRule(2)
@test typeof(quad) == QuadratureRule{2,Float64}
@test typeof(quad.points) == Matrix{Float64}
@test typeof(quad.weights) == Vector{Float64}
@test size(quad.points) == (2,0)
@test length(quad.weights) == 0

quad = QuadratureRule(3)
@test typeof(quad) == QuadratureRule{3,Float64}
@test typeof(quad.points) == Matrix{Float64}
@test typeof(quad.weights) == Vector{Float64}
@test size(quad.points) == (3,0)
@test length(quad.weights) == 0

function test_multidim_iteration(quad,points,weights)
    flag = true
    dim,npoints = size(points)
    count = 1
    for (p,w) in quad
        flag = flag && p ≈ points[:,count]
        flag = flag && w ≈ weights[count]
        count += 1
    end
    return flag
end

points = Array(reshape(1.0:30.0,3,10))
weights = Array(0.1:0.1:1.0)
quad = QuadratureRule(points, weights)
@test test_multidim_iteration(quad,points,weights)
p,w = quad[3]
@test p ≈ points[:,3]
@test w ≈ weights[3]

quad = QuadratureRule(2)
points = Array(reshape(1.0:20.0,2,10))
weights = Array(1.0:10.0)
update!(quad,points,weights)
@test quad.points ≈ points
@test quad.weights ≈ weights
more_points = Array(reshape(31.0:50.0,2,10))
more_weights = Array(11.0:20.0)
update!(quad,more_points,more_weights)
@test quad.points ≈ hcat(points,more_points)
@test quad.weights ≈ vcat(weights,more_weights)

more_points = Array(reshape(1.0:40.0,4,10))
more_weights = Array(1.0:10.0)
@test_throws DimensionMismatch update!(quad,more_points,more_weights)
