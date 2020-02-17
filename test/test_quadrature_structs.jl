using ImplicitDomainQuadrature
using Test
using StaticArrays
using FastGaussQuadrature

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
    count = 1
    for (p,w) in quad
        @test points[count] ≈ p
        @test weights[count] ≈ w
        count += 1
    end
end

test_iteration(quad,p,w)

pq,wq = quad[3]
@test pq ≈ p[3]
@test wq ≈ w[3]
