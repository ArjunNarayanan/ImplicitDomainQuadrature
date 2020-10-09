using Test
using PolynomialBasis
using Roots
using IntervalArithmetic
using Revise
using ImplicitDomainQuadrature

IDQ = ImplicitDomainQuadrature

function circle_distance_function(coords, center, radius)
    difference = (coords .- center).^2
    distance = radius .- mapslices(sum,difference,dims=1)'
    return distance
end

numqp = 4
polyorder = 2
poly = InterpolatingPolynomial(1,2,polyorder)

center = [3.,0.5]
radius = 1.5
coords = [0.0  0.0  0.0  1.5  1.5  1.5  3.0  3.0  3.0
          0.0  0.5  1.0  0.0  0.5  1.0  0.0  0.5  1.0]
coeffs = circle_distance_function(coords,center,radius)
update!(poly,coeffs)

interpgrad = IDQ.interpolating_gradient(poly)

grad(x) = gradient(poly,x)

box = IntervalBox(-1..1,2)



# quad1d = ReferenceQuadratureRule(numqp)
# pquad = area_quadrature(poly,+1,box,quad1d)
