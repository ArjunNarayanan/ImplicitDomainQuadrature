using ProfileView
using BenchmarkTools
using PolynomialBasis
using ImplicitDomainQuadrature

IDQ = ImplicitDomainQuadrature

function plane_distance_function(coords, normal, x0)
    return (coords .- x0)' * normal
end


x0 = [0.0,0.0]
normal = [1.0,0.0]
polyorder = 3
numqp = 5
levelset = InterpolatingPolynomial(1,2,polyorder)
levelsetcoeffs = plane_distance_function(levelset.basis.points,normal,x0)
update!(levelset,levelsetcoeffs)

# @profiler run_quadrature_iterations(levelset,numqp,100)
