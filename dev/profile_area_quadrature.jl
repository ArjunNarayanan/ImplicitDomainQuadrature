using BenchmarkTools
using PolynomialBasis
using ImplicitDomainQuadrature

IDQ = ImplicitDomainQuadrature

function plane_distance_function(coords, normal, x0)
    return (coords .- x0)' * normal
end

function run_quadrature_iterations(levelset,numqp,niter)
    for i = 1:niter
        pquad = IDQ.area_quadrature(levelset,+1,[-1.,-1.],[1.,1.],numqp)
    end
end

x0 = [0.0,0.0]
normal = [1.0,0.0]
polyorder = 2
numqp = 5
levelset = InterpolatingPolynomial(1,2,polyorder)
levelsetcoeffs = plane_distance_function(levelset.basis.points,normal,x0)
update!(levelset,levelsetcoeffs)

# @profiler run_quadrature_iterations(levelset,numqp,100)
