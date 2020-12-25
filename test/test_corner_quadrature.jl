using Test
using LinearAlgebra
using PolynomialBasis
using IntervalArithmetic
using Revise
using ImplicitDomainQuadrature

IDQ = ImplicitDomainQuadrature

function corner_distance_function(x::V,xc) where {V<:AbstractVector}
    v = xc - x
    if all(x .<= xc)
        minimum(v)
    elseif all(x.> xc)
        return -sqrt(v'*v)
    elseif x[2] > xc[2]
        return v[2]
    else
        return v[1]
    end
end

function corner_distance_function(points::M,xc) where {M<:AbstractMatrix}
    return vec(mapslices(x->corner_distance_function(x,xc),points,dims=1))
end


corner = [0.,0.]
polyorder = 2
numqp = 5
poly = InterpolatingPolynomial(1,2,polyorder)
coeffs = corner_distance_function(poly.basis.points,corner)
update!(poly,coeffs .+ 1e-3)

quad = area_quadrature(poly,+1,[-1.,-1.],[1.,1.],numqp)
squad = surface_quadrature(poly,[-1.,-1.],[1.,1.],numqp)

using Plots
xrange = -1:1e-2:1
contour(xrange,xrange,(x,y)->poly(x,y),levels=[0.0],aspect_ratio=:equal)
scatter!(quad.points[1,:],quad.points[2,:])
scatter!(squad.points[1,:],squad.points[2,:])
