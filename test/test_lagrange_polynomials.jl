using ImplicitDomainQuadrature
using Test
import DynamicPolynomials

DynamicPolynomials.@polyvar x
@test x - 1.0 == ImplicitDomainQuadrature.polynomial_from_roots(x,[-1.0,1.0],1)
@test x^2 - 1.0 ==  ImplicitDomainQuadrature.polynomial_from_roots(x,[-1.0,0.0,1.0],2)
@test DynamicPolynomials.Polynomial{true,BigFloat} == typeof(ImplicitDomainQuadrature.polynomial_from_roots(x,BigFloat.([-1.0,0.0,1.0]),2))

@test -1.0 == ImplicitDomainQuadrature.normalization([-1.0,0.0,1.0],2)
@test typeof(ImplicitDomainQuadrature.normalization(BigFloat[-1.0,0.0,1.0],2)) == BigFloat
@test 0.5x^2 - 0.5x == ImplicitDomainQuadrature.lagrange_polynomial(x,[-1.0,0.0,1.0],1)

basis = ImplicitDomainQuadrature.lagrange_polynomials(x,[-1.0,0.0,1.0])
@test basis[1] == 0.5x^2 - 0.5x
@test basis[2] == -x^2 + 1.0
@test basis[3] == 0.5x^2 + 0.5x

@test Array{DynamicPolynomials.Polynomial{true,BigFloat},1} == typeof(ImplicitDomainQuadrature.lagrange_polynomials(x,BigFloat.([-1.0,0.0,1.0])))
