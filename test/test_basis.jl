using ImplicitDomainQuadrature
using Test
import DynamicPolynomials

DynamicPolynomials.@polyvar x y

roots = [-1.0,0.0,1.0]
basis_funcs = ImplicitDomainQuadrature.lagrange_polynomials(x,roots)
@test_throws ArgumentError ImplicitDomainQuadrature.LagrangePolynomialBasis([basis_funcs[1]],roots)
@test_throws ArgumentError ImplicitDomainQuadrature.LagrangePolynomialBasis(basis_funcs,[roots[1]])
@test_throws ArgumentError ImplicitDomainQuadrature.LagrangePolynomialBasis([basis_funcs[1],basis_funcs[1],basis_funcs[1]],roots)
@test_throws ArgumentError ImplicitDomainQuadrature.LagrangePolynomialBasis([-y, 0.0, y] + basis_funcs, roots)
@test_throws ArgumentError ImplicitDomainQuadrature.LagrangePolynomialBasis([1.0-y^2, x^2, y^2 - x^2], roots)

basis = ImplicitDomainQuadrature.LagrangePolynomialBasis(basis_funcs,roots)
@test basis.funcs.polys[1].coefficients == [-0.5,0.5]
@test basis.funcs.polys[2].coefficients == [1.0,-1.0]
@test basis.funcs.polys[3].coefficients == [0.5,0.5]
@test all([basis.points[i] == roots[i] for i in 1:length(roots)])
