using ImplicitDomainQuadrature
using Test
using StaticArrays
import DynamicPolynomials

IDQ = ImplicitDomainQuadrature

P = IDQ.InterpolatingPolynomial

basis = IDQ.LagrangePolynomialBasis(2)
coeffs = @SMatrix [1.0 2.0 -1.0]
P = IDQ.InterpolatingPolynomial(coeffs, basis)
@test P(-1.0) ≈ coeffs[1]
@test P(0.0) ≈ coeffs[2]
@test P(1.0) ≈ coeffs[3]

P = IDQ.InterpolatingPolynomial(Float64, 1, basis)
@test P.coeffs ≈ @SMatrix [0.0 0.0 0.0]
update!(P, [1.0,1.0,1.0])
@test P(0.5) ≈ 1.0
@test gradient(P, 0.5) ≈ 0.0

big_basis = IDQ.LagrangePolynomialBasis(2, BigFloat(-1.0), BigFloat(1.0))
P = IDQ.InterpolatingPolynomial(BigFloat, 1, big_basis)
@test eltype(P.coeffs) == BigFloat

P = IDQ.InterpolatingPolynomial(1, basis)
@test eltype(P.coeffs) == Float64

P = IDQ.InterpolatingPolynomial(1, 2, 2, BigFloat(-1.0), BigFloat(1.0))
@test eltype(P.basis.basis.funcs.polys[1].coefficients) == BigFloat
