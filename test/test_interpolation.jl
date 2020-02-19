using ImplicitDomainQuadrature
using Test
using StaticArrays
import DynamicPolynomials

IDQ = ImplicitDomainQuadrature

P = IDQ.InterpolatingPolynomial

basis = IDQ.LagrangePolynomialBasis(2)
coeffs = @SMatrix [1.0 2.0 -1.0]
big_basis = IDQ.LagrangePolynomialBasis(2, start = BigFloat(-1.0), stop = BigFloat(1.0))
@test_logs (:warn, "Coefficient type and polynomial type are not the same") IDQ.InterpolatingPolynomial(coeffs, big_basis)

P = IDQ.InterpolatingPolynomial(coeffs, basis)
@test P(-1.0) ≈ coeffs[1]
@test P(0.0) ≈ coeffs[2]
@test P(1.0) ≈ coeffs[3]

P = IDQ.InterpolatingPolynomial(Float64, 1, basis)
@test P.coeffs ≈ @SMatrix [0.0 0.0 0.0]
update!(P, [1.0,1.0,1.0])
@test P(0.5) ≈ 1.0
@test gradient(P, 0.5) ≈ 0.0

big_basis = IDQ.LagrangePolynomialBasis(2, start = BigFloat(-1.0), stop = BigFloat(1.0))
P = IDQ.InterpolatingPolynomial(BigFloat, 1, big_basis)
@test eltype(P.coeffs) == BigFloat

P = IDQ.InterpolatingPolynomial(1, basis)
@test eltype(P.coeffs) == Float64

P = IDQ.InterpolatingPolynomial(1, 2, 2, BigFloat(-1.0), BigFloat(1.0))
@test eltype(P.basis.basis.funcs.polys[1].coefficients) == BigFloat

P = IDQ.InterpolatingPolynomial(1, 2, 2)
@test eltype(P.basis.basis.funcs.polys[1].coefficients) == Float64

IDQ.update!(P, repeat([1.0,2.0,3.0],3))
@test P(0.0,-1.0) ≈ 1.0
@test P(1.0,0.0) ≈ 2.0
@test P(-1.0,1.0) ≈ 3.0

IDQ.update!(P, [1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 3.0, 3.0, 3.0])
@test P(-1.0, 0.0) ≈ 1.0
@test P(0.0,-1.0) ≈ 2.0
@test P(1.0,1.0) ≈ 3.0

P = IDQ.InterpolatingPolynomial(2,2,2)
coeffs = [1.0 1.0 1.0 2.0 2.0 2.0 3.0 3.0 3.0
          5.0 5.0 5.0 7.0 7.0 7.0 9.0 9.0 9.0]
IDQ.update!(P, coeffs)
@test P(-1.0,0.0) ≈ [1.0,5.0]
@test P(0.0,-1.0) ≈ [2.0,7.0]
@test P(1.0,1.0) ≈ [3.0,9.0]

P = IDQ.InterpolatingPolynomial(1,1,2)
f(x) = 3x^2 + 2x - 5
df(x) = 6x + 2
coeffs = f.(P.basis.points)
update!(P, coeffs)
@test gradient(P, -1.0) ≈ df(-1.0)
@test gradient(P, 0.19) ≈ df(0.19)

P = IDQ.InterpolatingPolynomial(1,2,2)
f(x,y) = 3x^2*y + 2y^2 + 2x*y
dfx(x,y) = 6*x*y + 2*y
dfy(x,y) = 3x^2 + 4y + 2x
coeffs = [f((P.basis.points[:,i])...) for i = 1:9]
update!(P, coeffs)
@test gradient(P, 0.1, 0.2) ≈ [dfx(0.1, 0.2) dfy(0.1,0.2)]
@test gradient(P, 1, 0.7, -0.3) ≈ dfx(0.7, -0.3)
@test gradient(P, 2, [-0.5, 0.75]) ≈ dfy(-0.5, 0.75)
@test gradient(P, [0.1, 0.45]) ≈ [dfx(0.1, 0.45) dfy(0.1, 0.45)]

P = IDQ.InterpolatingPolynomial(2,2,2)
f(x,y) = [7x^2 + 2*x*y^2 + 3*x,
          -6y^2 + 12*x^2*y + 10*y]
dfx(x,y) = [14x + 2y^2 + 3,
            24x*y]
dfy(x,y) = [4x*y,
            -12y + 12x^2 + 10]
coeffs = hcat([f(P.basis.points[:,i]...) for i in 1:9]...)
update!(P, coeffs)
@test gradient(P, 0.1, 0.5) ≈ hcat(dfx(0.1,0.5), dfy(0.1,0.5))
@test gradient(P, [0.9, -0.7]) ≈ hcat(dfx(0.9, -0.7), dfy(0.9, -0.7))
@test vec(gradient(P, 1, 0.7, 0.3)) ≈ dfx(0.7, 0.3)
