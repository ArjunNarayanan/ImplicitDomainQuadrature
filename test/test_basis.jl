using ImplicitDomainQuadrature
using Test
using StaticArrays
import DynamicPolynomials

IDQ = ImplicitDomainQuadrature

DynamicPolynomials.@polyvar x y

roots = [-1.0,0.0,1.0]
basis_funcs = IDQ.lagrange_polynomials(x,roots)
@test_throws ArgumentError IDQ.LagrangePolynomialBasis([basis_funcs[1]],roots)
@test_throws ArgumentError IDQ.LagrangePolynomialBasis(basis_funcs,[roots[1]])
@test_throws ArgumentError IDQ.LagrangePolynomialBasis([basis_funcs[1],basis_funcs[1],basis_funcs[1]],roots)
@test_throws ArgumentError IDQ.LagrangePolynomialBasis([-y, 0.0, y] + basis_funcs, roots)
@test_throws ArgumentError IDQ.LagrangePolynomialBasis([1.0-y^2, x^2, y^2 - x^2], roots)
@test_throws ArgumentError IDQ.LagrangePolynomialBasis([1.0 - x^3, x^2, x^3 - x^2], roots)

basis = IDQ.LagrangePolynomialBasis(basis_funcs,roots)
@test basis.funcs.polys[1].coefficients == [-0.5,0.5]
@test basis.funcs.polys[2].coefficients == [1.0,-1.0]
@test basis.funcs.polys[3].coefficients == [0.5,0.5]
@test all([basis.points[i] == roots[i] for i in 1:length(roots)])

big_basis = IDQ.LagrangePolynomialBasis(2, start = BigFloat(-1.0), stop = BigFloat(1.0))
@test typeof(big_basis.funcs.polys[1].coefficients) == Array{BigFloat,1}
@test typeof(big_basis.points) == SMatrix{1,3,BigFloat,3}

basis2 = IDQ.LagrangePolynomialBasis(2)
@test basis == basis2
@test IDQ.LagrangePolynomialBasis(3) != basis

@test IDQ.AbstractBasis1D{10} <: IDQ.AbstractBasis{10}
@test IDQ.AbstractTensorProductBasis{2,IDQ.LagrangePolynomialBasis{3},9} <: IDQ.AbstractBasis{9}
@test IDQ.TensorProductBasis{2,IDQ.LagrangePolynomialBasis{3},9} <: IDQ.AbstractTensorProductBasis{2,IDQ.LagrangePolynomialBasis{3},9}
@test_throws ArgumentError IDQ.TensorProductBasis(0,basis)
@test_throws ArgumentError IDQ.TensorProductBasis(4,basis)

tp1 = IDQ.TensorProductBasis(2,basis)
@test typeof(tp1) == IDQ.TensorProductBasis{2,IDQ.LagrangePolynomialBasis{3},9}

big_tp1 = IDQ.TensorProductBasis(2,big_basis)
@test typeof(big_tp1(0.5,0.2)) == SVector{9,BigFloat}

tp2 = IDQ.TensorProductBasis(2,2)
@test tp1 == tp2
tp3 = IDQ.TensorProductBasis(3,basis)
@test tp3 != tp2

v1 = @SVector [1.0,0.0,0.0]
v2 = @SVector [0.0,1.0,0.0]
v3 = @SVector [0.0,0.0,1.0]
@test basis(-1.0) ≈ v1
@test basis(0.0) ≈ v2
@test basis(1.0) ≈ v3
@test typeof(big_basis(-1.0)) == SVector{3,BigFloat}

d1 = @SMatrix [-1.5;2.0;-0.5]
d2 = @SMatrix [-0.5;0.0;0.5]
d3 = @SMatrix [0.5;-2.0;1.5]

@test IDQ.derivative(basis, -1.0) ≈ d1
@test IDQ.derivative(basis, 0.0) ≈ d2
@test IDQ.derivative(basis, 1.0) ≈ d3
@test typeof(IDQ.derivative(big_basis, -1.0)) == SMatrix{3,1,BigFloat,3}

v,d = IDQ.value_and_derivative(basis, -1.0)
@test v == v1
@test d == d1
v,d = IDQ.value_and_derivative(basis, 0.0)
@test v == v2
@test d == d2
v,d = IDQ.value_and_derivative(basis, 1.0)
@test v == v3
@test d == d3

tp1 = TensorProductBasis(1,basis)

@test tp1(-1.0) ≈ v1
@test tp1(0.0) ≈ v2
@test tp1(1.0) ≈ v3
@test tp1([1.0]) ≈ v3

@test gradient(tp1, -1.0) ≈ d1
@test gradient(tp1, 0.0) ≈ d2
@test gradient(tp1, 1.0) ≈ d3
@test gradient(tp1, [1.0]) ≈ d3
@test gradient(tp1, 1, [1.0]) ≈ d3

tp2 = TensorProductBasis(2,basis)
for i in 1:size(tp2.points)[2]
    vals = zeros(9)
    vals[i] = 1.0
    p = tp2.points[:,i]
    @test tp2(p[1],p[2]) ≈ vals
    @test tp2(p) ≈ vals
end

@test_throws BoundsError gradient(tp2, 3, -1.0, +1.0)
@test gradient(tp2, 1, -1.0, -1.0) ≈ kron(d1,v1)
@test gradient(tp2, 1, -1.0, +0.0) ≈ kron(d1,v2)
@test gradient(tp2, 1, -1.0, +1.0) ≈ kron(d1,v3)
@test gradient(tp2, 1, +0.0, -1.0) ≈ kron(d2,v1)
@test gradient(tp2, 1, +0.0, +0.0) ≈ kron(d2,v2)
@test gradient(tp2, 1, +0.0, +1.0) ≈ kron(d2,v3)
@test gradient(tp2, 1, +1.0, -1.0) ≈ kron(d3,v1)
@test gradient(tp2, 1, +1.0, +0.0) ≈ kron(d3,v2)
@test gradient(tp2, 1, +1.0, +1.0) ≈ kron(d3,v3)

@test gradient(tp2, 1, [-1.0, -1.0]) ≈ kron(d1,v1)
@test gradient(tp2, 1, [-1.0, +0.0]) ≈ kron(d1,v2)
@test gradient(tp2, 1, [-1.0, +1.0]) ≈ kron(d1,v3)
@test gradient(tp2, 1, [+0.0, -1.0]) ≈ kron(d2,v1)
@test gradient(tp2, 1, [+0.0, +0.0]) ≈ kron(d2,v2)
@test gradient(tp2, 1, [+0.0, +1.0]) ≈ kron(d2,v3)
@test gradient(tp2, 1, [+1.0, -1.0]) ≈ kron(d3,v1)
@test gradient(tp2, 1, [+1.0, +0.0]) ≈ kron(d3,v2)
@test gradient(tp2, 1, [+1.0, +1.0]) ≈ kron(d3,v3)

@test gradient(tp2, 2, -1.0, -1.0) ≈ kron(v1,d1)
@test gradient(tp2, 2, -1.0, +0.0) ≈ kron(v1,d2)
@test gradient(tp2, 2, -1.0, +1.0) ≈ kron(v1,d3)
@test gradient(tp2, 2, +0.0, -1.0) ≈ kron(v2,d1)
@test gradient(tp2, 2, +0.0, +0.0) ≈ kron(v2,d2)
@test gradient(tp2, 2, +0.0, +1.0) ≈ kron(v2,d3)
@test gradient(tp2, 2, +1.0, -1.0) ≈ kron(v3,d1)
@test gradient(tp2, 2, +1.0, +0.0) ≈ kron(v3,d2)
@test gradient(tp2, 2, +1.0, +1.0) ≈ kron(v3,d3)

@test gradient(tp2, 2, [-1.0, -1.0]) ≈ kron(v1,d1)
@test gradient(tp2, 2, [-1.0, +0.0]) ≈ kron(v1,d2)
@test gradient(tp2, 2, [-1.0, +1.0]) ≈ kron(v1,d3)
@test gradient(tp2, 2, [+0.0, -1.0]) ≈ kron(v2,d1)
@test gradient(tp2, 2, [+0.0, +0.0]) ≈ kron(v2,d2)
@test gradient(tp2, 2, [+0.0, +1.0]) ≈ kron(v2,d3)
@test gradient(tp2, 2, [+1.0, -1.0]) ≈ kron(v3,d1)
@test gradient(tp2, 2, [+1.0, +0.0]) ≈ kron(v3,d2)
@test gradient(tp2, 2, [+1.0, +1.0]) ≈ kron(v3,d3)

@test gradient(tp2, -1.0, -1.0) ≈ hcat(gradient(tp2, 1, -1.0, -1.0), gradient(tp2, 2, -1.0, -1.0))
@test gradient(tp2, -1.0, +0.0) ≈ hcat(gradient(tp2, 1, -1.0, +0.0), gradient(tp2, 2, -1.0, +0.0))
@test gradient(tp2, -1.0, +1.0) ≈ hcat(gradient(tp2, 1, -1.0, +1.0), gradient(tp2, 2, -1.0, +1.0))
@test gradient(tp2, +0.0, -1.0) ≈ hcat(gradient(tp2, 1, +0.0, -1.0), gradient(tp2, 2, +0.0, -1.0))
@test gradient(tp2, +0.0, +0.0) ≈ hcat(gradient(tp2, 1, +0.0, +0.0), gradient(tp2, 2, +0.0, +0.0))
@test gradient(tp2, +0.0, +1.0) ≈ hcat(gradient(tp2, 1, +0.0, +1.0), gradient(tp2, 2, +0.0, +1.0))
@test gradient(tp2, +1.0, -1.0) ≈ hcat(gradient(tp2, 1, +1.0, -1.0), gradient(tp2, 2, +1.0, -1.0))
@test gradient(tp2, +1.0, +0.0) ≈ hcat(gradient(tp2, 1, +1.0, +0.0), gradient(tp2, 2, +1.0, +0.0))
@test gradient(tp2, +1.0, +1.0) ≈ hcat(gradient(tp2, 1, +1.0, +1.0), gradient(tp2, 2, +1.0, +1.0))

tp3 = TensorProductBasis(3,basis)
for i in 1:size(tp3.points)[2]
    vals = zeros(27)
    vals[i] = 1.0
    p = tp3.points[:,i]
    @test tp3(p[1],p[2],p[3]) ≈ vals
    @test tp3(p) ≈ vals
end

@test gradient(tp3, 1, +1.0, -1.0, +0.0) ≈ kron(gradient(basis, +1.0), basis(-1.0), basis(+0.0))
@test gradient(tp3, 1, +0.0, -1.0, +0.0) ≈ kron(gradient(basis, +0.0), basis(-1.0), basis(+0.0))
@test gradient(tp3, 1, +1.0, -1.0, +1.0) ≈ kron(gradient(basis, +1.0), basis(-1.0), basis(+1.0))
@test_throws BoundsError gradient(tp3, 4, +1.0, -1.0, +0.0)
@test gradient(tp3, 0.3, 0.9, 0.1) ≈ hcat(gradient(tp3, 1, 0.3, 0.9, 0.1), gradient(tp3, 2, 0.3, 0.9, 0.1), gradient(tp3, 3, 0.3, 0.9, 0.1))
p = [0.15, 0.25, 0.35]
@test gradient(tp3, p) ≈ hcat(gradient(tp3, 1, p), gradient(tp3, 2, p), gradient(tp3, 3, p))
