using ImplicitDomainQuadrature
using Test
import DynamicPolynomials: @polyvar

function test_basis_val_on_nodes(B::ImplicitDomainQuadrature.LagrangePolynomialBasis;
    atol = 1e-15)

    npts = length(B.points)
    flag = true
    for i in 1:npts
        p = B.points[i]
        vals = B(p)
        for j in 1:npts
            if i != j
                flag = flag && isapprox(vals[j], 0.0, atol = atol)
            else
                flag = flag && isapprox(vals[j], 1.0, atol = atol)
            end
        end
    end
    return flag
end

@polyvar x

max_order = 4
orders = range(1,stop=max_order)
basis = [ImplicitDomainQuadrature.LagrangePolynomialBasis(x,i) for i in orders]
@testset "Evaluate order 1 to $max_order Lagrange Polynomials on support points" begin
    @test test_basis_val_on_nodes(basis[1])
    @test test_basis_val_on_nodes(basis[2])
    @test test_basis_val_on_nodes(basis[3])
    @test test_basis_val_on_nodes(basis[4])
end
