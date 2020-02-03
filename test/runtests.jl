using SafeTestsets

@safetestset "Test Lagrange Polynomial construction" begin
    include("test_lagrange_polynomials.jl")
end

@safetestset "Test Basis and Tensor Product Basis" begin
    include("test_basis.jl")
end
