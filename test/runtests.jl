using SafeTestsets

# @safetestset "Test Lagrange Polynomial construction" begin
#     include("test_lagrange_polynomials.jl")
# end
#
# @safetestset "Test Basis and Tensor Product Basis" begin
#     include("test_basis.jl")
# end
#
# @safetestset "Test Interpolating Polynomials" begin
#     include("test_interpolation.jl")
# end

@safetestset "Test Function Bounds" begin
    include("test_bounds.jl")
end

@safetestset "Test Quadrature Structs" begin
    include("test_new_quadrature_structs.jl")
end

@safetestset "Test Dimension Reduction Algorithm" begin
    include("test_new_dimension_reduction.jl")
end
