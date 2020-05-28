using SafeTestsets

@safetestset "Test Function Bounds" begin
    include("test_bounds.jl")
end

@safetestset "Test Quadrature Structs" begin
    include("test_quadrature_structs.jl")
end

@safetestset "Test Dimension Reduction Algorithm" begin
    include("test_dimension_reduction.jl")
end
