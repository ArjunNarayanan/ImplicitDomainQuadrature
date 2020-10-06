using SafeTestsets

@safetestset "Test Arithmetic Definitions for Taylor Models" begin
    include("test_arithmetic.jl")
end

@safetestset "Test Function Bounds" begin
    include("test_bounds.jl")
end

@safetestset "Test Quadrature Structs" begin
    include("test_quadrature_structs.jl")
end

@safetestset "Test Dimension Reduction Algorithm" begin
    include("test_dimension_reduction.jl")
end

@safetestset "Test Subdivision & Perturbation Algorithm" begin
    include("test_subdivision.jl")
end
