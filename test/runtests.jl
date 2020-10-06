using SafeTestsets

@safetestset "Test Arithmetic Definitions for Taylor Models" begin
    include("test_arithmetic.jl")
end

@safetestset "Test Function Bounds" begin
    include("test_bounds.jl")
end

@safetestset "Test Utilities" begin
    include("test_utilities.jl")
end

@safetestset "Test Quadrature Structs" begin
    include("test_quadrature_structs.jl")
end

@safetestset "Test One-Dimensional Quadrature" begin
    include("test_one_dimensional_quadrature.jl")
end

@safetestset "Test Area Quadrature" begin
    include("test_area_quadrature.jl")
end

@safetestset "Test Surface Quadrature" begin
    include("test_surface_quadrature.jl")
end

@safetestset "Test Subdivision & Perturbation Algorithm" begin
    include("test_subdivision.jl")
end
