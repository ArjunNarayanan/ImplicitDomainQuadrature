module ImplicitDomainQuadrature

using LinearAlgebra
using IntervalArithmetic
using FastGaussQuadrature
using Roots
using StaticArrays
using PolynomialBasis

include("bounds.jl")
include("utilities.jl")
include("quadrature_structs.jl")
include("one_dimensional_quadrature.jl")
include("area_quadrature.jl")
include("surface_quadrature.jl")

# export quadrature, QuadratureRule, tensor_product_quadrature, tensor_product

end # module
