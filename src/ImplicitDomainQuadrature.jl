module ImplicitDomainQuadrature

using PolynomialBasis
using StaticArrays
using TaylorModels
using LinearAlgebra
using FastGaussQuadrature
using Roots
using IntervalRootFinding

include("arithmetic.jl")
include("bounds.jl")
include("utilities.jl")
include("quadrature_structs.jl")
include("one_dimensional_quadrature.jl")
include("area_quadrature.jl")
include("surface_quadrature.jl")

# export quadrature, QuadratureRule, tensor_product_quadrature, tensor_product

end # module
