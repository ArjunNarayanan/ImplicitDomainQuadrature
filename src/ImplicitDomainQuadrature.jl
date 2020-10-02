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
include("quadrature_structs.jl")
include("dimension_reduction.jl")

export quadrature, QuadratureRule, tensor_product_quadrature, tensor_product

end # module
