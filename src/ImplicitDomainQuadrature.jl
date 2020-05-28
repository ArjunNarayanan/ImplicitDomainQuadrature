module ImplicitDomainQuadrature

using PolynomialBasis
using StaticArrays, BranchAndPrune, TaylorModels, LinearAlgebra
using FastGaussQuadrature
using Roots, IntervalRootFinding


include("bounds.jl")
include("quadrature_structs.jl")
include("dimension_reduction.jl")

export quadrature, QuadratureRule, TensorProductQuadratureRule

end # module
