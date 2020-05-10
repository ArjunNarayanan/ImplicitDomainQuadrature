module ImplicitDomainQuadrature

using StaticArrays, BranchAndPrune, TaylorModels, LinearAlgebra
using FastGaussQuadrature
using Roots, IntervalRootFinding
import DynamicPolynomials
import StaticPolynomials

# abbreviations
DP = DynamicPolynomials
SP = StaticPolynomials

include("lagrange_polynomials.jl")
include("basis.jl")
include("interpolation.jl")
include("bounds.jl")
include("quadrature_structs.jl")
include("dimension_reduction.jl")

export AbstractBasis, TensorProductBasis, InterpolatingPolynomial, update!,
       interpolation_points, gradient, sign, ==, isequal, quadrature,
       QuadratureRule, TensorProductQuadratureRule, ReferenceQuadratureRule

end # module
