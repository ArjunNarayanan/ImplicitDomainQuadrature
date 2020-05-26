module ImplicitDomainQuadrature

using PolynomialBasis
using StaticArrays, BranchAndPrune, TaylorModels, LinearAlgebra
using FastGaussQuadrature
using Roots, IntervalRootFinding
import DynamicPolynomials
import StaticPolynomials

# abbreviations
DP = DynamicPolynomials
SP = StaticPolynomials

# include("lagrange_polynomials.jl")
# include("basis.jl")
# include("interpolation.jl")
include("bounds.jl")
include("quadrature_structs.jl")
include("dimension_reduction.jl")

export sign, quadrature, QuadratureRule, TensorProductQuadratureRule

end # module
