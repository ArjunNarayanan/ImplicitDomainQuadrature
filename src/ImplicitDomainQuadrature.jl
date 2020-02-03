module ImplicitDomainQuadrature

using StaticArrays, BranchAndPrune, TaylorModels
using RangeEnclosures
import DynamicPolynomials
import StaticPolynomials

# abbreviations
DP = DynamicPolynomials
SP = StaticPolynomials

include("lagrange_polynomials.jl")
include("basis.jl")
include("interpolation.jl")
include("bounds.jl")

export TensorProductBasis, InterpolatingPolynomial, update!, interpolation_points,
       gradient, sign, ==, isequal

end # module
