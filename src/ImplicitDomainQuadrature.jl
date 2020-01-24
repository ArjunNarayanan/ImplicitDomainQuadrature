module ImplicitDomainQuadrature

using StaticArrays, BranchAndPrune, RangeEnclosures
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
       gradient, sign

end # module
