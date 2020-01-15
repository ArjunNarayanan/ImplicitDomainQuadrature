module ImplicitDomainQuadrature

using StaticArrays
import DynamicPolynomials
import StaticPolynomials

# abbreviations
DP = DynamicPolynomials
SP = StaticPolynomials

include("lagrange_polynomials.jl")
include("basis.jl")
include("interpolation.jl")

export LagrangePolynomialBasis, TensorProductBasis,
       InterpolatingPolynomial, update!

end # module
