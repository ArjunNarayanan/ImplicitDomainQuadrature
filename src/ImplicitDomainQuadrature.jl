module ImplicitDomainQuadrature

using LinearAlgebra
using IntervalArithmetic
using FastGaussQuadrature
using Roots
using PolynomialBasis

include("bounds.jl")
include("utilities.jl")
include("quadrature_structs.jl")
include("one_dimensional_quadrature.jl")
include("area_quadrature.jl")
include("surface_quadrature.jl")

export sign_allow_perturbations,
    area_quadrature,
    surface_quadrature,
    tensor_product_quadrature,
    QuadratureRule,
    ReferenceQuadratureRule,
    update_interpolating_gradient!

end # module
