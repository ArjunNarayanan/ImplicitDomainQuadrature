function polytype(B::LagrangePolynomialBasis)
    return eltype(B.funcs.polys[1].coefficients)
end

function polytype(B::TensorProductBasis)
    return eltype(B.basis.funcs.polys[1].coefficients)
end

"""
    InterpolatingPolynomial{N,NFuncs,B<:AbstractBasis,T}
interpolate a VECTOR of `N` with a basis `B` composed of `NFuncs` functions
# Fields
    - `coeffs::SMatrix{N,NFuncs,T}`
    - `basis::B`
"""
mutable struct InterpolatingPolynomial{N,NFuncs,B<:AbstractBasis,T}
    coeffs::SMatrix{N,NFuncs,T}
    basis::B
    function InterpolatingPolynomial(coeffs::SMatrix{N,NFuncs,T},
        basis::B) where {B<:AbstractBasis{NFuncs}} where {N,NFuncs,T}

        S = polytype(basis)
        if S != T
            @warn "Coefficient type and polynomial type are not the same"
        end
        new{N,NFuncs,B,T}(coeffs,basis)
    end
end

"""
    InterpolatingPolynomial(T::Type{<:Number}, N::Int,
        basis::AbstractBasis{NFuncs}) where {NFuncs}
initialize an `InterpolatingPolynomial{N,NFuncs,T}` object with coefficients
`zeros(T,N,NFuncs)`
    InterpolatingPolynomial(N::Int, basis::AbstractBasis)
initialize an `InterpolatingPolynomial` object with `Float64` coefficients.
    InterpolatingPolynomial(N::Int, dim::Int, order::Int, start::T, stop::T) where {T<:Real}
initialize a `dim` dimensional basis of order `order` and pass this to the
`InterpolatingPolynomial` constructor.
    InterpolatingPolynomial(N::Int, dim::Int, order::Int; start = -1.0, stop = 1.0)
initialize a `dim` dimensional basis of order `order` and pass this to the
`InterpolatingPolynomial` constructor.
"""
function InterpolatingPolynomial(T::Type{<:Number}, N::Int,
    basis::AbstractBasis{NFuncs}) where {NFuncs}

    coeffs = SMatrix{N,NFuncs}(zeros(T,N,NFuncs))
    return InterpolatingPolynomial(coeffs,basis)
end

function InterpolatingPolynomial(N::Int, basis::AbstractBasis)
    return InterpolatingPolynomial(Float64, N, basis)
end

function InterpolatingPolynomial(N::Int, dim::Int, order::Int, start::T, stop::T) where {T<:Real}
    basis = TensorProductBasis(dim, order, start, stop)
    return InterpolatingPolynomial(T, N, basis)
end

function InterpolatingPolynomial(N::Int, dim::Int, order::Int; start = -1.0, stop = 1.0)
    basis = TensorProductBasis(dim, order, start, stop)
    return InterpolatingPolynomial(N,basis)
end

"""
    update!(P::InterpolatingPolynomial, coeffs::AbstractMatrix)
update `P.coeffs = coeffs`
"""
function update!(P::InterpolatingPolynomial, coeffs::AbstractVecOrMat)
    P.coeffs = coeffs
end

"""
    (P::InterpolatingPolynomial{1})(x...)
evaluate `P` at `x`, the result is a scalar
    (P::InterpolatingPolynomial)(x...)
evaluate `P` at `x`
    (P::InterpolatingPolynomial)(x::AbstractVector)
evaluate `P` at the point vector `x`
"""
function (P::InterpolatingPolynomial{1})(x...)
    return ((P.coeffs)*(P.basis(x...)))[1]
end

function (P::InterpolatingPolynomial)(x...)
    return ((P.coeffs)*(P.basis(x...)))
end

"""
    gradient(P::InterpolatingPolynomial)(x)
return the gradient of `P` evaluated at `x`.
    gradient(P::InterpolatingPolynomial, dir::Int, x...)
return the gradient of `P` along direction `dir` evaluated at `x`.
    gradient(P::InterpolatingPolynomial, x::AbstractVector)
evaluate the gradient of `P` at the point vector `x`
"""
function gradient(P::InterpolatingPolynomial{1}, x::Number)
    return ((P.coeffs)*(gradient(P.basis, x...)))[1]
end
function gradient(P::InterpolatingPolynomial, x...)
    return ((P.coeffs)*(gradient(P.basis, x...)))
end

function gradient(P::InterpolatingPolynomial, dir::Int, x...)
    return (P.coeffs)*(gradient(P.basis, dir, x...))
end

function gradient(P::InterpolatingPolynomial{1}, dir::Int, x...)
    return ((P.coeffs)*(gradient(P.basis, dir, x...)))[1]
end

function gradient(P::InterpolatingPolynomial, x::AbstractVector)
    return (P.coeffs)*(gradient(P.basis, x))
end
