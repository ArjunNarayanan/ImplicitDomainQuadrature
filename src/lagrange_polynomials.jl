# This is a naive implementation of lagrange polynomials
# Such a method is prone to cancellation errors due to finite precision
# Use the numerically more stable barycentric formula for accurate results
# The current approach is fine as long as polynomial order ~ 10

"""
    polynomial_from_roots(x::DP.PolyVar,roots::AbstractVector{T},skipindex::Int) where {T}
construct the polynomial with variable `x` from its `roots` by skipping the
`skipindex`th root.
"""
function polynomial_from_roots(x::DP.PolyVar,roots::AbstractVector{T},skipindex::Int) where {T}
 f = 0.0*x + 1.0
 for (idx,r) in enumerate(roots)
     if idx != skipindex
         f *= (x - r)
     end
 end
 return f
end

"""
    normalization(roots::AbstractVector{T},skipindex::Int) where {T}
return the normalization factor such that the polynomial evaluated on the
`skipindex`th root has value `1.0`.
"""
function normalization(roots::AbstractVector{T},skipindex::Int) where {T}
 v = 1.0
 a = roots[skipindex]
 for (idx,r) in enumerate(roots)
     if idx != skipindex
         v *= (a - r)
     end
 end
 return v
end

"""
    lagrange_polynomial(x::DP.PolyVar,roots::AbstractVector{T},index::Int) where {T}
return the `index`th lagrange polynomial from the vector of `roots` with
variable `x`.
"""
function lagrange_polynomial(x::DP.PolyVar,roots::AbstractVector{T},index::Int) where {T}

 f = polynomial_from_roots(x,roots,index)
 d = normalization(roots,index)
 return f/d
end

"""
    lagrange_polynomials(x::DP.PolyVar,roots::AbstractVector{T}) where {T}
return a vector of lagrange polynomial basis constructed from the given `roots`
"""
function lagrange_polynomials(x::DP.PolyVar,roots::AbstractVector{T}) where {T}
 num_roots = length(roots)
 basis = [lagrange_polynomial(x,roots,i) for i in 1:num_roots]
 return basis
end
