# This is a naive implementation of lagrange polynomials
# Such a method is prone to cancellation errors due to finite precision.
# More accurate algorithms such as Barycentric Interpolation formulae exist
# but unfortunately these cannot be used with Taylor Models which is necessary
# in some cases.
# The current approach is fine as long as polynomial order ~ 10. For higher orders
# consider using BigFloat

"""
    polynomial_from_roots(x::DP.PolyVar,roots::AbstractVector{T},skipindex::Int) where {T}

Return a `DynamicPolynomial` with variable `x` with zeros at `roots` but skipping the
`skipindex`th root.

# Examples
```julia-repl
julia> import DynamicPolynomials.@polyvar
julia> @polyvar x
(x,)
julia> ImplicitDomainQuadrature.polynomial_from_roots(x,[-1,1],1)
x - 1
```
"""
function polynomial_from_roots(x::DP.PolyVar,roots::AbstractVector{T},skipindex::Int) where {T}
 f = zero(T)*x + one(T)
 for (idx,r) in enumerate(roots)
     if idx != skipindex
         f *= (x - r)
     end
 end
 return f
end

"""
    normalization(roots::AbstractVector{T},skipindex::Int) where {T}

Return the denominator of the standard Lagrange polynomial formula.

Dividing `f = polynomial_from_roots(x,roots,skipindex)/normalization(roots,skipindex)`
gives a normalized polynomial which evaluates to `1.0` on the `skipindex`th
root i.e. `f(roots[skipindex] = 1.0`.

# Examples
```julia-repl
julia> ImplicitDomainQuadrature.normalization([-1,1],1)
-2
```
"""
function normalization(roots::AbstractVector{T},skipindex::Int) where {T}
 v = one(T)
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

Return the `index`th lagrange polynomial from the vector of `roots` with
variable `x`.

The polynomial is normalized such that it evaluates to `1.0` on `roots[index]`.

# Examples
```julia-repl
julia> import DynamicPolynomials.@polyvar
julia> @polyvar x
(x,)
julia> ImplicitDomainQuadrature.lagrange_polynomial(x,[-1,0,1],2)
-x² + 1.0
```

See also: [`lagrange_polynomials`](@ref)
"""
function lagrange_polynomial(x::DP.PolyVar,roots::AbstractVector{T},index::Int) where {T}

 f = polynomial_from_roots(x,roots,index)
 d = normalization(roots,index)
 return f/d
end

"""
    lagrange_polynomials(x::DP.PolyVar,roots::AbstractVector{T}) where {T}

Return a vector of lagrange polynomial bases constructed from the given `roots`.

# Examples
```julia-repl
julia> import DynamicPolynomials.@polyvar
julia> @polyvar x
(x,)
julia> ImplicitDomainQuadrature.lagrange_polynomials(x, [-1,0,1])
3-element Array{DynamicPolynomials.Polynomial{true,Float64},1}:
 0.5x² - 0.5x
 -x² + 1.0
 0.5x² + 0.5x
```

See also: [`lagrange_polynomial`](@ref)
"""
function lagrange_polynomials(x::DP.PolyVar,roots::AbstractVector{T}) where {T}
 num_roots = length(roots)
 basis = [lagrange_polynomial(x,roots,i) for i in 1:num_roots]
 return basis
end
