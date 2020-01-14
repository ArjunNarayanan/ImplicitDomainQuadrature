using StaticArrays
import DynamicPolynomials
import StaticPolynomials
import Base: kron

# abbreviations
DP = DynamicPolynomials
SP = StaticPolynomials

"""
    AbstractBasis{N}
abstract supertype for a function basis with `N` functions
"""
abstract type AbstractBasis{N} end

"""
    AbstractBasis1D{N}
abstract supertype for a 1D function basis with `N` functions
"""
abstract type AbstractBasis1D{N} <: AbstractBasis{N} end

"""
    LagrangePolynomialBasis{N} <: AbstractBasis1D{N}
A basis of `N` Lagrange polynomials. The polynomial order is `N + 1`.
# Fields
    - `funcs::PolynomialSystem{N,1}` - a system of static polynomials
    - `points::StaticVector{N}` - a static vector of support points
"""
struct LagrangePolynomialBasis{NFuncs} <: AbstractBasis1D{NFuncs}
    funcs::SP.PolynomialSystem{NFuncs,1}
    points::SVector{NFuncs}
    function LagrangePolynomialBasis{NFuncs}(funcs::AbstractVector{DP.Polynomial{C,T1}},
        points::AbstractVector{T2}) where {NFuncs} where {C,T1,T2}

        order = NFuncs - 1
        npoints = length(points)
        if NFuncs < 2
            msg = "Number of functions must be greater than 1"
            throw(ArgumentError(msg))
        end
        if NFuncs != length(funcs)
            length_funcs = length(funcs)
            msg = "Expected $NFuncs functions, got $length_funcs"
            throw(ArgumentError(msg))
        end
        if NFuncs != npoints
            msg = "Number of functions must be equal to number of points"
            throw(ArgumentError(msg))
        end
        if !isapprox(sum(funcs),1.0)
            val = sum(funcs)
            msg = "Polynomial basis must sum to 1.0, got $val"
        end
        vars = funcs[1].x.vars[1]
        for f in funcs
            if length(f.x.vars) != 1
                nvars = length(f.x.vars)
                msg = "Require univariate polynomials, got $nvars variables"
                throw(ArgumentError(msg))
            end
            if !(vars in f.x.vars)
                msg = "Polynomial functions must have the same variables"
                throw(ArgumentError(msg))
            end
            ord = maximum(maximum.(f.x.Z))
            if ord != order
                msg = "Polynomial functions must have the same order"
                throw(ArgumentError(msg))
            end
        end
        polysystem = SP.PolynomialSystem(funcs)
        static_points = SVector{NFuncs,T2}(points)
        new{NFuncs}(polysystem,static_points)
    end
end
