function unique_root_intervals(f, x1, x2)
    all_roots = roots(f, Interval(x1, x2))
    @assert all([r.status == :unique for r in all_roots])
    return [r.interval for r in all_roots]
end

function unique_roots(f, x1, x2)
    root_intervals = unique_root_intervals(f, x1, x2)
    _roots = zeros(length(root_intervals))
    for (idx,int) in enumerate(root_intervals)
        _roots[idx] = find_zero(f, (int.lo, int.hi), Order1())
    end
    return _roots
end

function roots_and_ends(F, x1, x2)
    r = [x1,x2]
    for f in F
        roots = unique_roots(f, x1, x2)
        append!(r,roots)
    end
    return sort!(r)
end

function extend(x0::T,k,x) where {T<:Number}
    if k == 1
        return [x, x0]
    elseif k == 2
        return [x0, x]
    else
        throw(ArgumentError("Expected k ∈ {1,2}, got $k"))
    end
end

function extend(x0::V,k,x::T) where {V<:AbstractVector,T<:Number}
    dim = length(x0)
    if dim == 1
        return extend(x0[1],k,x)
    else
        throw(ArgumentError("Current support for dim = 1, got dim = $dim"))
    end
end

function extend(x0::V,k,x::M) where {V<:AbstractVector,M<:AbstractMatrix}
    old_dim = length(x0)
    dim, npoints = size(x)
    if dim != 1
        msg = "Current support for dim = 1, got dim = $dim"
        throw(ArgumentError(msg))
    end

    old_row = repeat(x0, inner = (1,npoints))
    if old_dim == 1 && k == 1
        return vcat(x, old_row)
    elseif old_dim == 1 && k == 2
        return vcat(old_row, x)
    else
        throw(ArgumentError("Current support for dim = 1 and k ∈ {1,2}, got dim = $dim, k = $k"))
    end
end

mutable struct TemporaryQuadrature{T}
    points::Matrix{T}
    weights::Vector{T}
    function TemporaryQuadrature(p::Matrix{T},w::Vector{T}) where {T}
        d,np = size(p)
        nw = length(w)
        @assert np == nw
        new{T}(p,w)
    end
end

function sign_conditions_satisfied(funcs,xc,sign_conditions)
    return all(i -> funcs[i](xc)*sign_conditions[i] >= 0.0, 1:length(funcs))
end

function TemporaryQuadrature(T,D::Z) where {Z<:Integer}
    @assert 1 <= D <= 3
    p = Matrix{T}(undef,D,0)
    w = Vector{T}(undef,0)
    return TemporaryQuadrature(p,w)
end

function update!(quad::TemporaryQuadrature,p::V,w) where {V<:AbstractVector}
    d = length(p)
    nw = length(w)
    dq,nqp = size(quad.points)
    @assert dq == d
    @assert nw == 1
    quad.points = hcat(quad.points,p)
    append!(quad.weights,w)
end

function update!(quad::TemporaryQuadrature,p::M,w) where {M<:AbstractMatrix}
    d,np = size(p)
    nw = length(w)
    dq,nqp = size(quad.points)
    @assert dq == d
    @assert nw == np
    quad.points = hcat(quad.points,p)
    append!(quad.weights,w)
end

function quadrature(F,sign_conditions,lo,hi,
    quad1d::ReferenceQuadratureRule{N,T}) where {N,T}

    nfuncs = length(F)
    nconds = length(sign_conditions)
    @assert nfuncs == nconcs

end
