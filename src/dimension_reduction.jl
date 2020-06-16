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

function quadrature(funcs,sign_conditions,lo,hi,
    quad1d::ReferenceQuadratureRule{N,T}) where {N,T}

    nfuncs = length(funcs)
    nconds = length(sign_conditions)
    @assert nfuncs == nconds

    quad = TemporaryQuadrature(T,1)

    roots = roots_and_ends(funcs,lo,hi)
    for j in 1:length(roots)-1
        xc = 0.5*(roots[j] + roots[j+1])
        if sign_conditions_satisfied(funcs,xc,sign_conditions)
            p,w = transform(quad1d, roots[j], roots[j+1])
            update!(quad, p, w)
        end
    end
    return quad

end

function quadrature(funcs,sign_conditions,interval,quad1d)

    return quadrature(funcs,sign_conditions,interval.lo,interval.hi,quad1d)
end

function quadrature(funcs,sign_conditions,height_dir,lo,hi,x0::V,w0,
    quad1d::ReferenceQuadratureRule{N,T}) where {N,T,V<:AbstractVector}

    @assert length(funcs) == length(sign_conditions)

    lower_dim = length(x0)
    dim = lower_dim + 1
    quad = TemporaryQuadrature(T,dim)

    extended_funcs = [x -> f(extend(x0,height_dir,x)) for f in funcs]
    roots = roots_and_ends(extended_funcs,lo,hi)
    for j in 1:length(roots)-1
        xc = 0.5*(roots[j] + roots[j+1])
        if sign_conditions_satisfied(extended_funcs,xc,sign_conditions)
            p,w = transform(quad1d, roots[j], roots[j+1])
            extended_points = extend(x0,height_dir,p)
            update!(quad,extended_points,w0*w)
        end
    end
    return quad

end

function quadrature(funcs,sign_conditions,height_dir,lo,hi,x0::M,w0::V,
    quad1d::ReferenceQuadratureRule{N,T}) where {N,T,V<:AbstractVector,M<:AbstractMatrix}

    @assert length(funcs) == length(sign_conditions)
    lower_dim, npoints = size(x0)
    nweights = length(w0)
    @assert npoints == nweights

    dim = lower_dim + 1
    quad = TemporaryQuadrature(T,dim)

    for i in 1:npoints
        x0p = view(x0,:,i)
        w0p = w0[i]
        lq = quadrature(funcs,sign_conditions,height_dir,lo,hi,x0p,w0p,quad1d)
        update!(quad,lq.points,lq.weights)
    end
    return quad
end

function quadrature(funcs,sign_conditions,height_dir,interval::Interval,
    x0,w0,quad1d)

    return quadrature(funcs,sign_conditions,height_dir,
            interval.lo,interval.hi,x0,w0,quad1d)
end

function surface_quadrature(func,height_dir,lo,hi,x0::V,w0) where {V<:AbstractVector}

    extended_func(x) = func(extend(x0,height_dir,x))
    _roots = unique_roots(extended_func, lo, hi)

    num_roots = length(_roots)
    num_roots == 1 || throw(ArgumentError("Expected 1 root in given interval, got $num_roots roots"))

    root = _roots[1]
    p = extend(x0,height_dir,root)
    gradF = gradient(func,p)
    jac = norm(gradF)/(abs(gradF[height_dir]))
    w = w0*jac
    return p,w
end

function surface_quadrature(func,height_dir,lo,hi,x0::M,w0::V) where {M<:AbstractMatrix{T},V<:AbstractVector} where {T}

    lower_dim, npoints = size(x0)
    nweights = length(w0)
    @assert npoints == nweights

    dim = lower_dim + 1
    quad = TemporaryQuadrature(T,dim)

    for i in 1:npoints
        x0p = view(x0,:,i)
        w0p = w0[i]
        qp,qw = surface_quadrature(func,height_dir,lo,hi,x0p,w0p)
        update!(quad,qp,qw)
    end
    return quad
end

function surface_quadrature(func,height_dir,interval::Interval,x0,w0)

    return surface_quadrature(func,height_dir,interval.lo,interval.hi,x0,w0)
end

function height_direction(func,xc)
    grad = vec(gradient(func,xc))
    @assert !(norm(grad) ≈ 0.0)
    return argmax(abs.(grad))
end

function height_direction(func,box::IntervalBox)
    return height_direction(func,mid(box))
end

function is_suitable(height_dir,func,box;order=5,tol=1e-3)

    gradf(x...) = gradient(func,height_dir,x...)
    s = sign(gradf,box,order=order,tol=tol)
    flag = s == 0 ? false : true
    return flag, s

end

function Base.sign(m::Z,s::Z,S::Bool,sigma::Z) where {Z<:Integer}

    if S || m == sigma*s
        return sigma*m
    else
        return 0
    end
end

function quadrature(func,sign_condition,surface,box::IntervalBox{2},
    quad1d::ReferenceQuadratureRule{NQ,T}) where {NQ,T}

    s = sign(func,box)
    if s*sign_condition < 0
        quad = TemporaryQuadrature(T,2)
        return QuadratureRule(quad.points,quad.weights)
    elseif s*sign_condition > 0
        return tensor_product(quad1d,box)
    else
        height_dir = height_direction(func,box)
        flag, gradient_sign = is_suitable(height_dir,func,box)
        @assert gradient_sign != 0 "Subdivision not implemented yet"

        lower(x) = func(extend(x,height_dir,box[height_dir].lo))
        upper(x) = func(extend(x,height_dir,box[height_dir].hi))

        lower_sign = sign(gradient_sign,sign_condition,surface,-1)
        upper_sign = sign(gradient_sign,sign_condition,surface,+1)

        lower_box = height_dir == 1 ? box[2] : box[1]
        quad = quadrature([lower,upper], [lower_sign,upper_sign],
            lower_box, quad1d)

        if surface
            newquad = surface_quadrature(func,height_dir,box[height_dir],
                        quad.points,quad.weights)
            return QuadratureRule(newquad.points,newquad.weights)
        else
            newquad = quadrature([func],[sign_condition],height_dir,
                        box[height_dir],quad.points,quad.weights,quad1d)
            return QuadratureRule(newquad.points,newquad.weights)
        end
    end
end
