function unique_root_intervals(f, x1, x2)
    all_roots = roots(f, Interval(x1, x2))
    return [r.interval for r in all_roots]
end

function unique_roots(f, x1, x2)
    root_intervals = unique_root_intervals(f, x1, x2)
    _roots = zeros(length(root_intervals))
    for (idx, int) in enumerate(root_intervals)
        _roots[idx] = find_zero(f, (int.lo, int.hi), Order1())
    end
    return _roots
end

function roots_and_ends(F, x1, x2)
    r = [x1, x2]
    for f in F
        roots = unique_roots(f, x1, x2)
        append!(r, roots)
    end
    return unique!(sort!(r))
end

function extend(x0::T, k, x) where {T<:Number}
    if k == 1
        return [x, x0]
    elseif k == 2
        return [x0, x]
    else
        throw(ArgumentError("Expected k ∈ {1,2}, got $k"))
    end
end

function extend(x0::V, k, x::T) where {V<:AbstractVector,T<:Number}
    dim = length(x0)
    if dim == 1
        return extend(x0[1], k, x)
    else
        throw(ArgumentError("Current support for dim = 1, got dim = $dim"))
    end
end

function extend(x0::V, k, x::M) where {V<:AbstractVector,M<:AbstractMatrix}
    old_dim = length(x0)
    dim, npoints = size(x)
    if dim != 1
        msg = "Current support for dim = 1, got dim = $dim"
        throw(ArgumentError(msg))
    end

    old_row = repeat(x0, inner = (1, npoints))
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
    function TemporaryQuadrature(
        p::M,
        w::V,
    ) where {M<:AbstractMatrix{T},V<:AbstractVector} where {T}
        d, np = size(p)
        nw = length(w)
        @assert np == nw
        new{T}(p, w)
    end
end

function TemporaryQuadrature(T, D::Z) where {Z<:Integer}
    @assert 1 <= D <= 3
    p = Matrix{T}(undef, D, 0)
    w = Vector{T}(undef, 0)
    return TemporaryQuadrature(p, w)
end

function QuadratureRule(quad::TemporaryQuadrature)
    return QuadratureRule(quad.points, quad.weights)
end

function temporary_tensor_product(
    quad::ReferenceQuadratureRule,
    box::IntervalBox{2},
)

    p1, w1 = transform(quad, box[1])
    p2, w2 = transform(quad, box[2])
    points = tensor_product_points(p1, p2)
    weights = kron(w1, w2)
    return TemporaryQuadrature(points, weights)
end

function update!(quad::TemporaryQuadrature, p::V, w) where {V<:AbstractVector}
    d = length(p)
    nw = length(w)
    dq, nqp = size(quad.points)
    @assert dq == d
    @assert nw == 1
    quad.points = hcat(quad.points, p)
    append!(quad.weights, w)
end

function update!(quad::TemporaryQuadrature, p::M, w) where {M<:AbstractMatrix}
    d, np = size(p)
    nw = length(w)
    dq, nqp = size(quad.points)
    @assert dq == d
    @assert nw == np
    quad.points = hcat(quad.points, p)
    append!(quad.weights, w)
end

function sign_conditions_satisfied(funcs, xc, sign_conditions)
    return all(i -> funcs[i](xc) * sign_conditions[i] >= 0.0, 1:length(funcs))
end

function one_dimensional_quadrature(
    funcs,
    sign_conditions,
    lo,
    hi,
    quad1d::ReferenceQuadratureRule{N,T},
) where {N,T}

    nfuncs = length(funcs)
    nconds = length(sign_conditions)
    @assert nfuncs == nconds

    quad = TemporaryQuadrature(T, 1)

    roots = roots_and_ends(funcs, lo, hi)
    for j = 1:length(roots)-1
        xc = 0.5 * (roots[j] + roots[j+1])
        if sign_conditions_satisfied(funcs, xc, sign_conditions)
            p, w = transform(quad1d, roots[j], roots[j+1])
            update!(quad, p, w)
        end
    end
    return quad

end

function one_dimensional_quadrature(funcs, sign_conditions, interval, quad1d)

    return one_dimensional_quadrature(
        funcs,
        sign_conditions,
        interval.lo,
        interval.hi,
        quad1d,
    )
end

function extended_one_dimensional_quadrature(
    funcs,
    sign_conditions,
    height_dir,
    lo,
    hi,
    x0::V,
    w0,
    quad1d::ReferenceQuadratureRule{N,T},
) where {N,T,V<:AbstractVector}

    @assert length(funcs) == length(sign_conditions)

    lower_dim = length(x0)
    dim = lower_dim + 1
    quad = TemporaryQuadrature(T, dim)

    extended_funcs = [x -> f(extend(x0, height_dir, x)) for f in funcs]
    roots = roots_and_ends(extended_funcs, lo, hi)
    for j = 1:length(roots)-1
        xc = 0.5 * (roots[j] + roots[j+1])
        if sign_conditions_satisfied(extended_funcs, xc, sign_conditions)
            p, w = transform(quad1d, roots[j], roots[j+1])
            extended_points = extend(x0, height_dir, p)
            update!(quad, extended_points, w0 * w)
        end
    end
    return quad
end

function extended_one_dimensional_quadrature(
    funcs,
    sign_conditions,
    height_dir,
    lo,
    hi,
    x0::M,
    w0::V,
    quad1d::ReferenceQuadratureRule{N,T},
) where {N,T,V<:AbstractVector,M<:AbstractMatrix}

    @assert length(funcs) == length(sign_conditions)
    lower_dim, npoints = size(x0)
    nweights = length(w0)
    @assert npoints == nweights

    dim = lower_dim + 1
    quad = TemporaryQuadrature(T, dim)

    for i = 1:npoints
        x0p = view(x0, :, i)
        w0p = w0[i]
        lq = extended_one_dimensional_quadrature(
            funcs,
            sign_conditions,
            height_dir,
            lo,
            hi,
            x0p,
            w0p,
            quad1d,
        )
        update!(quad, lq.points, lq.weights)
    end
    return quad
end

function extended_one_dimensional_quadrature(
    funcs,
    sign_conditions,
    height_dir,
    interval,
    x0,
    w0,
    quad1d,
)

    return extended_one_dimensional_quadrature(
        funcs,
        sign_conditions,
        height_dir,
        interval.lo,
        interval.hi,
        x0,
        w0,
        quad1d,
    )
end

function surface_quadrature(
    func,
    height_dir,
    lo,
    hi,
    x0::V,
    w0,
) where {V<:AbstractVector}

    extended_func(x) = func(extend(x0, height_dir, x))
    _roots = unique_roots(extended_func, lo, hi)

    num_roots = length(_roots)
    num_roots == 1 ||
        throw(ArgumentError("Expected 1 root in given interval, got $num_roots roots"))

    root = _roots[1]
    p = extend(x0, height_dir, root)
    gradF = gradient(func, p)
    jac = norm(gradF) / (abs(gradF[height_dir]))
    w = w0 * jac
    return p, w
end

function surface_quadrature(
    func,
    height_dir,
    lo,
    hi,
    x0::M,
    w0::V,
) where {M<:AbstractMatrix{T},V<:AbstractVector} where {T}

    lower_dim, npoints = size(x0)
    nweights = length(w0)
    @assert npoints == nweights

    dim = lower_dim + 1
    quad = TemporaryQuadrature(T, dim)

    for i = 1:npoints
        x0p = view(x0, :, i)
        w0p = w0[i]
        qp, qw = surface_quadrature(func, height_dir, lo, hi, x0p, w0p)
        update!(quad, qp, qw)
    end
    return quad
end

function surface_quadrature(func, height_dir, interval, x0, w0)

    return surface_quadrature(
        func,
        height_dir,
        interval.lo,
        interval.hi,
        x0,
        w0,
    )
end

function height_direction(grad, xc)
    grad = vec(grad(xc))
    if norm(grad) ≈ 0.0
        return 0
    else
        return argmax(abs.(grad))
    end
end

function height_direction(grad, box::IntervalBox{N,T}) where {N,T}
    return height_direction(grad, mid(box))
end

function curvature_measure(grad, k, C)
    return sum(grad .^ 2) - C^2 * grad[k]^2
end

function is_suitable(height_dir, grad, box; order = 5, tol = 1e-3, C = 4)

    if height_dir == 0
        return false,0
    else
        s = sign(x -> grad(x)[height_dir], box, order = order, tol = tol)
        curvatureflag = true
        if s != 0
            curve(x) = curvature_measure(grad(x), height_dir, C)
            t = sign(curve, box, order = order, tol = tol)
            curvatureflag = t == -1 ? false : true
        end
        flag = (s == 0 || curvatureflag) ? false : true
        return flag, s
    end
end

function Base.sign(m::Z, s::Z, S::Bool, sigma::Z) where {Z<:Integer}

    if S || m == sigma * s
        return sigma * m
    else
        return 0
    end
end

function combine_quadratures(splitquads)
    points = hcat([sq.points for sq in splitquads]...)
    weights = vcat([sq.weights for sq in splitquads]...)
    return TemporaryQuadrature(points, weights)
end

function area(box::IntervalBox{2,T}) where {T}
    return prod([box[i].hi - box[i].lo for i = 1:2])
end

function cut_area_quadrature(
    func,
    grad,
    sign_condition,
    box,
    quad1d,
    recursionlevel,
    maxlevels,
    perturbation,
    numperturbation,
    maxperturbations,
)
    heightdir = height_direction(grad, box)
    issuitable, gradsign = is_suitable(heightdir, grad, box)
    if !issuitable
        splitboxes = split_box(box)
        splitquads = [
            subdivision_area_quadrature(
                func,
                grad,
                sign_condition,
                sb,
                quad1d,
                recursionlevel + 1,
                maxlevels,
                perturbation,
                numperturbation,
                maxperturbations,
            ) for sb in splitboxes
        ]
        return combine_quadratures(splitquads)
    else
        lower(x) = func(extend(x, heightdir, box[heightdir].lo))
        upper(x) = func(extend(x, heightdir, box[heightdir].hi))

        lowersign = sign(gradsign, sign_condition, false, -1)
        uppersign = sign(gradsign, sign_condition, false, +1)

        lowerbox = heightdir == 1 ? box[2] : box[1]

        edgequad = one_dimensional_quadrature(
            [lower, upper],
            [lowersign, uppersign],
            lowerbox,
            quad1d,
        )

        return extended_one_dimensional_quadrature(
            [func],
            [sign_condition],
            heightdir,
            box[heightdir],
            edgequad.points,
            edgequad.weights,
            quad1d,
        )
    end
end


function subdivision_area_quadrature(
    func,
    grad,
    sign_condition,
    box,
    quad1d::ReferenceQuadratureRule{NQ,T},
    recursionlevel,
    maxlevels,
    perturbation,
    numperturbation,
    maxperturbations,
) where {NQ,T}

    @assert sign_condition == +1 || sign_condition == -1

    if recursionlevel == maxlevels
        xc = reshape(mid(box), 2, 1)
        wt = [area(box)]
        return TemporaryQuadrature(xc, wt)
    else
        
        s = 2
        try
            s = sign(func, box)
        catch e
            if numperturbation == maxperturbations
                errorstr = "Failed to construct quadrature after $numperturbation perturbations of size $perturbation"
                error(errorstr)
            else
                return subdivision_area_quadrature(
                    x -> func(x) + perturbation,
                    grad,
                    sign_condition,
                    box,
                    quad1d,
                    recursionlevel,
                    maxlevels,
                    perturbation,
                    numperturbation + 1,
                    maxperturbations,
                )
            end
        end

        @assert s == -1 || s == 0 || s == 1

        if s * sign_condition < 0
            return TemporaryQuadrature(T, 2)
        elseif s * sign_condition > 0
            return temporary_tensor_product(quad1d, box)
        else
            return cut_area_quadrature(
                func,
                grad,
                sign_condition,
                box,
                quad1d,
                recursionlevel,
                maxlevels,
                perturbation,
                numperturbation,
                maxperturbations,
            )
        end
    end
end

function area_quadrature(
    func,
    grad,
    sign_condition,
    box,
    quad1d;
    maxlevels = 5,
    perturbation = 1e-2,
    maxperturbations = 2,
)

    return subdivision_area_quadrature(
        func,
        grad,
        sign_condition,
        box,
        quad1d,
        1,
        maxlevels,
        perturbation,
        0,
        maxperturbations,
    )
end
