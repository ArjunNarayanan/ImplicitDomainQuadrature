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
        # _roots = unique_roots(f, x1, x2)
        _roots = find_zeros(f, x1, x2)
        append!(r, _roots)
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

function is_suitable(
    height_dir,
    grad,
    box;
    order = 5,
    tol = 1e-3,
    C = 4,
    perturbation = 1e-2,
    maxperturbations = 5,
)

    if height_dir == 0
        return false, 0
    else
        gradk(x) = grad(x)[height_dir]
        s = sign_allow_perturbations(
            gradk,
            box,
            tol = tol,
            perturbation = perturbation,
            maxperturbations = 5,
        )
        curvatureflag = true
        if s != 0
            curve(x) = curvature_measure(grad(x), height_dir, C)
            t = sign(curve, box, tol = tol)
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
