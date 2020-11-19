function extend_edge_quadrature_to_surface_quadrature(
    func,
    grad,
    height_dir,
    lo,
    hi,
    x0::V,
    w0,
) where {V<:AbstractVector{T}} where {T}

    extended_func(x) = func(extend(x0, height_dir, x))
    _roots = find_zeros(extended_func, lo, hi)

    num_roots = length(_roots)
    @assert num_roots == 1 || num_roots == 0

    if num_roots == 1
        root = _roots[1]
        p = extend(x0, height_dir, root)
        gradF = grad(p)
        jac = norm(gradF) / (abs(gradF[height_dir]))
        w = w0 * jac
        return p, w
    else
        return nothing,nothing
    end
end

function extend_edge_quadrature_to_surface_quadrature(
    func,
    grad,
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
        qp, qw = extend_edge_quadrature_to_surface_quadrature(
            func,
            grad,
            height_dir,
            lo,
            hi,
            x0p,
            w0p,
        )
        if !isnothing(qp)
            update!(quad, qp, qw)
        end
    end
    return quad
end

function extend_edge_quadrature_to_surface_quadrature(
    func,
    grad,
    height_dir,
    interval,
    edgequad,
)

    return extend_edge_quadrature_to_surface_quadrature(
        func,
        grad,
        height_dir,
        interval.lo,
        interval.hi,
        edgequad.points,
        edgequad.weights,
    )
end

function cut_surface_quadrature(
    func,
    grad,
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
            subdivision_surface_quadrature(
                func,
                grad,
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

        lowersign = sign(gradsign, +1, true, -1)
        uppersign = sign(gradsign, +1, true, +1)

        lowerbox = heightdir == 1 ? box[2] : box[1]

        edgequad = one_dimensional_quadrature(
            [lower, upper],
            [lowersign, uppersign],
            lowerbox,
            quad1d,
        )

        return extend_edge_quadrature_to_surface_quadrature(
            func,
            grad,
            heightdir,
            box[heightdir],
            edgequad,
        )
    end
end

function subdivision_surface_quadrature(
    func,
    grad,
    box,
    quad1d::ReferenceQuadratureRule{T},
    recursionlevel,
    maxlevels,
    perturbation,
    numperturbation,
    maxperturbations,
) where {T}

    if recursionlevel >= maxlevels
        error("Failed to construct an appropriate surface integration rule")
    else
        s = sign(func, box)
        if s == -1 || s == 0 || s == +1
            if s != 0
                return TemporaryQuadrature(T, 2)
            else
                return cut_surface_quadrature(
                    func,
                    grad,
                    box,
                    quad1d,
                    recursionlevel,
                    maxlevels,
                    perturbation,
                    numperturbation,
                    maxperturbations,
                )
            end
        else
            if numperturbation >= maxperturbations
                errorstr = "Failed to construct surface quadrature after $numperturbation perturbations of size $perturbation"
                error(errorstr)
            else
                perturbedfunc(x) = func(x) + perturbation
                return subdivision_surface_quadrature(
                    perturbedfunc,
                    grad,
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
    end
end

function surface_quadrature(
    func,
    grad,
    xL,
    xR,
    quad1d;
    maxlevels = 5,
    perturbation = 1e-2,
    maxperturbations = 2,
)

    box = IntervalBox(xL,xR)
    tempquad = subdivision_surface_quadrature(
        func,
        grad,
        box,
        quad1d,
        1,
        maxlevels,
        perturbation,
        0,
        maxperturbations,
    )
    return QuadratureRule(tempquad)
end

function surface_quadrature(
    poly::InterpolatingPolynomial,
    xL,
    xR,
    quad1d;
    maxlevels = 5,
    perturbation = 1e-2,
    maxperturbations = 2,
)

    interpgrad = interpolating_gradient(poly)
    return surface_quadrature(
        poly,
        interpgrad,
        xL,
        xR,
        quad1d,
        maxlevels = maxlevels,
        perturbation = perturbation,
        maxperturbations = maxperturbations,
    )
end
