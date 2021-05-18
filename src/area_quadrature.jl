function area(box::IntervalBox{2,T}) where {T}
    return prod([box[i].hi - box[i].lo for i = 1:2])
end

function extend_edge_quadrature_to_area_quadrature(
    funcs,
    sign_conditions,
    height_dir,
    lo,
    hi,
    x0::V,
    w0,
    quad1d::ReferenceQuadratureRule{T},
) where {T,V<:AbstractVector}

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

function extend_edge_quadrature_to_area_quadrature(
    funcs,
    sign_conditions,
    height_dir,
    lo,
    hi,
    x0::M,
    w0::V,
    quad1d::ReferenceQuadratureRule{T},
) where {T,V<:AbstractVector,M<:AbstractMatrix}

    @assert length(funcs) == length(sign_conditions)
    lower_dim, npoints = size(x0)
    nweights = length(w0)
    @assert npoints == nweights

    dim = lower_dim + 1
    quad = TemporaryQuadrature(T, dim)

    for i = 1:npoints
        x0p = view(x0, :, i)
        w0p = w0[i]
        lq = extend_edge_quadrature_to_area_quadrature(
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

function extend_edge_quadrature_to_area_quadrature(
    funcs,
    sign_conditions,
    height_dir,
    interval,
    edgequad,
    quad1d,
)

    return extend_edge_quadrature_to_area_quadrature(
        funcs,
        sign_conditions,
        height_dir,
        interval.lo,
        interval.hi,
        edgequad.points,
        edgequad.weights,
        quad1d,
    )
end

function cut_area_quadrature(
    func,
    grad,
    sign_condition,
    box,
    quad1d,
    recursionlevel,
    maxlevels,
    numsplits,
)
    heightdir = height_direction(grad, box)
    issuitable, gradsign = is_suitable(heightdir, grad, box)
    if !issuitable
        splitboxes = split_box(box, numsplits)
        splitquads = [
            subdivision_area_quadrature(
                func,
                grad,
                sign_condition,
                sb,
                quad1d,
                recursionlevel + 1,
                maxlevels,
                numsplits,
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

        return extend_edge_quadrature_to_area_quadrature(
            [func],
            [sign_condition],
            heightdir,
            box[heightdir],
            edgequad,
            quad1d,
        )
    end
end


function subdivision_area_quadrature(
    func,
    grad,
    sign_condition,
    box,
    quad1d::ReferenceQuadratureRule{T},
    recursionlevel,
    maxlevels,
    numsplits,
) where {T}

    @assert sign_condition == +1 || sign_condition == -1

    if recursionlevel >= maxlevels
        error("Failed to construct an appropriate area integration rule after $maxlevels subdivisions")
        # xc = reshape(mid(box), 2, 1)
        # wt = [area(box)]
        # return TemporaryQuadrature(xc, wt)
    else
        s = sign(func, box)
        if s == +1 || s == -1
            if s != sign_condition
                return TemporaryQuadrature(T, 2)
            else
                return temporary_tensor_product(quad1d, box)
            end
        elseif s == 0
            return cut_area_quadrature(
                func,
                grad,
                sign_condition,
                box,
                quad1d,
                recursionlevel,
                maxlevels,
                numsplits,
            )
        else
            splitboxes = split_box(box, numsplits)
            splitquads = [
                subdivision_area_quadrature(
                    func,
                    grad,
                    sign_condition,
                    sb,
                    quad1d,
                    recursionlevel + 1,
                    maxlevels,
                    numsplits,
                ) for sb in splitboxes
            ]
            return combine_quadratures(splitquads)
        end
    end
end

function area_quadrature(
    func,
    grad,
    sign_condition,
    xL,
    xR,
    numqp;
    maxlevels = 5,
    numsplits = 2,
)

    box = IntervalBox(xL, xR)
    quad1d = ReferenceQuadratureRule(numqp)

    tempquad = subdivision_area_quadrature(
        func,
        grad,
        sign_condition,
        box,
        quad1d,
        0,
        maxlevels,
        numsplits,
    )
    return QuadratureRule(tempquad)
end

# function area_quadrature(
#     poly::InterpolatingPolynomial{1},
#     interpgrad,
#     sign_condition,
#     xL,
#     xR,
#     numqp;
#     maxlevels = 5,
#     numsplits = 2,
# ) where {NF,B,T}
#
#     update_interpolating_gradient!(interpgrad,poly)
#     return area_quadrature(
#         poly,
#         interpgrad,
#         sign_condition,
#         xL,
#         xR,
#         numqp,
#         maxlevels = maxlevels,
#         numsplits = numsplits,
#     )
# end
