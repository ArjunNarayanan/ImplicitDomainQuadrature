function sign_conditions_satisfied(funcs, xc, sign_conditions)
    return all(i -> funcs[i](xc) * sign_conditions[i] >= 0.0, 1:length(funcs))
end

function one_dimensional_quadrature(
    funcs,
    sign_conditions,
    lo,
    hi,
    quad1d::ReferenceQuadratureRule{T},
) where {T}

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
