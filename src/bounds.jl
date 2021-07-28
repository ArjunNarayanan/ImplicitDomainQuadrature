function corners(box::IntervalBox{2,T}) where {T}
    xL = [box[1].lo, box[2].lo]
    xR = [box[1].hi, box[2].hi]
    return xL, xR
end

function split_box(box, numsplits)
    return mince(box, numsplits)
end

function min_diam(box)
    return minimum(diam.(box))
end

function interval_arithmetic_sign_search(
    func,
    initialbox,
    tol,
    perturbation,
    numsplits,
)

    rtol = tol * diam(initialbox)

    foundpos = foundneg = breachedtol = false
    queue = [initialbox]

    while !isempty(queue)
        if (foundpos && foundneg)
            break
        else
            box = popfirst!(queue)
            if min_diam(box) < rtol
                breachedtol = true
                break
            end
            funcrange = func(box)
            if inf(funcrange) > -perturbation
                foundpos = true
            elseif sup(funcrange) < perturbation
                foundneg = true
            else
                newboxes = split_box(box, numsplits)
                push!(queue, newboxes...)
            end
        end
    end

    if breachedtol
        return 2
    elseif foundpos && foundneg
        return 0
    elseif foundpos && !foundneg
        return +1
    elseif !foundpos && foundneg
        return -1
    else
        error("Unexpected scenario")
    end
end


function Base.sign(func, box; tol = 1e-3, perturbation = 0.0, numsplits = 2)
    return interval_arithmetic_sign_search(
        func,
        box,
        tol,
        perturbation,
        numsplits,
    )
end
"""
    sign(f, box)
return
- `+1` if `f` is uniformly positive on `int`
- `-1` if `f` is uniformly negative on `int`
- `0` if `f` has at least one zero crossing in `int` (f assumed continuous)
"""
function Base.sign(func, xL, xR; tol = 1e-3, perturbation = 0.0, numsplits = 2)
    box = IntervalBox(xL, xR)
    return sign(
        func,
        box;
        tol = tol,
        perturbation = perturbation,
        numsplits = numsplits,
    )
end

function extremal_coeffs_in_box(poly, xL, xR)

    max_coeff = -Inf
    min_coeff = Inf
    points = interpolation_points(basis(poly))
    dim, npoints = size(points)
    for i = 1:npoints
        p = view(points, :, i)
        if all(xL .<= p .<= xR)
            coeff = poly.coeffs[i]
            min_coeff = min(min_coeff, coeff)
            max_coeff = max(max_coeff, coeff)
        end
    end
    return max_coeff, min_coeff
end


function Base.sign(
    P::InterpolatingPolynomial{1,B},
    xL,
    xR;
    tol = 1e-3,
    perturbation = 0.0,
    numsplits = 2,
) where {B<:LagrangeTensorProductBasis}

    max_coeff, min_coeff = extremal_coeffs_in_box(P, xL, xR)
    if max_coeff > 0 && min_coeff < 0
        return 0
    else
        box = IntervalBox(xL, xR)
        return interval_arithmetic_sign_search(
            P,
            box,
            tol,
            perturbation,
            numsplits,
        )
    end
end


function (IP::InterpolatingPolynomial{N,B,T})(
    box::IntervalBox{2,S},
) where {N,B,T,S}
    @assert PolynomialBasis.dimension(IP) == 2
    return IP(box[1], box[2])
end

function PolynomialBasis.gradient(
    IP::InterpolatingPolynomial{1,B,T},
    box::IntervalBox{2,S},
) where {B,T,S}
    @assert PolynomialBasis.dimension(IP) == 2
    return gradient(IP, box[1], box[2])
end

function (IP::InterpolatingPolynomial{N,B,T})(
    box::IntervalBox{1,S},
) where {N,B,T,S}
    @assert PolynomialBasis.dimension(IP) == 1
    return IP(box[1])
end

function PolynomialBasis.gradient(
    IP::InterpolatingPolynomial{1,B,T},
    box::IntervalBox{1,S},
) where {B,T,S}
    @assert PolynomialBasis.dimension(IP) == 1
    return gradient(IP, box[1])
end
