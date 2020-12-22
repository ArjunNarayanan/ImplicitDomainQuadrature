function corners(box::IntervalBox{2,T}) where {T}
    xL = [box[1].lo, box[2].lo]
    xR = [box[1].hi, box[2].hi]
    return xL, xR
end

function split_box(box::IntervalBox{1,T}) where {T}
    return bisect(box, 0.5)
end

function split_box(box::IntervalBox{2,T}) where {T}
    xL, xR = corners(box)
    xM = 0.5 * (xL + xR)
    b1 = IntervalBox(xL, xM)
    b2 = IntervalBox([xM[1], xL[2]], [xR[1], xM[2]])
    b3 = IntervalBox(xM, xR)
    b4 = IntervalBox([xL[1], xM[2]], [xM[1], xR[2]])
    return b1, b2, b3, b4
end

function min_diam(box)
    return minimum(diam.(box))
end

function interval_arithmetic_sign_search(func, initialbox, tol, perturbation)

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
                newboxes = split_box(box)
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


function Base.sign(func, box; tol = 1e-3, perturbation = 0.0)
    return interval_arithmetic_sign_search(func, box, tol, perturbation)
end
"""
    sign(f, box)
return
- `+1` if `f` is uniformly positive on `int`
- `-1` if `f` is uniformly negative on `int`
- `0` if `f` has at least one zero crossing in `int` (f assumed continuous)
"""
function Base.sign(func, xL, xR; tol = 1e-3, perturbation = 0.0)
    box = IntervalBox(xL, xR)
    return sign(func, box; tol = tol, perturbation = perturbation)
end

function extremal_coeffs_in_box(poly, xL, xR)

    max_coeff = -Inf
    min_coeff = Inf
    points = poly.basis.points
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


function Base.sign(P::InterpolatingPolynomial{1}, xL, xR; tol = 1e-3, perturbation = 0.0)

    max_coeff, min_coeff = extremal_coeffs_in_box(P, xL, xR)
    if max_coeff > 0 && min_coeff < 0
        return 0
    else
        box = IntervalBox(xL, xR)
        return interval_arithmetic_sign_search(P, box, tol, perturbation)
    end
end


function (IP::InterpolatingPolynomial{N,B,T})(box::IntervalBox{2,S}) where {N,B,T,S}
    @assert PolynomialBasis.dimension(IP) == 2
    return IP(box[1], box[2])
end

function PolynomialBasis.gradient(
    IP::InterpolatingPolynomial{1,B,T},
    box::IntervalBox{2,S},
) where {B,T,S}
    return gradient(IP, box[1], box[2])
end
