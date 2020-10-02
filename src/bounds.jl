function corners(box::IntervalBox{2,T}) where {T}
    xL = [box[1].lo,box[2].lo]
    xR = [box[1].hi,box[2].hi]
    return xL,xR
end

function split_box(box::IntervalBox{1,T}) where {T}
    return bisect(box,0.5)
end

function split_box(box::IntervalBox{2,T}) where {T}
    xL,xR = corners(box)
    xM = 0.5*(xL+xR)
    b1 = IntervalBox(xL,xM)
    b2 = IntervalBox([xM[1],xL[2]],[xR[1],xM[2]])
    b3 = IntervalBox(xM,xR)
    b4 = IntervalBox([xL[1],xM[2]],[xM[1],xR[2]])
    return b1,b2,b3,b4
end

function min_diam(box)
    return minimum(diam.(box))
end

@inline zeroBox(N) = IntervalBox(0..0, N)
@inline symBox(N) = IntervalBox(-1..1, N)

function normalizedTaylorN(order, box::IntervalBox{N,T}) where {N,T}
    zBoxN = zeroBox(N)
    sBoxN = symBox(N)
    x0 = mid(box)

    x = [TaylorModelN(i, order, IntervalBox(x0), box) for i=1:N]
    xnorm = [normalize_taylor(xi.pol, box - x0, true) for xi in x]
    return [TaylorModelN(xi_norm, 0..0, zBoxN, sBoxN) for xi_norm in xnorm], sBoxN
end

function bound(f, box, order)
    tm, sBoxN = normalizedTaylorN(order, box)
    ftm = f(tm)
    return evaluate(ftm, sBoxN)
end

function taylor_models_sign_search(func,initialbox::IntervalBox{N,T},order,tol) where {N,T}

    set_variables(T, "x", order = 2order, numvars = N)
    rtol = tol*diam(initialbox)

    foundpos = foundneg = breachedtol = false
    queue = [initialbox]

    counter = 1
    while !isempty(queue)
        if (foundpos && foundneg)
            break
        else
            box = popfirst!(queue)
            if min_diam(box) < rtol
                breachedtol = true
                break
            end
            funcrange = bound(func,box,order)
            if inf(funcrange) > 0.0
                foundpos = true
            elseif sup(funcrange) < 0.0
                foundneg = true
            else
                newboxes = split_box(box)
                push!(queue,newboxes...)
            end
            counter += 1
        end
    end

    if breachedtol
        error("Bisection reduced interval size below tolerance.")
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

"""
    sign(f, box)
return
- `+1` if `f` is uniformly positive on `int`
- `-1` if `f` is uniformly negative on `int`
- `0` if `f` has at least one zero crossing in `int` (f assumed continuous)
"""
function Base.sign(func,box;order = 5, tol = 1e-3)
    return taylor_models_sign_search(func,box,order,tol)
end

function extremal_coeffs_in_box(poly,box)

    max_coeff = -Inf
    min_coeff = Inf
    points = poly.basis.points
    dim,npoints = size(points)
    for i in 1:npoints
        p = view(points,:,i)
        if p in box
            coeff = poly.coeffs[i]
            min_coeff = min(min_coeff,coeff)
            max_coeff = max(max_coeff,coeff)
        end
    end
    return max_coeff, min_coeff
end

"""
    Base.sign(P::InterpolatingPolynomial{1}, int::IntervalBox; algorithm = :TaylorModels, tol = 1e-3, order = 5)
special function for an interpolating polynomial type.
"""
function Base.sign(P::InterpolatingPolynomial{1},int; tol = 1e-2, order = 5)

    max_coeff, min_coeff = extremal_coeffs_in_box(P,int)
    if max_coeff > 0 && min_coeff < 0
        return 0
    else
        return sign((x...) -> P(x...), int, order = order, tol = tol)
    end
end
