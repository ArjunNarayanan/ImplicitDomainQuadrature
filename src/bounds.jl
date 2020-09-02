"""
    SignSearch{N,T} <: AbstractBreadthFirstSearch{IntervalBox{N,T}}
a `BranchAndPrune` search type that is used to determine the sign of
a function over a given interval
"""
mutable struct SignSearch{N,T} <: AbstractBreadthFirstSearch{IntervalBox{N,T}}
    f
    initial::IntervalBox{N,T}
    tol::T
    found_positive::Bool
    found_negative::Bool
    breached_tolerance::Bool
    order::Int
    function SignSearch(f, initial::IntervalBox{N,T}, order::Int, tol::Real) where {N,T}

        if tol <= 0
            throw(ArgumentError("Tolerance must be a positive number, got $tol"))
        end
        if order < 1
            throw(ArgumentError("Order must be > 0, got $order"))
        end
        set_variables(T, "x", order = 2order, numvars = N)
        relative_tolerance = tol*diam(initial)
        new{N,T}(f,initial,relative_tolerance,false,false,false,order)
    end
end

"""
    min_diam(box::IntervalBox)
return the minimum diameter of `box`
"""
function min_diam(box::IntervalBox)
    return minimum(diam.(box))
end

function Base.muladd(a::TaylorModelN{N,Interval{T},T}, b::T, c::T) where {N,T<:Real}
    return a*b + c
end

function Base.muladd(a::T, b::TaylorModelN{N,Interval{T},T}, c::T) where {N,T<:Real}
    return a*b + c
end

function Base.muladd(a::T, b::T, c::TaylorModelN{N,Interval{T},T}) where {N,T<:Real}
    return a*b + c
end

function Base.muladd(a::TaylorModelN{N,Interval{T},T}, b::TaylorModelN{N,Interval{T},T}, c::T) where {N,T<:Real}
    return a*b + c
end

function Base.muladd(a::TaylorModelN{N,Interval{T},T}, b::T, c::TaylorModelN{N,Interval{T},T}) where {N,T<:Real}
    return a*b + c
end

function Base.muladd(a::T, b::TaylorModelN{N,Interval{T},T}, c::TaylorModelN{N,Interval{T},T}) where {N,T<:Real}
    return a*b + c
end

function Base.muladd(a::TaylorModelN{N,Interval{T},T}, b::TaylorModelN{N,Interval{T},T}, c::TaylorModelN{N,Interval{T},T}) where {N,T<:Real}
    return a*b + c
end

@inline zeroBox(N) = IntervalBox(0..0, N)
@inline symBox(N) = IntervalBox(-1..1, N)

function normalizedTaylorN(order::Z, box::IntervalBox{N}) where {Z<:Integer,N}
    zBoxN = zeroBox(N)
    sBoxN = symBox(N)
    x0 = mid(box)

    x = [TaylorModelN(i, order, IntervalBox(x0), box) for i=1:N]
    xnorm = [normalize_taylor(xi.pol, box - x0, true) for xi in x]
    return [TaylorModelN(xi_norm, 0..0, zBoxN, sBoxN) for xi_norm in xnorm], sBoxN
end

function bound(f, box::IntervalBox, order::Z) where {Z<:Integer}
    tm, sBoxN = normalizedTaylorN(order, box)
    ftm = f(tm...)
    return evaluate(ftm, sBoxN)
end

"""
    BranchAndPrune.process(search::SignSearch, interval::IntervalBox)
process the given `interval` to determine the sign of `search.f` in this interval
"""
function BranchAndPrune.process(search::SignSearch, interval::IntervalBox)

    f_range = bound(search.f, interval, search.order)

    if (search.found_positive && search.found_negative) || (search.breached_tolerance)
        return :discard, interval
    elseif inf(f_range) > 0.0
        search.found_positive = true
        return :store, interval
    elseif sup(f_range) < 0.0
        search.found_negative = true
        return :store, interval
    elseif min_diam(interval) < search.tol
        search.breached_tolerance = true
        return :discard, interval
    else
        return :bisect, interval
    end
end

"""
    BranchAndPrune.bisect(::SignSearch, interval::IntervalBox)
split the given `interval` along its largest dimension
"""
function BranchAndPrune.bisect(::SignSearch, interval::IntervalBox)
    idx = argmax(diam.(interval))
    return bisect(interval,idx,0.5)
end

"""
    run_search(f, interval, algorithm, tol, order)
run a `BranchAndPrune` search until the sign of `f` in the given interval is determined
"""
function run_search(f,interval::IntervalBox,order,tol)

    search = SignSearch(f, interval, order, tol)
    local endtree = nothing
    for working_tree in search
        endtree = working_tree
    end
    return endtree, search
end

"""
    sign(f, int::Interval)
return
- `+1` if `f` is uniformly positive on `int`
- `-1` if `f` is uniformly negative on `int`
- `0` if `f` has at least one zero crossing in `int` (f assumed continuous)
"""
function Base.sign(f,int::IntervalBox; order = 5, tol = 1e-2)
    tree, search = run_search(f,int,order,tol)
    if search.found_positive && search.found_negative
        return 0
    elseif search.breached_tolerance
        throw(ArgumentError("Bisection reduced interval size below tolerance. Reduce tolerance or increase the order of the method"))
    elseif search.found_positive
        return 1
    else
        return -1
    end
end

function extremal_coeffs_in_box(P::InterpolatingPolynomial{1},box::IntervalBox)

    max_coeff = -Inf
    min_coeff = Inf
    points = P.basis.points
    dim,npoints = size(points)
    for i in 1:npoints
        p = view(points,:,i)
        if p in box
            coeff = P.coeffs[i]
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
function Base.sign(P::InterpolatingPolynomial{1},
    int::IntervalBox; tol = 1e-2, order = 5)

    max_coeff, min_coeff = extremal_coeffs_in_box(P,int)
    if !isinf(max_coeff) && !isinf(min_coeff) && max_coeff > 0 && min_coeff < 0
        return 0
    else
        return sign((x...) -> P(x...), int, order = order, tol = tol)
    end
end
