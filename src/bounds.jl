"""
    SignSearch{N,T} <: AbstractBreadthFirstSearch{IntervalBox{N,T}}
a `BranchAndPrune` search type that is used to determine the sign of
a function over a given interval
"""
mutable struct SignSearch{N,T} <: AbstractBreadthFirstSearch{IntervalBox{N,T}}
    f
    initial::IntervalBox{N,T}
    algorithm::Symbol
    tol::Real
    found_positive::Bool
    found_negative::Bool
    breached_tolerance::Bool
    order::Int
    function SignSearch(f, initial::IntervalBox{N,T},
        algorithm::Symbol, tol::Real, order::Int) where {N,T}

        if tol <= 0
            throw(ArgumentError("Tolerance must be a positive number, got $tol"))
        end
        if order < 1
            throw(ArgumentError("Order must be > 0, got $order"))
        end
        relative_tolerance = tol*diam(initial)
        new{N,T}(f,initial,algorithm,relative_tolerance,false,false,false,order)
    end
end

"""
    min_diam(box::IntervalBox)
return the minimum diameter of `box`
"""
function min_diam(box::IntervalBox)
    return minimum(diam.(box))
end

"""
    Base.muladd(tm::TaylorModelN, a::Number, b::Number)
overloading `muladd` to avoid an error
"""
function Base.muladd(tm::TaylorModelN, a::Number, b::Number)
    return a*tm+b
end

function Base.muladd(a::Number, tm::TaylorModelN, b::Number)
    return a*tm+b
end

function Base.muladd(a::TaylorModelN, tm::TaylorModelN, b::Number)
    return a*tm+b
end

"""
    BranchAndPrune.process(search::SignSearch, interval::IntervalBox)
process the given `interval` to determine the sign of `search.f` in this interval
"""
function BranchAndPrune.process(search::SignSearch, interval::IntervalBox)

    f_range = enclose((x...) -> search.f(x...), interval, search.algorithm, order = search.order)

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
function run_search(f, interval, algorithm, tol, order)

    search = SignSearch(f, interval, algorithm, tol, order)
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
function Base.sign(f, int::IntervalBox; algorithm = :TaylorModels, tol = 1e-3, order = 5)
    tree, search = run_search(f,int,algorithm,tol,order)
    if search.found_positive && search.found_negative
        return 0
    elseif search.breached_tolerance
        error("Search tolerance not tight enough")
    elseif search.found_positive
        return 1
    else
        return -1
    end
end

function extremal_coeffs_in_box(P::InterpolatingPolynomial{1}, box::IntervalBox)

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
function Base.sign(P::InterpolatingPolynomial{1}, int::IntervalBox; algorithm = :TaylorModels, tol = 1e-3, order = 5)

    max_coeff, min_coeff = extremal_coeffs_in_box(P,int)
    if !isinf(max_coeff) && !isinf(min_coeff) && max_coeff > 0 && min_coeff < 0
        return 0
    else
        return sign((x...) -> P(x...), int, algorithm = algorithm, tol = tol, order = order)
    end
end
