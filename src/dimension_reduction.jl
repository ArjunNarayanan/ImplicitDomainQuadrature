function checkUniqueRoots(_roots::Vector{Root{Interval{T}}}) where {T}
    for r in _roots
        if r.status != :unique
            msg = "Intervals with possibly non-unique roots were found.\nPossible Fix: Adjust the discretization."
            throw(ErrorException(msg))
        end
    end
end

"""
    unique_root_intervals(f, x1::T, x2::T) where {T<:Real}
returns sub-intervals in `[x1,x2]` which contain unique roots of `f`
"""
function unique_root_intervals(f, x1::T, x2::T) where {T<:Real}
    all_roots = roots(f, Interval(x1, x2))
    checkUniqueRoots(all_roots)
    return [r.interval for r in all_roots]
end
