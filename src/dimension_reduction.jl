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

"""
    unique_roots(f, x1::T, x2::T) where {T<:Real}
returns the unique roots of `f` in the interval `(x1, x2)`
"""
function unique_roots(f, x1::T, x2::T) where {T<:Real}
    root_intervals = unique_root_intervals(f, x1, x2)
    _roots = zeros(T, length(root_intervals))
    for (idx,int) in enumerate(root_intervals)
        _roots[idx] = find_zero(f, (int.lo, int.hi), Order1())
    end
    return _roots
end

"""
    roots_and_ends(F::Vector, x1::T, x2::T)
return a sorted array containing `[x1, <roots of each f in F>, x2]`
"""
function roots_and_ends(F::Vector, x1::T, x2::T) where {T<:Real}
    r = [x1,x2]
    for f in F
        roots = unique_roots(f, x1, x2)
        append!(r,roots)
    end
    return sort!(r)
end

"""
    extend(x0::Number, k::Int, x::Number)
extend `x0` into 2D space by inserting `x` in direction `k`
    extend(x0::AbstractVector, k::Int, x::Number)
extend the point vector `x0` into `d+1` dimensional space by using `x` as the
`k`th coordinate value in the new point vector
    extend(x0::AbstractVector, k::Int, x::AbstractMatrix)
extend the point vector `x0` along direction `k` treating `x` as the new
coordinate values
"""
function extend(x0::Number, k::Int, x::Number)
    if k == 1
        return [x, x0]
    elseif k == 2
        return [x0, x]
    else
        throw(ArgumentError("Expected k ∈ {1,2}, got $k"))
    end
end

function extend(x0::AbstractVector, k::Int, x::Number)
    dim = length(x0)
    if dim == 1
        return extend(x0[1],k,x)
    else
        throw(ArgumentError("Current support for dim = 1, got dim = $dim"))
    end
end

function extend(x0::AbstractVector, k::Int, x::AbstractMatrix)
    old_dim = length(x0)
    dim, npoints = size(x)
    if dim != 1
        msg = "Extension can only be performed one dimension at a time, got dim = $dim"
        throw(ArgumentError(msg))
    end

    old_row = repeat(x0, inner = (1,npoints))
    if old_dim == 1 && k == 1
        return vcat(x, old_row)
    elseif old_dim == 1 && k == 2
        return vcat(old_row, x)
    else
        throw(ArgumentError("Current support for dim = 1 and k ∈ {1,2}, got dim = $dim, k = $k"))
    end
end

"""
    signConditionsSatisfied(funcs,xc,sign_conditions)
return true if each `f` in `funcs` evaluated at `xc` has the sign specified in
`sign_conditions`.
Note: if `sign_conditions[i] == 0` then the function always returns `true`.
"""
function signConditionsSatisfied(funcs,xc,sign_conditions)
    return all(i -> funcs[i](xc)*sign_conditions[i] >= 0.0, 1:length(funcs))
end


function check_num_funcs_conds(nfuncs,nconds)
    if nfuncs != nconds
        msg = "Require number of functions same as number of sign conditions, got $nfuncs != $nconds"
        throw(DimensionMismatch(msg))
    end
end
"""
    quadrature(F::Vector, sign_conditions::Vector{Int}, lo::T, hi::T, quad1d::ReferenceQuadratureRule{N,T}) where {N,T}
return a 1D quadrature rule that can be used to integrate in the domain where
each `f in F` has sign specified in `sign_conditions` in the interval `[lo,hi]`
    quadrature(F::Vector, sign_conditions::Vector{Int}, int::Interval{T}, quad1d::ReferenceQuadratureRule{N,T}) where {N,T}
conveniance function for when the interval over which the quadrature rule is required is specified by an `Interval` type
    quadrature(F::Vector, sign_conditions::Vector{Int}, height_dir::Int, lo::T, hi::T, x0::AbstractVector{T}, w0::T, quad1d::ReferenceQuadratureRule{N,T}) where {N,T}
return a tuple `points,weights` representing quadrature points and weights obtained by transforming `quad1d` into an interval bounded by `x0` and the
zero level set of `F` by extending along `height_dir`
    quadrature(F::Vector, sign_conditions::Vector{Int}, height_dir::Int, lo::T, hi::T, x0::AbstractMatrix, w0::AbstractVector, quad1d::ReferenceQuadratureRule{N,T}) where {N,T}
"""
function quadrature(F::Vector, sign_conditions::Vector{Int}, lo::T, hi::T, quad1d::ReferenceQuadratureRule{N,T}) where {N,T}
    nfuncs = length(F)
    nconds = length(sign_conditions)
    check_num_funcs_conds(nfuncs,nconds)

    quad = QuadratureRule(T,1)

    roots = roots_and_ends(F,lo,hi)
    for j in 1:length(roots)-1
        xc = 0.5*(roots[j] + roots[j+1])
        if signConditionsSatisfied(F,xc,sign_conditions)
            p,w = transform(quad1d, roots[j], roots[j+1])
            update!(quad, p, w)
        end
    end
    return quad
end

function quadrature(F::Vector, sign_conditions::Vector{Int}, int::Interval{T}, quad1d::ReferenceQuadratureRule{N,T}) where {N,T}
    return quadrature(F, sign_conditions, int.lo, int.hi, quad1d)
end

function quadrature(F::Vector, sign_conditions::Vector{Int}, height_dir::Int, lo::T, hi::T, x0::AbstractVector{T}, w0::T, quad1d::ReferenceQuadratureRule{N,T}) where {N,T}

    @assert length(F) == length(sign_conditions)


    lower_dim = length(x0)
    dim = lower_dim + 1
    points = Matrix{T}(undef, dim, 0)
    weights = Vector{T}(undef, 0)

    extended_funcs = [x -> f(extend(x0,height_dir,x)) for f in F]
    roots = roots_and_ends(extended_funcs,lo,hi)
    for j in 1:length(roots)-1
        xc = 0.5*(roots[j] + roots[j+1])
        if signConditionsSatisfied(extended_funcs,xc,sign_conditions)
            p,w = transform(quad1d, roots[j], roots[j+1])
            extended_points = extend(x0,height_dir,p)
            points = hcat(points, extended_points)
            append!(weights, w0*w)
        end
    end
    return points, weights
end

function quadrature(F::Vector, sign_conditions::Vector{Int}, height_dir::Int, int::Interval{T}, x0::AbstractVector{T}, w0::T, quad1d::ReferenceQuadratureRule{N,T}) where {N,T}
    return quadrature(F,sign_conditions,height_dir,int.lo,int.hi,x0,w0,quad1d)
end

function quadrature(F::Vector, sign_conditions::Vector{Int}, height_dir::Int, lo::T, hi::T, x0::AbstractMatrix{T}, w0::AbstractVector{T}, quad1d::ReferenceQuadratureRule{N,T}) where {N,T}

    lower_dim, npoints = size(x0)
    dim = lower_dim + 1
    quad = QuadratureRule(T, dim)

    for i in 1:npoints
        x0p = view(x0, :, i)
        w0p = w0[i]
        qp, qw = quadrature(F, sign_conditions, height_dir, lo, hi, x0p, w0p, quad1d)
        update!(quad, qp, qw)
    end
    return quad
end

function quadrature(F::Vector, sign_conditions::Vector{Int}, height_dir::Int, int::Interval{T}, x0::AbstractMatrix{T}, w0::AbstractVector{T}, quad1d::ReferenceQuadratureRule{N,T}) where {N,T}
    return quadrature(F,sign_conditions,height_dir,int.lo,int.hi,x0,w0,quad1d)
end

"""
    surface_quadrature(F::InterpolatingPolynomial, height_dir::Int, lo::T, hi::T, x0::AbstractVector{T}, w0::T) where {T}
return a `point,weight` pair representing a quadrature point and weight that is obtained by
projecting `x0` along `height_dir` onto the zero level set of `F` within the interval `(lo,hi)`.
`weight` is scaled appropriately by the curvature of `F`
    surface_quadrature(F::InterpolatingPolynomial, height_dir::Int, int::Interval{T}, x0::AbstractVector{T}, w0::T) where {T}
convenience function when the interval within which to project `x0` is defined by an `Interval` type
    surface_quadrature(F::InterpolatingPolynomial, height_dir::Int, lo::T, hi::T, x0::AbstractMatrix{T}, w0::AbstractVector{T}) where {T}
project each column of `x0` onto the zero level set of `F`
    surface_quadrature(F::InterpolatingPolynomial, height_dir::Int, int::Interval{T}, x0::AbstractMatrix{T}, w0::AbstractVector{T}) where {T}
    quadrature(P::InterpolatingPolynomial{N,NF,B,T}, sign_condition::Int, surface::Bool, box::IntervalBox{2,T}, quad1d::ReferenceQuadratureRule{NQ,T}) where {B<:TensorProductBasis{2}} where {N,NF,NQ,T}
corresponds to Algorithm 3 of Robert Saye's 2015 SIAM paper
"""
function surface_quadrature(F::InterpolatingPolynomial, height_dir::Int, lo::T, hi::T, x0::AbstractVector{T}, w0::T) where {T}

    extended_func(x) = F(extend(x0,height_dir,x))
    root = lo
    try
        root = find_zero(extended_func, (lo, hi), Order1())
    catch err
        msg = "Failed to find a root along given height direction. Check if the given function is bounded by its zero level-set along $height_dir from "*string(x0)
        error(msg)
    end
    p = extend(x0,height_dir,root)
    gradF = gradient(F, p)
    jac = norm(gradF)/(abs(gradF[height_dir]))
    w = w0*jac
    return p, w
end

function surface_quadrature(F::InterpolatingPolynomial, height_dir::Int, int::Interval{T}, x0::AbstractVector{T}, w0::T) where {T}
    return surface_quadrature(F,height_dir,int.lo,int.hi,x0,w0)
end

function surface_quadrature(F::InterpolatingPolynomial, height_dir::Int, lo::T, hi::T, x0::AbstractMatrix{T}, w0::AbstractVector{T}) where {T}

    lower_dim, npoints = size(x0)
    dim = lower_dim + 1
    points = zeros(T, dim, npoints)
    weights = zeros(T, npoints)

    for i in 1:npoints
        x0p = view(x0, :, i)
        w0p = w0[i]
        qp, qw = surface_quadrature(F, height_dir, lo, hi, x0p, w0p)
        points[:,i] = qp
        weights[i] = qw
    end

    quad = QuadratureRule(points,weights)
    return quad
end

function surface_quadrature(F::InterpolatingPolynomial, height_dir::Int, int::Interval{T}, x0::AbstractMatrix{T}, w0::AbstractVector{T}) where {T}
    return surface_quadrature(F,height_dir,int.lo,int.hi,x0,w0)
end

function height_direction(P::InterpolatingPolynomial, box::IntervalBox)
    g = abs.(gradient(P, mid(box)))
    k = LinearIndices(g)[argmax(g)]
    return k
end

"""
    isSuitable(height_dir::Int, F::AbstractVector, box::IntervalBox)
checks if the given `height_dir` is a suitable height direction for each `f` in `F` in the domain specified by `box`
"""
function isSuitable(height_dir::Int, P::InterpolatingPolynomial, box::IntervalBox;order::Int = 5, tol::Float64 = 1e-3)
    flag = true
    gradFk(x...) = gradient(P,height_dir,x...)
    s = sign(gradFk,box,order,tol)
    if s == 0
        flag = false
    end
    return flag, s
end

"""
    Base.sign(m::Int,s::Int,S::Bool,sigma::Int)
See sec. 3.2.4 of Robert Saye's 2015 SIAM paper
"""
function Base.sign(m::Int,s::Int,S::Bool,sigma::Int)
    if S || m == sigma*s
        return sigma*m
    else
        return 0
    end
end

function quadrature(P::InterpolatingPolynomial{N,NF,B,T}, sign_condition::Int, surface::Bool, box::IntervalBox{2,T}, quad1d::ReferenceQuadratureRule{NQ,T}) where {B<:TensorProductBasis{2}} where {N,NF,NQ,T}

    s = sign(P,box)
    if s*sign_condition < 0
        return QuadratureRule(T,2)
    elseif s*sign_condition > 0
        return tensorProduct(quad1d, box)
    else
        height_dir = height_direction(P, box)
        flag, gradient_sign = isSuitable(height_dir,P,box)
        if gradient_sign == 0
            error("Subdivision not implemented yet")
        else
            lower(x) = P(extend(x,height_dir,box[height_dir].lo))
            upper(x) = P(extend(x,height_dir,box[height_dir].hi))
            lower_sign = sign(gradient_sign,sign_condition,surface,-1)
            upper_sign = sign(gradient_sign,sign_condition,surface,+1)
            lower_box = height_dir == 1 ? box[2] : box[1]
            quad = quadrature([lower, upper], [lower_sign, upper_sign], lower_box, quad1d)
            if surface
                return surface_quadrature(P, height_dir, box[height_dir], quad.points, quad.weights)
            else
                return quadrature([P], [sign_condition], height_dir, box[height_dir], quad.points, quad.weights, quad1d)
            end
        end
    end
end
