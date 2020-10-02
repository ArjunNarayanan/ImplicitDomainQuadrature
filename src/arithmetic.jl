
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
