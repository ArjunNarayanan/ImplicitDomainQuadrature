var documenterSearchIndex = {"docs":
[{"location":"#ImplicitDomainQuadrature.jl-1","page":"Home","title":"ImplicitDomainQuadrature.jl","text":"","category":"section"},{"location":"#","page":"Home","title":"Home","text":"","category":"page"},{"location":"#","page":"Home","title":"Home","text":"Modules = [ImplicitDomainQuadrature]","category":"page"},{"location":"#ImplicitDomainQuadrature.InterpolatingPolynomial","page":"Home","title":"ImplicitDomainQuadrature.InterpolatingPolynomial","text":"InterpolatingPolynomial{N,NFuncs,B<:AbstractBasis,T}\n\ninterpolate a VECTOR of N with a basis B composed of NFuncs functions\n\nFields\n\n- `coeffs::SMatrix{N,NFuncs,T}`\n- `basis::B`\n\n\n\n\n\n","category":"type"},{"location":"#ImplicitDomainQuadrature.InterpolatingPolynomial-Tuple","page":"Home","title":"ImplicitDomainQuadrature.InterpolatingPolynomial","text":"(P::InterpolatingPolynomial{1})(x...)\n\nevaluate P at x, the result is a scalar     (P::InterpolatingPolynomial)(x...) evaluate P at x     (P::InterpolatingPolynomial)(x::AbstractVector) evaluate P at the point vector x\n\n\n\n\n\n","category":"method"},{"location":"#ImplicitDomainQuadrature.InterpolatingPolynomial-Union{Tuple{NFuncs}, Tuple{Type{#s15} where #s15<:Number,Int64,ImplicitDomainQuadrature.AbstractBasis{NFuncs}}} where NFuncs","page":"Home","title":"ImplicitDomainQuadrature.InterpolatingPolynomial","text":"InterpolatingPolynomial(T::Type{<:Number}, N::Int,\n    basis::AbstractBasis{NFuncs}) where {NFuncs}\n\ninitialize an InterpolatingPolynomial{N,NFuncs,T} object with coefficients zeros(T,N,NFuncs)     InterpolatingPolynomial(N::Int, basis::AbstractBasis) initialize an InterpolatingPolynomial object with Float64 coefficients.     InterpolatingPolynomial(N::Int, dim::Int, order::Int, start::T, stop::T) where {T<:Real} initialize a dim dimensional basis of order order and pass this to the InterpolatingPolynomial constructor.     InterpolatingPolynomial(N::Int, dim::Int, order::Int; start = -1.0, stop = 1.0) initialize a dim dimensional basis of order order and pass this to the InterpolatingPolynomial constructor.\n\n\n\n\n\n","category":"method"},{"location":"#ImplicitDomainQuadrature.QuadratureRule","page":"Home","title":"ImplicitDomainQuadrature.QuadratureRule","text":"QuadratureRule{D,T}\n\na quadrature rule in D spatial dimensions. This is a mutable type; points and weights can be added via the update! method.\n\nInner Constructor:\n\nQuadratureRule(points::Matrix{T}, weights::Vector{T}) where {T<:Real}\n\nOuter Constructors:\n\nQuadratureRule(T::Type{<:Real}, dim::Int)\n\ninitializes (dim,0) matrix of points and (0) vector of weights. Both are of type T     QuadratureRule(dim::Int) initializes (dim,0) matrix of points and (0) vector of weights of type Float64\n\n\n\n\n\n","category":"type"},{"location":"#ImplicitDomainQuadrature.TensorProductBasis","page":"Home","title":"ImplicitDomainQuadrature.TensorProductBasis","text":"TensorProductBasis{N,NFuncs,T<:AbstractBasis1D} <: AbstractBasis{NFuncs}\n\nconstruct an N dimensional polynomial basis by tensor product of T with a total of NFuncs functions. Note that NFuncs can be inferred from T{nfuncs} as NFuncs = nfuncs^N.\n\nFields\n\n- `basis::T` the underlying 1D basis\n- `points::SMatrix{N,NFuncs}` a matrix of point vectors\n\n\n\n\n\n","category":"type"},{"location":"#ImplicitDomainQuadrature.TensorProductBasis-Union{Tuple{Int64,Int64}, Tuple{T}} where T<:Real","page":"Home","title":"ImplicitDomainQuadrature.TensorProductBasis","text":"TensorProductBasis(dim::Int, order::Int; start = -1.0, stop = 1.0)\n\nconstruct a dim dimensional polynomial basis with variable x using a tensor product of order order LagrangePolynomialBasis. The polynomials are equispaced from start to stop.\n\n\n\n\n\n","category":"method"},{"location":"#Base.:==-Union{Tuple{NF}, Tuple{ImplicitDomainQuadrature.LagrangePolynomialBasis{NF},ImplicitDomainQuadrature.LagrangePolynomialBasis{NF}}} where NF","page":"Home","title":"Base.:==","text":"==(b1::LagrangePolynomialBasis{NF},b2::LagrangePolynomialBasis{NF}) where {NF}\n\nReturns true if b1 and b2 have the same funcs and points, false otherwise.\n\nSame as isequal.\n\n\n\n\n\n","category":"method"},{"location":"#Base.:==-Union{Tuple{NF}, Tuple{T}, Tuple{D}, Tuple{TensorProductBasis{D,T,NF},TensorProductBasis{D,T,NF}}} where NF where T where D","page":"Home","title":"Base.:==","text":"==(tp1::TensorProductBasis{D,T,NF}, tp2::TensorProductBasis{D,T,NF}) where {D,T,NF}\n\nreturns true if tp1 and tp2 have identical basis and points, false otherwise.\n\n\n\n\n\n","category":"method"},{"location":"#Base.isequal-Union{Tuple{NF}, Tuple{ImplicitDomainQuadrature.LagrangePolynomialBasis{NF},ImplicitDomainQuadrature.LagrangePolynomialBasis{NF}}} where NF","page":"Home","title":"Base.isequal","text":"Base.isequal(b1::LagrangePolynomialBasis{NF},b2::LagrangePolynomialBasis{NF}) where {NF}\n\nReturns true if b1 and b2 have the same funcs and points, false otherwise.\n\nSame as ==.\n\n\n\n\n\n","category":"method"},{"location":"#Base.isequal-Union{Tuple{NF}, Tuple{T}, Tuple{D}, Tuple{TensorProductBasis{D,T,NF},TensorProductBasis{D,T,NF}}} where NF where T where D","page":"Home","title":"Base.isequal","text":"Base.isequal(tp1::TensorProductBasis{D,T,NF}, tp2::TensorProductBasis{D,T,NF}) where {D,T,NF}\n\nreturns true if tp1 and tp2 have identical basis and points, false otherwise.\n\n\n\n\n\n","category":"method"},{"location":"#Base.sign-Tuple{Any,IntervalArithmetic.IntervalBox,Int64,Float64}","page":"Home","title":"Base.sign","text":"sign(f, int::Interval)\n\nreturn\n\n+1 if f is uniformly positive on int\n-1 if f is uniformly negative on int\n0 if f has at least one zero crossing in int (f assumed continuous)\n\n\n\n\n\n","category":"method"},{"location":"#Base.sign-Tuple{Int64,Int64,Bool,Int64}","page":"Home","title":"Base.sign","text":"Base.sign(m::Int,s::Int,S::Bool,sigma::Int)\n\nSee sec. 3.2.4 of Robert Saye's 2015 SIAM paper\n\n\n\n\n\n","category":"method"},{"location":"#Base.sign-Union{Tuple{B}, Tuple{S}, Tuple{D}, Tuple{T}, Tuple{NF}, Tuple{InterpolatingPolynomial{1,NF,B,T},IntervalArithmetic.IntervalBox}} where B<:(TensorProductBasis{D,S,NF} where NF) where S<:ImplicitDomainQuadrature.LagrangePolynomialBasis where D where T where NF","page":"Home","title":"Base.sign","text":"Base.sign(P::InterpolatingPolynomial{1}, int::IntervalBox; algorithm = :TaylorModels, tol = 1e-3, order = 5)\n\nspecial function for an interpolating polynomial type.\n\n\n\n\n\n","category":"method"},{"location":"#ImplicitDomainQuadrature.gradient-Tuple{InterpolatingPolynomial{1,NFuncs,B,T} where T where B<:ImplicitDomainQuadrature.AbstractBasis where NFuncs,Number}","page":"Home","title":"ImplicitDomainQuadrature.gradient","text":"gradient(P::InterpolatingPolynomial)(x)\n\nreturn the gradient of P evaluated at x.     gradient(P::InterpolatingPolynomial, dir::Int, x...) return the gradient of P along direction dir evaluated at x.     gradient(P::InterpolatingPolynomial, x::AbstractVector) evaluate the gradient of P at the point vector x\n\n\n\n\n\n","category":"method"},{"location":"#ImplicitDomainQuadrature.gradient-Union{Tuple{N}, Tuple{TensorProductBasis{N,T,NF} where NF where T,Int64,AbstractArray{T,1} where T}} where N","page":"Home","title":"ImplicitDomainQuadrature.gradient","text":"gradient(B::TensorProductBasis{N}, dir::Int, x::AbstractVector) where {N}\n\nevaluate the gradient of B along direction dir at the point vector x     gradient(B::TensorProductBasis{dim}, x::AbstractVector) where {dim} return an (N,dim) matrix such that the Ith row is the gradient of basis I. Here N is the total number of basis functions.     gradient(B::TensorProductBasis{3}, dir::Int, x::Number, y::Number, z::Number) evaluate the gradient of B at (x,y,z) along direction dir.     gradient(B::TensorProductBasis{3}, x::Number, y::Number, z::Number) returns an (N,3) matrix, where row I is the (2D) gradient vector of the Ith basis function. Here N is the total number of basis functions.     gradient(B::TensorProductBasis{2}, x::Number, y::Number) returns an (N,2) matrix, where row I is the (2D) gradient vector of the Ith basis function. Here N is the total number of basis functions.\n\n\n\n\n\n","category":"method"},{"location":"#ImplicitDomainQuadrature.gradient-Union{Tuple{T}, Tuple{ImplicitDomainQuadrature.LagrangePolynomialBasis,T}} where T<:Number","page":"Home","title":"ImplicitDomainQuadrature.gradient","text":"gradient(B::LagrangePolynomialBasis, x::Number)\n\nalias for derivative(B, x)\n\n\n\n\n\n","category":"method"},{"location":"#ImplicitDomainQuadrature.interpolation_points-Union{Tuple{NF}, Tuple{Int64,ImplicitDomainQuadrature.AbstractBasis1D{NF}}} where NF","page":"Home","title":"ImplicitDomainQuadrature.interpolation_points","text":"interpolation_points(B::TensorProductBasis{N,NFuncs,T}) where {NFuncs,T}\n\nreturn the interpolation points of the tensor product basis\n\n\n\n\n\n","category":"method"},{"location":"#ImplicitDomainQuadrature.quadrature-Union{Tuple{T}, Tuple{N}, Tuple{Array{T,1} where T,Array{Int64,1},T,T,ImplicitDomainQuadrature.ReferenceQuadratureRule{N,T}}} where T where N","page":"Home","title":"ImplicitDomainQuadrature.quadrature","text":"quadrature(F::Vector, sign_conditions::Vector{Int}, lo::T, hi::T, quad1d::ReferenceQuadratureRule{N,T}) where {N,T}\n\nreturn a 1D quadrature rule that can be used to integrate in the domain where each f in F has sign specified in sign_conditions in the interval [lo,hi]     quadrature(F::Vector, signconditions::Vector{Int}, int::Interval{T}, quad1d::ReferenceQuadratureRule{N,T}) where {N,T} conveniance function for when the interval over which the quadrature rule is required is specified by an Interval type     quadrature(F::Vector, signconditions::Vector{Int}, heightdir::Int, lo::T, hi::T, x0::AbstractVector{T}, w0::T, quad1d::ReferenceQuadratureRule{N,T}) where {N,T} return a tuple points,weights representing quadrature points and weights obtained by transforming quad1d into an interval bounded by x0 and the zero level set of F by extending along `heightdir`     quadrature(F::Vector, signconditions::Vector{Int}, heightdir::Int, lo::T, hi::T, x0::AbstractMatrix, w0::AbstractVector, quad1d::ReferenceQuadratureRule{N,T}) where {N,T}\n\n\n\n\n\n","category":"method"},{"location":"#ImplicitDomainQuadrature.update!-Tuple{InterpolatingPolynomial,Union{AbstractArray{T,1}, AbstractArray{T,2}} where T}","page":"Home","title":"ImplicitDomainQuadrature.update!","text":"update!(P::InterpolatingPolynomial, coeffs::AbstractMatrix)\n\nupdate P.coeffs = coeffs\n\n\n\n\n\n","category":"method"},{"location":"#ImplicitDomainQuadrature.update!-Union{Tuple{T}, Tuple{D}, Tuple{QuadratureRule{D,T},AbstractArray{T,2},AbstractArray{T,1}}} where T where D","page":"Home","title":"ImplicitDomainQuadrature.update!","text":"update!(quad::QuadratureRule{D,T}, points::AbstractMatrix{T}, weights::AbstractVector{T}) where {D,T}\n\nconcatenates points and weights into quad.points and quad.weights and increments quad.N appropriately.     update!(quad::QuadratureRule{D,T}, point::AbstractVector{T}, weight::T) where {D,T} adds a single point,weight pair to quad\n\n\n\n\n\n","category":"method"},{"location":"#ImplicitDomainQuadrature.AbstractBasis","page":"Home","title":"ImplicitDomainQuadrature.AbstractBasis","text":"AbstractBasis{N}\n\nAbstract supertype for a function basis with N functions.\n\n\n\n\n\n","category":"type"},{"location":"#ImplicitDomainQuadrature.AbstractBasis1D","page":"Home","title":"ImplicitDomainQuadrature.AbstractBasis1D","text":"AbstractBasis1D{N}\n\nAbstract supertype for a univariate function basis with N functions.\n\n\n\n\n\n","category":"type"},{"location":"#ImplicitDomainQuadrature.LagrangePolynomialBasis","page":"Home","title":"ImplicitDomainQuadrature.LagrangePolynomialBasis","text":"LagrangePolynomialBasis{N} <: AbstractBasis1D{N}\n\nA basis of N Lagrange polynomials where each polynomial function is of order N - 1.\n\n\n\n\n\n","category":"type"},{"location":"#ImplicitDomainQuadrature.LagrangePolynomialBasis-Union{Tuple{Int64}, Tuple{T}} where T<:Real","page":"Home","title":"ImplicitDomainQuadrature.LagrangePolynomialBasis","text":"LagrangePolynomialBasis(order::Int; start::T = -1.0, stop::T = 1.0) where {T<:Real}\n\nReturn a LagrangePolynomialBasis with specified polynomial order centered at equispaced points between start and stop.\n\n\n\n\n\n","category":"method"},{"location":"#ImplicitDomainQuadrature.ReferenceQuadratureRule","page":"Home","title":"ImplicitDomainQuadrature.ReferenceQuadratureRule","text":"ReferenceQuadratureRule{N,T}\n\na one-dimensional quadrature rule defined on the reference domain [-1,1].\n\nInner Constructors:\n\nReferenceQuadratureRule(points::SMatrix{1,N,T}, weights::SVector{N,T}) where {T<:Real}\n\nOuter Constructors:\n\nReferenceQuadratureRule(points::AbstractMatrix, weights::AbstractVector)\nReferenceQuadratureRule(points::AbstractVector, weights::AbstractVector)\nReferenceQuadratureRule(N::Int)\n\nreturn an N point gauss legendre quadrature rule on [-1,1]\n\n\n\n\n\n","category":"type"},{"location":"#ImplicitDomainQuadrature.SignSearch","page":"Home","title":"ImplicitDomainQuadrature.SignSearch","text":"SignSearch{N,T} <: AbstractBreadthFirstSearch{IntervalBox{N,T}}\n\na BranchAndPrune search type that is used to determine the sign of a function over a given interval\n\n\n\n\n\n","category":"type"},{"location":"#Base.getindex-Union{Tuple{D}, Tuple{QuadratureRule{D,T} where T,Int64}} where D","page":"Home","title":"Base.getindex","text":"Base.getindex(quad::QuadratureRule{D}, i::Int) where {D}\n\nreturn (p,w) where p is an appropriate view into the ith quadrature point, and w is the ith quadrature weight.\n\n\n\n\n\n","category":"method"},{"location":"#Base.getindex-Union{Tuple{N}, Tuple{ImplicitDomainQuadrature.ReferenceQuadratureRule{N,T} where T,Int64}} where N","page":"Home","title":"Base.getindex","text":"Base.getindex(quad::ReferenceQuadratureRule{N}, i::Int) where {N}\n\nreturn the ith point weight pair p,w.\n\n\n\n\n\n","category":"method"},{"location":"#Base.iterate-Union{Tuple{ImplicitDomainQuadrature.ReferenceQuadratureRule{N,T} where T}, Tuple{N}, Tuple{ImplicitDomainQuadrature.ReferenceQuadratureRule{N,T} where T,Any}} where N","page":"Home","title":"Base.iterate","text":"Base.iterate(quad::ReferenceQuadratureRule{N}, state=1)\n\nreturn each point, weight pair as a tuple (p,w)\n\n\n\n\n\n","category":"method"},{"location":"#Base.iterate-Union{Tuple{QuadratureRule{D,T} where T}, Tuple{D}, Tuple{QuadratureRule{D,T} where T,Any}} where D","page":"Home","title":"Base.iterate","text":"Base.iterate(quad::QuadratureRule{D}, state=1) where {D}\n\nreturn each point, weight pair as a tuple (p,w) where p is an appropriate view into the point matrix quad.points\n\n\n\n\n\n","category":"method"},{"location":"#BranchAndPrune.bisect-Tuple{ImplicitDomainQuadrature.SignSearch,IntervalArithmetic.IntervalBox}","page":"Home","title":"BranchAndPrune.bisect","text":"BranchAndPrune.bisect(::SignSearch, interval::IntervalBox)\n\nsplit the given interval along its largest dimension\n\n\n\n\n\n","category":"method"},{"location":"#BranchAndPrune.process-Tuple{ImplicitDomainQuadrature.SignSearch,IntervalArithmetic.IntervalBox}","page":"Home","title":"BranchAndPrune.process","text":"BranchAndPrune.process(search::SignSearch, interval::IntervalBox)\n\nprocess the given interval to determine the sign of search.f in this interval\n\n\n\n\n\n","category":"method"},{"location":"#ImplicitDomainQuadrature.coefficient_number_type-Tuple{ImplicitDomainQuadrature.LagrangePolynomialBasis}","page":"Home","title":"ImplicitDomainQuadrature.coefficient_number_type","text":"coefficient_number_type(B::LagrangePolynomialBasis)\ncoefficient_number_type(B::TensorProductBasis)\n\nreturn the type of the coefficient, i.e. Float64, Float32\n\n\n\n\n\n","category":"method"},{"location":"#ImplicitDomainQuadrature.derivative-Union{Tuple{T}, Tuple{ImplicitDomainQuadrature.LagrangePolynomialBasis,T}} where T<:Number","page":"Home","title":"ImplicitDomainQuadrature.derivative","text":"derivative(B::LagrangePolynomialBasis, x::Number)\n\nevaluate the derivative of basis B at the point x\n\n\n\n\n\n","category":"method"},{"location":"#ImplicitDomainQuadrature.extend-Tuple{Number,Int64,Number}","page":"Home","title":"ImplicitDomainQuadrature.extend","text":"extend(x0::Number, k::Int, x::Number)\n\nextend x0 into 2D space by inserting x in direction k     extend(x0::AbstractVector, k::Int, x::Number) extend the point vector x0 into d+1 dimensional space by using x as the kth coordinate value in the new point vector     extend(x0::AbstractVector, k::Int, x::AbstractMatrix) extend the point vector x0 along direction k treating x as the new coordinate values\n\n\n\n\n\n","category":"method"},{"location":"#ImplicitDomainQuadrature.isSuitable-Tuple{Int64,InterpolatingPolynomial,IntervalArithmetic.IntervalBox}","page":"Home","title":"ImplicitDomainQuadrature.isSuitable","text":"isSuitable(height_dir::Int, P::InterpolatingPolynomial, box::IntervalBox; order::Int = 5, tol::Float64 = 1e-3)\n\nchecks if the given height_dir is a suitable height direction.\n\n\n\n\n\n","category":"method"},{"location":"#ImplicitDomainQuadrature.lagrange_polynomial-Union{Tuple{T}, Tuple{DynamicPolynomials.PolyVar,AbstractArray{T,1},Int64}} where T","page":"Home","title":"ImplicitDomainQuadrature.lagrange_polynomial","text":"lagrange_polynomial(x::DP.PolyVar,roots::AbstractVector{T},index::Int) where {T}\n\nReturn the indexth lagrange polynomial from the vector of roots with variable x.\n\nThe polynomial is normalized such that it evaluates to 1.0 on roots[index].\n\nExamples\n\njulia> import DynamicPolynomials.@polyvar\njulia> @polyvar x\n(x,)\njulia> ImplicitDomainQuadrature.lagrange_polynomial(x,[-1,0,1],2)\n-x² + 1.0\n\nSee also: lagrange_polynomials\n\n\n\n\n\n","category":"method"},{"location":"#ImplicitDomainQuadrature.lagrange_polynomials-Union{Tuple{T}, Tuple{DynamicPolynomials.PolyVar,AbstractArray{T,1}}} where T","page":"Home","title":"ImplicitDomainQuadrature.lagrange_polynomials","text":"lagrange_polynomials(x::DP.PolyVar,roots::AbstractVector{T}) where {T}\n\nReturn a vector of lagrange polynomial bases constructed from the given roots.\n\nExamples\n\njulia> import DynamicPolynomials.@polyvar\njulia> @polyvar x\n(x,)\njulia> ImplicitDomainQuadrature.lagrange_polynomials(x, [-1,0,1])\n3-element Array{DynamicPolynomials.Polynomial{true,Float64},1}:\n 0.5x² - 0.5x\n -x² + 1.0\n 0.5x² + 0.5x\n\nSee also: lagrange_polynomial\n\n\n\n\n\n","category":"method"},{"location":"#ImplicitDomainQuadrature.min_diam-Tuple{IntervalArithmetic.IntervalBox}","page":"Home","title":"ImplicitDomainQuadrature.min_diam","text":"min_diam(box::IntervalBox)\n\nreturn the minimum diameter of box\n\n\n\n\n\n","category":"method"},{"location":"#ImplicitDomainQuadrature.normalization-Union{Tuple{T}, Tuple{AbstractArray{T,1},Int64}} where T","page":"Home","title":"ImplicitDomainQuadrature.normalization","text":"normalization(roots::AbstractVector{T},skipindex::Int) where {T}\n\nReturn the denominator of the standard Lagrange polynomial formula.\n\nDividing f = polynomial_from_roots(x,roots,skipindex)/normalization(roots,skipindex) gives a normalized polynomial which evaluates to 1.0 on the skipindexth root i.e. f(roots[skipindex] = 1.0.\n\nExamples\n\njulia> ImplicitDomainQuadrature.normalization([-1,1],1)\n-2\n\n\n\n\n\n","category":"method"},{"location":"#ImplicitDomainQuadrature.polynomial_from_roots-Union{Tuple{T}, Tuple{DynamicPolynomials.PolyVar,AbstractArray{T,1},Int64}} where T","page":"Home","title":"ImplicitDomainQuadrature.polynomial_from_roots","text":"polynomial_from_roots(x::DP.PolyVar,roots::AbstractVector{T},skipindex::Int) where {T}\n\nReturn a DynamicPolynomial with variable x with zeros at roots but skipping the skipindexth root.\n\nExamples\n\njulia> import DynamicPolynomials.@polyvar\njulia> @polyvar x\n(x,)\njulia> ImplicitDomainQuadrature.polynomial_from_roots(x,[-1,1],1)\nx - 1\n\n\n\n\n\n","category":"method"},{"location":"#ImplicitDomainQuadrature.quadrature_transformers-Tuple{Any,Any}","page":"Home","title":"ImplicitDomainQuadrature.quadrature_transformers","text":"quadrature_transformers(lo, hi)\n\nreturn 0.5(hi - lo), 0.5*(hi + lo) which are the factors used to transform a quadrature rule from (-1,1) to (lo,hi)\n\n\n\n\n\n","category":"method"},{"location":"#ImplicitDomainQuadrature.roots_and_ends-Union{Tuple{T}, Tuple{Array{T,1} where T,T,T}} where T<:Real","page":"Home","title":"ImplicitDomainQuadrature.roots_and_ends","text":"roots_and_ends(F::Vector, x1::T, x2::T)\n\nreturn a sorted array containing [x1, <roots of each f in F>, x2]\n\n\n\n\n\n","category":"method"},{"location":"#ImplicitDomainQuadrature.run_search-Tuple{Any,IntervalArithmetic.IntervalBox,Int64,Float64}","page":"Home","title":"ImplicitDomainQuadrature.run_search","text":"run_search(f, interval, algorithm, tol, order)\n\nrun a BranchAndPrune search until the sign of f in the given interval is determined\n\n\n\n\n\n","category":"method"},{"location":"#ImplicitDomainQuadrature.signConditionsSatisfied-Tuple{Any,Any,Any}","page":"Home","title":"ImplicitDomainQuadrature.signConditionsSatisfied","text":"signConditionsSatisfied(funcs,xc,sign_conditions)\n\nreturn true if each f in funcs evaluated at xc has the sign specified in sign_conditions. Note: if sign_conditions[i] == 0 then the function always returns true.\n\n\n\n\n\n","category":"method"},{"location":"#ImplicitDomainQuadrature.surface_quadrature-Union{Tuple{T}, Tuple{InterpolatingPolynomial,Int64,T,T,AbstractArray{T,1},T}} where T","page":"Home","title":"ImplicitDomainQuadrature.surface_quadrature","text":"surface_quadrature(F::InterpolatingPolynomial, height_dir::Int, lo::T, hi::T, x0::AbstractVector{T}, w0::T) where {T}\n\nreturn a point,weight pair representing a quadrature point and weight that is obtained by projecting x0 along height_dir onto the zero level set of F within the interval (lo,hi). weight is scaled appropriately by the curvature of F     surfacequadrature(F::InterpolatingPolynomial, heightdir::Int, int::Interval{T}, x0::AbstractVector{T}, w0::T) where {T} convenience function when the interval within which to project x0 is defined by an Interval type     surfacequadrature(F::InterpolatingPolynomial, heightdir::Int, lo::T, hi::T, x0::AbstractMatrix{T}, w0::AbstractVector{T}) where {T} project each column of x0 onto the zero level set of F     surfacequadrature(F::InterpolatingPolynomial, heightdir::Int, int::Interval{T}, x0::AbstractMatrix{T}, w0::AbstractVector{T}) where {T}     quadrature(P::InterpolatingPolynomial{N,NF,B,T}, sign_condition::Int, surface::Bool, box::IntervalBox{2,T}, quad1d::ReferenceQuadratureRule{NQ,T}) where {B<:TensorProductBasis{2}} where {N,NF,NQ,T} corresponds to Algorithm 3 of Robert Saye's 2015 SIAM paper\n\n\n\n\n\n","category":"method"},{"location":"#ImplicitDomainQuadrature.tensorProduct-Tuple{ImplicitDomainQuadrature.ReferenceQuadratureRule,IntervalArithmetic.IntervalBox{2,T} where T}","page":"Home","title":"ImplicitDomainQuadrature.tensorProduct","text":"tensorProduct(quad1d::ReferenceQuadratureRule, box::IntervalBox{2})\n\nreturns a quadrature rule representing the \"tensor product\" or \"kroneker product\" of quad1d that is appropriately transformed into box\n\n\n\n\n\n","category":"method"},{"location":"#ImplicitDomainQuadrature.tensorProductPoints-Tuple{AbstractArray{T,2} where T,AbstractArray{T,2} where T}","page":"Home","title":"ImplicitDomainQuadrature.tensorProductPoints","text":"tensorProductPoints(p1::Matrix, p2::Matrix)\n\nreturns a matrix of points representing the \"tensor product\" (or kroneker product) of p1 and p2\n\n\n\n\n\n","category":"method"},{"location":"#ImplicitDomainQuadrature.transform-Union{Tuple{T}, Tuple{N}, Tuple{ImplicitDomainQuadrature.ReferenceQuadratureRule{N,T},T,T}} where T where N","page":"Home","title":"ImplicitDomainQuadrature.transform","text":"transform(quad::ReferenceQuadratureRule{N,T}, lo::T, hi::T) where {N,T}\n\ntransform quad from the interval [-1,1] to the interval [a,b]     transform(quad::ReferenceQuadratureRule, int::Interval) transform quad from the interval [-1,1] to int\n\n\n\n\n\n","category":"method"},{"location":"#ImplicitDomainQuadrature.unique_root_intervals-Union{Tuple{T}, Tuple{Any,T,T}} where T<:Real","page":"Home","title":"ImplicitDomainQuadrature.unique_root_intervals","text":"unique_root_intervals(f, x1::T, x2::T) where {T<:Real}\n\nreturns sub-intervals in [x1,x2] which contain unique roots of f\n\n\n\n\n\n","category":"method"},{"location":"#ImplicitDomainQuadrature.unique_roots-Union{Tuple{T}, Tuple{Any,T,T}} where T<:Real","page":"Home","title":"ImplicitDomainQuadrature.unique_roots","text":"unique_roots(f, x1::T, x2::T) where {T<:Real}\n\nreturns the unique roots of f in the interval (x1, x2)\n\n\n\n\n\n","category":"method"},{"location":"#ImplicitDomainQuadrature.value_and_derivative-Union{Tuple{T}, Tuple{ImplicitDomainQuadrature.LagrangePolynomialBasis,T}} where T<:Number","page":"Home","title":"ImplicitDomainQuadrature.value_and_derivative","text":"value_and_derivative(B::LagrangePolynomialBasis, x::Number)\n\nreturn (v,d) where v is a vector of the polynomial basis values and d is a vector of the derivative values of the polynomial basis at the point x.\n\n\n\n\n\n","category":"method"}]
}
