using Test
using PolynomialBasis
using IntervalArithmetic
# using Revise
using ImplicitDomainQuadrature

IDQ = ImplicitDomainQuadrature
PB = PolynomialBasis

function allapprox(v1, v2)
    return all(v1 .≈ v2)
end

f(x) = (x - 1.0) * (x - 1.5)
g(x) = (x - 0.5) * (x - 1.25)
r = IDQ.roots_and_ends([f, g], 0.0, 2.0)
testr = [0.0, 0.5, 1.0, 1.25, 1.5, 2.0]
@test allapprox(r, testr)

@test IDQ.extend(1.0, 1, 2.0) ≈ [2.0, 1.0]
@test IDQ.extend(1.0, 2, 2.0) ≈ [1.0, 2.0]
@test_throws ArgumentError IDQ.extend(1.0, 3, 2.0)

@test allapprox(IDQ.extend([1.0], 1, 2.0), [2.0, 1.0])
@test_throws ArgumentError IDQ.extend([1.0, 2.0], 1, 2.0)

x0 = [1.0]
x = [
    2.0 3.0 4.0 5.0
    6.0 7.0 8.0 9.0
]
@test_throws ArgumentError IDQ.extend(x0, 1, x)

x0 = [1.0]
x = [2.0 3.0 4.0 5.0]
testx = [
    2.0 3.0 4.0 5.0
    1.0 1.0 1.0 1.0
]
@test allapprox(IDQ.extend(x0, 1, x), testx)

x0 = [1.0]
x = [2.0 3.0 4.0 5.0]
testx = [
    1.0 1.0 1.0 1.0
    2.0 3.0 4.0 5.0
]
@test allapprox(IDQ.extend(x0, 2, x), testx)

x0 = [1.0]
x = [2.0 3.0 4.0 5.0]
@test_throws ArgumentError IDQ.extend(x0, 3, x)


f2(x) = x[2]
P = InterpolatingPolynomial(1, 2, 2)
coeffs = [f2(P.basis.points[:, i]) for i = 1:size(P.basis.points)[2]]
update!(P, coeffs)
@test IDQ.height_direction(x -> gradient(P, x), [0.0, 0.0]) == 2
box = IntervalBox(-1..1, 2)
@test IDQ.height_direction(x -> gradient(P, x), box) == 2

flag, s = IDQ.is_suitable(2, x -> gradient(P, x), box)
@test flag == true
@test s == 1

f2(x) = (x[2] + 0.5) * (x[2] - 0.5)
P = InterpolatingPolynomial(1, 2, 2)
coeffs = [f2(P.basis.points[:, i]) for i = 1:size(P.basis.points)[2]]
update!(P, coeffs)
flag, s = IDQ.is_suitable(2, x -> gradient(P, x), box)
@test flag == false

@test IDQ.sign(1, 1, true, -1) == -1
@test IDQ.sign(1, -1, false, -1) == -1
@test IDQ.sign(1, -1, false, 1) == 0




function testcubic(coeffs, v)
    a, b, c, d, e, f, g, h, i, j = coeffs
    x, y = v
    return a * x^3 +
           b * x^2 * y +
           c * x * y^2 +
           d * y^3 +
           e * x^2 +
           f * x * y +
           g * y^2 +
           h * x +
           i * y +
           j
end

function testcubicgrad(coeffs, v)
    a, b, c, d, e, f, g, h, i, j = coeffs
    x, y = v

    fx = 3 * a * x^2 + 2 * b * x * y + c * y^2 + 2 * e * x + f * y + h
    fy = b * x^2 + 2 * c * x * y + 3 * d * y^2 + f * x + 2 * g * y + i
    return [fx, fy]
end

coeffs = [10,12,6,17,9,5,2,1,8,11.0]
poly = InterpolatingPolynomial(1,2,3)
points = PolynomialBasis.interpolation_points(PolynomialBasis.basis(poly))
interpcoeffs = vec(mapslices(x->testcubic(coeffs,x),points,dims=1))
update!(poly,interpcoeffs)

testp = rand(2,10)
vals = mapslices(poly,testp,dims=1)
testvals = mapslices(x->testcubic(coeffs,x),testp,dims=1)
@test allapprox(vals,testvals)

interpgrad = InterpolatingPolynomial(2,2,3)
IDQ.update_interpolating_gradient!(interpgrad,poly)
vals = mapslices(interpgrad,testp,dims=1)
testvals = mapslices(x->testcubicgrad(coeffs,x),testp,dims=1)
@test allapprox(vals,testvals)
