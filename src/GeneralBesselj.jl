module GeneralBesselj

export besselj_v

using DualNumbers, HypergeometricFunctions, SpecialFunctions
import SpecialFunctions: besselj

const ATOL=0.0
const RTOL=1e-12
const MAXITERS=2^10

@inline _frac(x, y) = x / y
@inline function _frac(x::Complex, y::Complex)
  xr, xi = reim(x)
  yr, yi = reim(y)
  invyy = 1 / (yr^2 + yi^2)
  return Complex((muladd(xr, yr,   xi * yi)) * invyy,
                 (muladd(xi, yr, - xr * yi)) * invyy)
end
@inline function _frac(x, y::Complex)
  r, i = reim(y)
  invyy = 1 / (r^2 + i^2)
  return Complex(x * r * invyy, - x * i * invyy)
end
@inline _factor(a, loghalfz::Dual) = exp(a * loghalfz) / gamma(a + 1) # Can't do Complex(Dual)
@inline _factor(a, loghalfz) = exp(muladd(a, loghalfz, - loggamma(a + 1)))
function besselj(a::Number, z::Number; rtol=RTOL, atol=ATOL, maxiters=MAXITERS)
  T = float(promote_type(typeof(a), typeof(z)))
  halfz = z / 2
  loghalfz = log(Complex(halfz))
  return T(_factor(a, loghalfz) * HypergeometricFunctions.pFq(
    Tuple(()), (a + 1,), -halfz^2))
end
function besselj_v(a, z::Number; rtol=RTOL, atol=ATOL,
    maxiters=MAXITERS)
  T = float(promote_type(eltype(a), typeof(z)))
  halfz = z / 2
  loghalfz = log(Complex(halfz))
  output = similar(a, T)
  @inbounds @simd for i in 1:length(a)
    output[i] = HypergeometricFunctions.pFq(Tuple(()), (a[i] + 1,), -halfz^2)
    output[i] *= _factor(a[i], loghalfz)
  end
  return output
end

function besselj(n::Number, x::DualNumbers.Dual)
  r, d = realpart(x), dualpart(x)
  return Dual.(besselj(n, r), d * (besselj(n - 1, r) - besselj(n + 1, r)) / 2)
end
function besselj(n, x::DualNumbers.Dual)
  r, d = realpart(x), dualpart(x)
  return Dual.(besselj(n, r), d .* (besselj(n - 1, r) - besselj(n + 1, r)) / 2)
end

end # module GeneralBesselj
