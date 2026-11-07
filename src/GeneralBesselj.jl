module GeneralBesselj

export besselj_v

using DualNumbers, SpecialFunctions
import SpecialFunctions: besselj

const MAXITERS=2^10

include("vec8.jl")
include("kernel.jl")

#function _besselj_halfz(a, halfz::Number, loghalfz; maxiters=MAXITERS)
#  return _factor(a, loghalfz) * HypergeometricFunctions.pFq(
#    Tuple(()), (a + 1,), -halfz^2; kmax=maxiters)
#end

fastinv(z) = 1 / z
function fastinv(z::Complex{T}) where T
  a, b = real(z), imag(z)
  invdenom = 1 / (a * a + b * b)
  return Complex(a * invdenom, -b * invdenom)
end

# 0F1(;a+1;-halfz²) with the first term normalised to 1 (1/Γ(a+1) applied by caller).
# Unrolled by 4: the loop-carried chain is two multiplies per four terms, each pair of
# terms shares one reciprocal, and convergence is checked once per four terms.
@inline function _0f1(a, halfz, rtol, maxiters)
  m = -halfz^2
  m² = m * m
  term = one(m * fastinv(one(a)))
  result = term
  rtol² = abs2(rtol)
  for n in 1:4:maxiters
    d1 = muladd(n, a, n^2)
    d2 = muladd(n + 1, a, (n + 1)^2)
    d3 = muladd(n + 2, a, (n + 2)^2)
    d4 = muladd(n + 3, a, (n + 3)^2)
    i12 = fastinv(d1 * d2)
    i34 = fastinv(d3 * d4)
    t1 = term * ((m * d2) * i12)
    t2 = term * (m² * i12)
    t3 = t2 * ((m * d4) * i34)
    term = t2 * (m² * i34)
    result += (t1 + t2) + (t3 + term)
    abs2(term) < rtol² * abs2(result) && break
  end
  return result
end

# loggamma(w) for complex w: shift Re(w) up to >= 10 with a Pochhammer product, then
# Stirling with 8 Bernoulli terms (truncation error < 1e-17 at |w| >= 10).
# Returns (log Γ(w+M) up to 2πi, P = w(w+1)...(w+M-1)) so that 1/Γ(w) = P * exp(-lg).
const _STIRLING = (1/12, -1/360, 1/1260, -1/1680, 1/1188, -691/360360, 1/156, -3617/122400)
@inline function _loggamma_shifted(w::Complex{T}) where T
  P = one(w)
  while real(w) < 10
    P *= w
    w += 1
  end
  iw = fastinv(w)
  iw² = iw * iw
  series = iw * evalpoly(iw², _STIRLING)
  lg = muladd(w - T(0.5), log(w), -w) + T(0.9189385332046727) + series
  return lg, P
end

# Compute J_a(z) using 0F1 series
# J_a(z) = (z/2)^a / Γ(a+1) * 0F1(;a+1;-z²/4)
# exp(log prefactor) * series avoids complex gamma *and* the final complex log.
function _besselj_halfz(a, halfz, loghalfz, rtol=10eps(); maxiters=MAXITERS)
  return exp(a * loghalfz - loggamma(a + 1)) * _0f1(a, halfz, rtol, maxiters)
end
# complex order: cheap shifted Stirling instead of SpecialFunctions.loggamma, unless a
# is far into the left half plane (product would get long / overflow).
function _besselj_halfz(a::Complex{<:Union{Float32,Float64}}, halfz, loghalfz, rtol=10eps();
    maxiters=MAXITERS)
  real(a) < -30 && return exp(a * loghalfz - loggamma(a + 1)) * _0f1(a, halfz, rtol, maxiters)
  lg, P = _loggamma_shifted(a + 1)
  return P * exp(a * loghalfz - lg) * _0f1(a, halfz, rtol, maxiters)
end
# Dual order: no Complex(Dual) loggamma, keep the gamma route
function _besselj_halfz(a::Dual, halfz, loghalfz, rtol=10eps(); maxiters=MAXITERS)
  return exp(a * loghalfz) * fastinv(gamma(a + 1 + 0im)) * _0f1(a, halfz, rtol, maxiters)
end
# real order, complex argument: real logabsgamma is ~7x cheaper than complex gamma
function _besselj_halfz(a::Real, halfz, loghalfz, rtol=10eps(); maxiters=MAXITERS)
  lg, sgn = logabsgamma(a + 1)
  return sgn * exp(a * loghalfz - lg) * _0f1(a, halfz, rtol, maxiters)
end
# real order and real argument: series in real arithmetic, phase for z<0 via cispi
function _besselj_halfz(a::Real, halfz::Real, loghalfz, rtol=10eps(); maxiters=MAXITERS)
  lg, sgn = logabsgamma(a + 1)
  mag = sgn * exp(muladd(a, log(abs(halfz)), -lg)) * _0f1(a, halfz, rtol, maxiters)
  return signbit(halfz) ? mag * cispi(a) : complex(mag)
end

function besselj(a::Number, z::Number, rtol=10eps(); maxiters=MAXITERS)
  T = float(promote_type(typeof(a), typeof(z)))
  (!(typeof(a) <: Dual) && isinteger(a)) && return T(besselj(round(Int, real(a)), z))
  halfz = z / 2
  loghalfz = log(complex(halfz))
  return _besselj_halfz(a, halfz, loghalfz, rtol; maxiters=maxiters)
end
function besselj_v(a, z::Number, rtol=10eps(); maxiters=MAXITERS)
  T = complex(float(promote_type(eltype(a), typeof(z))))
  halfz = z / 2
  loghalfz = log(complex(halfz))
  output = similar(a, T)
  if a isa Union{Vector{Float64},Vector{ComplexF64}} && z isa Union{Float64,ComplexF64} && !iszero(z)
    scalar = ai -> isinteger(ai) ? ComplexF64(besselj(round(Int, real(ai)), z)) :
      ComplexF64(_besselj_halfz(ai, halfz, loghalfz, rtol; maxiters=maxiters))
    return _besselj_v_simd!(output, a, z, rtol, maxiters, scalar)
  end
  @inbounds for i in eachindex(a, output)
    ai = a[i]
    if !(eltype(a) <: Dual) && isinteger(ai)
      output[i] = besselj(round(Int, real(ai)), z)
    else
      output[i] = _besselj_halfz(ai, halfz, loghalfz, rtol; maxiters=maxiters)
    end
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
