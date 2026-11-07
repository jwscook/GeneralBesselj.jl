# Large-|z| Hankel asymptotic expansion (DLMF 10.17.3):
#   J_ν(z) ~ sqrt(2/(πz)) [cos ω Σ_k (-1)^k a_{2k}(ν)/z^{2k} - sin ω Σ_k (-1)^k a_{2k+1}(ν)/z^{2k+1}],
#   ω = z - νπ/2 - π/4,  a_k(ν) = (4ν²-1²)(4ν²-3²)...(4ν²-(2k-1)²) / (k! 8^k),
# valid for |arg z| < π. For Re z < 0 the principal branch is reached through
# J_ν(z) = e^{±iπν} J_ν(-z), with the sign taken from the sign bit of Im z so that it
# matches the series path's log(complex(z/2)) (Im z = +0.0 -> arg +π, -0.0 -> arg -π).
#
# The series is divergent: terms are summed while they decrease, and the result is used
# only if the smallest term is below eps relative to the sum. Otherwise `nothing` is
# returned and the caller uses the power series.

const ASYM_MIN_ABSZ = 17.0
# the expansion is evaluated in Float64: only use it for Float64-based, non-Dual inputs
_asym_ok(a, z) = !(a isa Dual) && !(z isa Dual) &&
  real(float(typeof(a))) === Float64 && real(float(typeof(z))) === Float64

function _besselj_asym(ν::Number, z::Number)
  T = Float64
  zc = complex(T(real(z)), T(imag(z)))
  abs(zc) < ASYM_MIN_ABSZ && return nothing
  νc = complex(T(real(ν)), T(imag(ν)))
  # reflect to Re z >= 0
  flip = real(zc) < 0
  w = flip ? -zc : zc
  μ = 4 * νc * νc
  iw8 = inv(8 * w)
  # a_k/w^k = a_{k-1}/w^{k-1} * (μ - (2k-1)²) / (8 k w); P takes even k, Q odd k,
  # each with sign (-1)^⌊k/2⌋
  P = one(νc); Q = zero(νc)
  term = one(νc)
  prev = T(Inf)
  conv = false
  for k in 1:200
    term *= (μ - (2k - 1)^2) * (iw8 / k)
    at = abs2(term)
    at > prev && break                       # divergent tail starts: stop
    prev = at
    c = iseven(k >> 1) ? term : -term
    if isodd(k); Q += c; else; P += c; end
    if k >= 2 && at <= (eps(T) / 2)^2 * (abs2(P) + abs2(Q))
      conv = true
      break
    end
  end
  conv || return nothing
  # cos ω, sin ω with ω = w - φ, φ = (ν/2 + 1/4)π; use the accurately reduced sincos of w
  sw, cw = sincos(w)
  sφ, cφ = sinpi(νc / 2 + T(0.25)), cospi(νc / 2 + T(0.25))
  cosω = cw * cφ + sw * sφ
  sinω = sw * cφ - cw * sφ
  J = sqrt(2 / (T(pi) * w)) * (cosω * P - sinω * Q)
  if flip
    J *= signbit(imag(zc)) ? cispi(-νc) : cispi(νc)
  end
  return J
end

# Hankel sums P, Q depend on ν only through ν², so J_ν and J_{-ν} share them.
@inline function _hankel_PQ(ν::ComplexF64, w::ComplexF64)
  T = Float64
  μ = 4 * ν * ν
  iw8 = inv(8 * w)
  P = one(ν); Q = zero(ν); term = one(ν); prev = T(Inf)
  for k in 1:200
    term *= (μ - (2k - 1)^2) * (iw8 / k)
    at = abs2(term)
    at > prev && return nothing
    prev = at
    c = iseven(k >> 1) ? term : -term
    if isodd(k); Q += c; else; P += c; end
    k >= 2 && at <= (eps(T) / 2)^2 * (abs2(P) + abs2(Q)) && return P, Q
  end
  return nothing
end

# (J_ν, J_{-ν}) for large |z|: one sum, one sincos(w), one sincospi(ν/2)
@inline function _besselj_asym_pair(ν::ComplexF64, zc::ComplexF64, sw, cw, A, flip, σ)
  w = flip ? -zc : zc
  pq = _hankel_PQ(ν, w)
  pq === nothing && return nothing
  P, Q = pq
  s, c = sincospi(ν / 2)
  r = 1 / sqrt(2.0)
  # φ± = ±ν/2 + 1/4
  sp, cp = (s + c) * r, (c - s) * r
  sm, cm = (c - s) * r, (c + s) * r
  Jp = A * ((cw * cp + sw * sp) * P - (sw * cp - cw * sp) * Q)
  Jm = A * ((cw * cm + sw * sm) * P - (sw * cm - cw * sm) * Q)
  if flip                                     # J_ν(z) = e^{iσπν} J_ν(-z)
    cπ = c * c - s * s; sπ = 2 * s * c        # cospi(ν), sinpi(ν)
    Jp *= complex(real(cπ) - σ * imag(sπ), imag(cπ) + σ * real(sπ))   # cπ + iσ sπ
    Jm *= complex(real(cπ) + σ * imag(sπ), imag(cπ) - σ * real(sπ))   # cπ - iσ sπ
  end
  return Jp, Jm
end

@inline function _asym_common(z)
  zc = complex(Float64(real(z)), Float64(imag(z)))
  flip = real(zc) < 0
  w = flip ? -zc : zc
  sw, cw = sincos(w)
  return zc, sw, cw, sqrt(2 / (pi * w)), flip, signbit(imag(zc)) ? -1.0 : 1.0
end

function _besselj_asym_pair(ν::ComplexF64, z::Number)
  abs(z) < ASYM_MIN_ABSZ && return nothing
  zc, sw, cw, A, flip, σ = _asym_common(z)
  return _besselj_asym_pair(ν, zc, sw, cw, A, flip, σ)
end

# (J_b, J_-b, J_{b-1}, J_{1-b}) for large |z|, sharing sincos(z) and sqrt
function _besselj_asym_pair2(b::ComplexF64, z::Number)
  abs(z) < ASYM_MIN_ABSZ && return nothing
  zc, sw, cw, A, flip, σ = _asym_common(z)
  r1 = _besselj_asym_pair(b, zc, sw, cw, A, flip, σ)
  r1 === nothing && return nothing
  r0 = _besselj_asym_pair(b - 1, zc, sw, cw, A, flip, σ)
  r0 === nothing && return nothing
  return r1[1], r1[2], r0[1], r0[2]
end
