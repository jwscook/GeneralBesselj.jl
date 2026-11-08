# J_b(z) and J_{-b}(z) together (complex b, same z).
#
# Series: lanes 1-4 hold terms n..n+3 of the +b series, lanes 5-8 terms n..n+3 of the
# -b series. Term ratios r_k = m / (k (k ± b)) are computed in parallel in all 8 lanes,
# a log-step prefix product within each half turns them into t_{n+j}/t_{n-1}, and one
# complex multiply by the broadcast previous term gives four new terms per order.
#
# Prefactors: one log-gamma (on the order with Re >= 0) and one exp; the other order
# uses Γ(1+b)Γ(1-b) = πb / sin(πb) and (z/2)^(-b) = 1 / (z/2)^b.

# shuffles within halves; lanes taken from the second operand are the identity element
@inline _sh1(x::V8, id::V8) = V8(Base.llvmcall("""
    %r = shufflevector <8 x double> %0, <8 x double> %1, <8 x i32> <i32 8, i32 0, i32 1, i32 2, i32 12, i32 4, i32 5, i32 6>
    ret <8 x double> %r""", _NT, Tuple{_NT,_NT}, x.v, id.v))
@inline _sh2(x::V8, id::V8) = V8(Base.llvmcall("""
    %r = shufflevector <8 x double> %0, <8 x double> %1, <8 x i32> <i32 8, i32 9, i32 0, i32 1, i32 12, i32 13, i32 4, i32 5>
    ret <8 x double> %r""", _NT, Tuple{_NT,_NT}, x.v, id.v))
# broadcast lane 4 to lanes 1-4 and lane 8 to lanes 5-8
@inline _bc_last(x::V8) = V8(Base.llvmcall("""
    %r = shufflevector <8 x double> %0, <8 x double> undef, <8 x i32> <i32 3, i32 3, i32 3, i32 3, i32 7, i32 7, i32 7, i32 7>
    ret <8 x double> %r""", _NT, Tuple{_NT}, x.v))
# sum within halves: (lanes 1-4 summed, lanes 5-8 summed)
@inline function _hsum_halves(x::V8)
  y = x + V8(Base.llvmcall("""
    %r = shufflevector <8 x double> %0, <8 x double> undef, <8 x i32> <i32 2, i32 3, i32 0, i32 1, i32 6, i32 7, i32 4, i32 5>
    ret <8 x double> %r""", _NT, Tuple{_NT}, x.v))
  y = y + V8(Base.llvmcall("""
    %r = shufflevector <8 x double> %0, <8 x double> undef, <8 x i32> <i32 1, i32 0, i32 3, i32 2, i32 5, i32 4, i32 7, i32 6>
    ret <8 x double> %r""", _NT, Tuple{_NT}, y.v))
  return y[1], y[5]
end

const _K0 = V8((1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0))
const _SGN = V8((1.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0, -1.0))

# returns (0F1(;1+b;m), 0F1(;1-b;m)) with m = -(z/2)^2; 8 terms per order per iteration,
# convergence (which needs a horizontal reduction) checked once per iteration
@inline function _series_pair(b::ComplexF64, m::ComplexF64, rtol², maxiters)
  ar = real(b) * _SGN; ai = imag(b) * _SGN
  mr = V8(real(m)); mi = V8(imag(m))
  one_ = V8(1.0); zero_ = V8(0.0)
  K = _K0
  tr = V8(1.0); ti = V8(0.0)                  # previous term, broadcast per half
  sr = V8(0.0); si = V8(0.0)                  # lane-wise partial sums
  n = 1
  while true
    for _ in 1:2
      # r = m / (K (K + a)) = m conj(d) / |d|²
      dr = vfma(K, ar, K * K); di = K * ai
      inv = 1.0 / vfma(dr, dr, di * di)
      rr, ri = _cmul(mr, mi, dr * inv, -di * inv)
      # inclusive prefix product within halves
      pr, pi = _cmul(rr, ri, _sh1(rr, one_), _sh1(ri, zero_))
      pr, pi = _cmul(pr, pi, _sh2(pr, one_), _sh2(pi, zero_))
      tr, ti = _cmul(tr, ti, pr, pi)          # terms n..n+3 for each order
      sr = sr + tr; si = si + ti
      tr = _bc_last(tr); ti = _bc_last(ti)
      K = K + 4.0
    end
    n += 8
    hr1, hr2 = _hsum_halves(sr); hi1, hi2 = _hsum_halves(si)
    S1 = complex(1.0 + hr1, hi1); S2 = complex(1.0 + hr2, hi2)
    t1 = abs2(complex(tr[1], ti[1])); t2 = abs2(complex(tr[5], ti[5]))
    ((t1 < rtol² * abs2(S1)) & (t2 < rtol² * abs2(S2))) | (n > maxiters) && return S1, S2
  end
end

# ((z/2)^b / Γ(1+b), (z/2)^(-b) / Γ(1-b)) from one log-gamma (on the order with Re >= 0),
# one exp and the reflection formula; nothing if exp would over/underflow
@inline function _pair_prefactors(b::ComplexF64, L::ComplexF64)
  flip = real(b) < 0
  c = flip ? -b : b                           # Re c >= 0: Stirling shift is short
  lg, P = _loggamma_shifted(c + 1)            # 1/Γ(1+c) = P exp(-lg)
  e = c * L - lg
  abs(real(e)) < 700 || return nothing
  F = P * exp(e)                              # (z/2)^c / Γ(1+c)
  G = sinpi(c) / (pi * c) / F                 # (z/2)^(-c) / Γ(1-c)
  return flip ? (G, F) : (F, G)
end

# (J_b(z), J_{-b}(z)); returns nothing when the fast path does not apply
@inline function _besselj_pair(b::ComplexF64, z, rtol, maxiters,
    h = complex(float(z) / 2), L = log(h))
  pf = _pair_prefactors(b, L)
  pf === nothing && return nothing
  Fb, Gb = pf
  Sp, Sm = _series_pair(b, -h * h, abs2(rtol), maxiters)
  return Fb * Sp, Gb * Sm
end

# (J_b, J_-b, J_{b-1}, J_{1-b}) at the same z: the b-1 prefactors follow from the b ones
# via Γ(1+b) = b Γ(b) and Γ(2-b) = (1-b) Γ(1-b); only the second series pair is extra.
@inline function _besselj_pair2(b::ComplexF64, z, rtol, maxiters)
  h = complex(float(z) / 2); L = log(h)
  pf = _pair_prefactors(b, L)
  pf === nothing && return nothing
  Fb, Gb = pf
  m = -h * h; r² = abs2(rtol)
  Sp, Sm = _series_pair(b, m, r², maxiters)
  Tp, Tm = _series_pair(b - 1, m, r², maxiters)
  return Fb * Sp, Gb * Sm, (Fb * b / h) * Tp, (Gb * h / (1 - b)) * Tm
end

# Dual z (real): [J_b, J_-b] with derivatives from
#   J_b' = (b/z) J_b - J_{b+1},   J_{-b}' = -J_{1-b} - (b/z) J_{-b}
# (the series route to J_{b-1} cancels catastrophically as b -> 0, giving garbage / NaN)
@inline function _besselj_pair_dual(b::ComplexF64, z::Dual, rtol, maxiters)
  x = realpart(z); d = dualpart(z)
  bz = b / x
  r = abs(x) >= ASYM_MIN_ABSZ ? _besselj_asym_pair2(b, x) : nothing
  if r !== nothing
    Jp, Jm, Jp1, J1m = r
    return Dual(Jp, d * (Jp1 - bz * Jp)), Dual(Jm, d * (-J1m - bz * Jm))
  end
  r = _besselj_pair2(b, x, rtol, maxiters)
  r === nothing && return nothing
  Jp, Jm, _, J1m = r
  q = _besselj_pair(b + 1, x, rtol, maxiters)
  q === nothing && return nothing
  return Dual(Jp, d * (bz * Jp - first(q))), Dual(Jm, d * (-J1m - bz * Jm))
end
