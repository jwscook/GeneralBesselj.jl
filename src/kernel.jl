# Batched J_a(z) for 8 orders at once (one z), using the V8 type from vec8.jl.

# atan(x) (fdlibm reduction, branch-free)
@inline function vatan(x::V8)
  ax = vsel_lt(x, 0.0, -x, x)
  # interval id: 0:[0,7/16) 1:[7/16,11/16) 2:[11/16,19/16) 3:[19/16,39/16) 4:[39/16,inf)
  c   = vsel_lt(ax, 0.4375, V8(0.0), vsel_lt(ax, 0.6875, V8(0.5),
          vsel_lt(ax, 1.1875, V8(1.0), V8(1.5))))
  hi  = vsel_lt(ax, 0.4375, V8(0.0), vsel_lt(ax, 0.6875, V8(4.63647609000806093515e-01),
          vsel_lt(ax, 1.1875, V8(7.85398163397448278999e-01), vsel_lt(ax, 2.4375,
          V8(9.82793723247329054082e-01), V8(1.57079632679489655800e+00)))))
  lo  = vsel_lt(ax, 0.4375, V8(0.0), vsel_lt(ax, 0.6875, V8(2.26987774529616870924e-17),
          vsel_lt(ax, 1.1875, V8(3.06161699786838301793e-17), vsel_lt(ax, 2.4375,
          V8(1.39033110312309984516e-17), V8(6.12323399573676603587e-17)))))
  u = vsel_lt(ax, 2.4375, (ax - c) / vfma(c, ax, 1.0), -1.0 / ax)
  z = u * u
  w = z * z
  s1 = z * vfma(w, vfma(w, vfma(w, vfma(w, vfma(w, 1.62858201153657823623e-02,
        4.97687799461593236017e-02), 6.66107313738753120669e-02), 9.09088713343650656196e-02),
        1.42857142725034663711e-01), 3.33333333333329318027e-01)
  s2 = w * vfma(w, vfma(w, vfma(w, vfma(w, -3.65315727442169155270e-02,
        -5.83357013379057348645e-02), -7.69187620504482999495e-02),
        -1.11111104054623557880e-01), -1.99999999998764832476e-01)
  r = hi - ((u * (s1 + s2) - lo) - u)
  return vsel_lt(x, 0.0, -r, r)
end

@inline _cmul(ar, ai, br, bi) = (vfma(ar, br, -(ai * bi)), vfma(ar, bi, ai * br))

# ---------- series, lockstep over 8 lanes (runs until the slowest lane converges) ----------

@inline _d(a::V8, k::Int) = vfma(a, Float64(k), Float64(k * k))     # k (k + a)

# d_n ... d_{n+7} and the pair reciprocals i_jk = 1/(d_j d_k), one division per 4 terms.
# (One division per 8 terms measured ~10% worse in error under cancellation; per 4 is noise.)
@inline function _recip8(a::V8, n::Int)
  d1 = _d(a, n); d2 = _d(a, n + 1); d3 = _d(a, n + 2); d4 = _d(a, n + 3)
  d5 = _d(a, n + 4); d6 = _d(a, n + 5); d7 = _d(a, n + 6); d8 = _d(a, n + 7)
  p12 = d1 * d2; p34 = d3 * d4; p56 = d5 * d6; p78 = d7 * d8
  inv1 = 1.0 / (p12 * p34); inv2 = 1.0 / (p56 * p78)
  return d2, p34 * inv1, d4, p12 * inv1, d6, p78 * inv2, d8, p56 * inv2
end

# NB association: odd terms are t * ((m d) i), as in the scalar _0f1. Writing
# t * ((d i) m) instead is 10-30x less accurate when the series cancels (large |z|).

# real orders, real m = -(z/2)^2; 8 terms per iteration, one convergence check
@inline function _series8(a::V8, m::Float64, rtol², maxiters)
  m² = m * m
  t = V8(1.0); s = V8(1.0)
  n = 1
  while n <= maxiters
    d2, i12, d4, i34, d6, i56, d8, i78 = _recip8(a, n)
    t1 = t * ((m * d2) * i12);  t2 = t * (m² * i12)
    t3 = t2 * ((m * d4) * i34); t4 = t2 * (m² * i34)
    t5 = t4 * ((m * d6) * i56); t6 = t4 * (m² * i56)
    t7 = t6 * ((m * d8) * i78); t = t6 * (m² * i78)
    s = s + (((t1 + t2) + (t3 + t4)) + ((t5 + t6) + (t7 + t)))
    vany_lt(rtol² * (s * s), t * t) || break
    n += 8
  end
  return s
end

# real orders, complex m; 8 terms per iteration
@inline function _series8(a::V8, m::ComplexF64, rtol², maxiters)
  mr, mi = reim(m); m²r, m²i = reim(m * m)
  tr = V8(1.0); ti = V8(0.0); sr = V8(1.0); si = V8(0.0)
  n = 1
  while n <= maxiters
    d2, i12, d4, i34, d6, i56, d8, i78 = _recip8(a, n)
    # odd: t * ((m d) i), even: t * (m² i); (m d) i and m² i are complex x real
    t1r, t1i = _cmul(tr, ti, (mr * d2) * i12, (mi * d2) * i12)
    t2r, t2i = _cmul(tr, ti, m²r * i12, m²i * i12)
    t3r, t3i = _cmul(t2r, t2i, (mr * d4) * i34, (mi * d4) * i34)
    t4r, t4i = _cmul(t2r, t2i, m²r * i34, m²i * i34)
    t5r, t5i = _cmul(t4r, t4i, (mr * d6) * i56, (mi * d6) * i56)
    t6r, t6i = _cmul(t4r, t4i, m²r * i56, m²i * i56)
    t7r, t7i = _cmul(t6r, t6i, (mr * d8) * i78, (mi * d8) * i78)
    tr, ti = _cmul(t6r, t6i, m²r * i78, m²i * i78)
    sr = sr + (((t1r + t2r) + (t3r + t4r)) + ((t5r + t6r) + (t7r + tr)))
    si = si + (((t1i + t2i) + (t3i + t4i)) + ((t5i + t6i) + (t7i + ti)))
    vany_lt(rtol² * vfma(sr, sr, si * si), vfma(tr, tr, ti * ti)) || break
    n += 8
  end
  return sr, si
end

# complex orders, complex m; 4 terms per iteration (complex reciprocals, one per pair)
@inline function _series8(ar::V8, ai::V8, m::ComplexF64, rtol², maxiters)
  mr = V8(real(m)); mi = V8(imag(m)); m²r = V8(real(m * m)); m²i = V8(imag(m * m))
  tr = V8(1.0); ti = V8(0.0); sr = V8(1.0); si = V8(0.0)
  n = 1
  while n <= maxiters
    # d_k = k(k+a) = (k² + k ar, k ai)
    d1r = vfma(ar, Float64(n), Float64(n * n));         d1i = ai * Float64(n)
    d2r = vfma(ar, Float64(n + 1), Float64((n + 1)^2)); d2i = ai * Float64(n + 1)
    d3r = vfma(ar, Float64(n + 2), Float64((n + 2)^2)); d3i = ai * Float64(n + 2)
    d4r = vfma(ar, Float64(n + 3), Float64((n + 3)^2)); d4i = ai * Float64(n + 3)
    D12r, D12i = _cmul(d1r, d1i, d2r, d2i); D34r, D34i = _cmul(d3r, d3i, d4r, d4i)
    n12 = 1.0 / vfma(D12r, D12r, D12i * D12i); n34 = 1.0 / vfma(D34r, D34r, D34i * D34i)
    i12r = D12r * n12; i12i = -D12i * n12           # 1/(d1 d2)
    i34r = D34r * n34; i34i = -D34i * n34
    c1r, c1i = _cmul(mr, mi, d2r, d2i); c1r, c1i = _cmul(c1r, c1i, i12r, i12i)    # (m d2) i12
    q12r, q12i = _cmul(m²r, m²i, i12r, i12i)                                      # m² i12
    c3r, c3i = _cmul(mr, mi, d4r, d4i); c3r, c3i = _cmul(c3r, c3i, i34r, i34i)
    q34r, q34i = _cmul(m²r, m²i, i34r, i34i)
    t1r, t1i = _cmul(tr, ti, c1r, c1i)
    t2r, t2i = _cmul(tr, ti, q12r, q12i)
    t3r, t3i = _cmul(t2r, t2i, c3r, c3i)
    tr, ti = _cmul(t2r, t2i, q34r, q34i)
    sr = sr + ((t1r + t2r) + (t3r + tr)); si = si + ((t1i + t2i) + (t3i + ti))
    vany_lt(rtol² * vfma(sr, sr, si * si), vfma(tr, tr, ti * ti)) || break
    n += 4
  end
  return sr, si
end

# ---------- complex 1/Γ(w) = P exp(-lg): exact per-lane shift to Re w >= 7, 12-term Stirling ----------
@inline function vloggamma_shifted(wr::V8, wi::V8)
  Pr = V8(1.0); Pi = V8(0.0)
  while vany_lt(wr, V8(_THR))
    # pair step where >= 2 needed, single where 1 needed, identity otherwise
    qr, qi = _cmul(wr, wi, wr + 1.0, wi)
    fr = vsel_lt(wr, _THR - 1.0, qr, vsel_lt(wr, _THR, wr, V8(1.0)))
    fi = vsel_lt(wr, _THR - 1.0, qi, vsel_lt(wr, _THR, wi, V8(0.0)))
    Pr, Pi = _cmul(Pr, Pi, fr, fi)
    wr = wr + (vsel_lt(wr, _THR, V8(1.0), V8(0.0)) + vsel_lt(wr, _THR - 1.0, V8(1.0), V8(0.0)))
  end
  n = 1.0 / vfma(wr, wr, wi * wi)
  ir = wr * n; ii = -wi * n
  zr, zi = _cmul(ir, ii, ir, ii)
  c = (-236364091/1506960, 77683/5796, -174611/125400, 43867/244188, -3617/122400, 1/156, -691/360360, 1/1188, -1/1680, 1/1260, -1/360, 1/12)
  sr = V8(c[1]); si = V8(0.0)
  for k in 2:length(c)
    sr, si = _cmul(sr, si, zr, zi); sr = sr + c[k]
  end
  sr, si = _cmul(sr, si, ir, ii)
  lr = 0.5 * vlog(vfma(wr, wr, wi * wi))            # log|w|
  li = vatan(wi / wr)                               # arg w (Re w >= 7 > 0)
  # (w - 1/2) log w - w + log(2π)/2 + series
  pr, pi = _cmul(wr - 0.5, wi, lr, li)
  return (pr - wr + 0.9189385332046727 + sr, pi - wi + si), (Pr, Pi)
end

# ---------- lane plumbing ----------
# lane masks (bit i set <=> lane i satisfies the predicate)
@inline vintmask(x::V8) = Base.llvmcall("""
    %t0 = fadd <8 x double> %0, <double 6755399441055744.0, double 6755399441055744.0, double 6755399441055744.0, double 6755399441055744.0, double 6755399441055744.0, double 6755399441055744.0, double 6755399441055744.0, double 6755399441055744.0>
    %r  = fsub <8 x double> %t0, <double 6755399441055744.0, double 6755399441055744.0, double 6755399441055744.0, double 6755399441055744.0, double 6755399441055744.0, double 6755399441055744.0, double 6755399441055744.0, double 6755399441055744.0>
    %i  = fcmp oeq <8 x double> %0, %r
    %m  = bitcast <8 x i1> %i to i8
    ret i8 %m""", UInt8, Tuple{_NT}, x.v)      # integer-valued (|x| < 2^51; larger caught by range)
# x < lo or x > hi or NaN
@inline voutmask(x::V8, lo::V8, hi::V8) = Base.llvmcall("""
    %a  = fcmp ult <8 x double> %0, %1
    %b  = fcmp ugt <8 x double> %0, %2
    %o  = or <8 x i1> %a, %b
    %m  = bitcast <8 x i1> %o to i8
    ret i8 %m""", UInt8, Tuple{_NT,_NT,_NT}, x.v, lo.v, hi.v)
@inline vzeromask(x::V8) = Base.llvmcall("""
    %a  = fcmp oeq <8 x double> %0, zeroinitializer
    %m  = bitcast <8 x i1> %a to i8
    ret i8 %m""", UInt8, Tuple{_NT}, x.v)
# lanes the kernel must not handle: integer orders (SpecialFunctions path; negative
# integers are poles of Γ), NaN/Inf, or outside the range the shift/exp is sized for.
@inline _badmask(a::V8) = vintmask(a) | voutmask(a, V8(-50.0), V8(150.0))
@inline _badmask(ar::V8, ai::V8) = (vintmask(ar) & vzeromask(ai)) |
  voutmask(ar, V8(-50.0), V8(150.0)) | voutmask(ai, V8(-150.0), V8(150.0))
# lanes whose mask bit is set are replaced by p
@inline vblend(mask::UInt8, x::V8, p::Float64) = V8(Base.llvmcall("""
    %k = bitcast i8 %1 to <8 x i1>
    %pi = insertelement <8 x double> undef, double %2, i32 0
    %pp = shufflevector <8 x double> %pi, <8 x double> undef, <8 x i32> zeroinitializer
    %r = select <8 x i1> %k, <8 x double> %pp, <8 x double> %0
    ret <8 x double> %r""", _NT, Tuple{_NT,UInt8,Float64}, x.v, mask, p))
# (re, im) planes -> interleaved complex, two registers
@inline _ilo(a::V8, b::V8) = V8(Base.llvmcall("""
    %r = shufflevector <8 x double> %0, <8 x double> %1, <8 x i32> <i32 0, i32 8, i32 1, i32 9, i32 2, i32 10, i32 3, i32 11>
    ret <8 x double> %r""", _NT, Tuple{_NT,_NT}, a.v, b.v))
@inline _ihi(a::V8, b::V8) = V8(Base.llvmcall("""
    %r = shufflevector <8 x double> %0, <8 x double> %1, <8 x i32> <i32 4, i32 12, i32 5, i32 13, i32 6, i32 14, i32 7, i32 15>
    ret <8 x double> %r""", _NT, Tuple{_NT,_NT}, a.v, b.v))
# interleaved complex (two registers) -> (re, im) planes
@inline _even(a::V8, b::V8) = V8(Base.llvmcall("""
    %r = shufflevector <8 x double> %0, <8 x double> %1, <8 x i32> <i32 0, i32 2, i32 4, i32 6, i32 8, i32 10, i32 12, i32 14>
    ret <8 x double> %r""", _NT, Tuple{_NT,_NT}, a.v, b.v))
@inline _odd(a::V8, b::V8) = V8(Base.llvmcall("""
    %r = shufflevector <8 x double> %0, <8 x double> %1, <8 x i32> <i32 1, i32 3, i32 5, i32 7, i32 9, i32 11, i32 13, i32 15>
    ret <8 x double> %r""", _NT, Tuple{_NT,_NT}, a.v, b.v))

@inline _vload(p::Ptr{Float64}) = V8(unsafe_load(Ptr{_NT}(p)))
@inline _vstore(p::Ptr{Float64}, x::V8) = unsafe_store!(Ptr{_NT}(p), x.v)
@inline _pad(a::AbstractVector, base, cnt, pad) =
  ntuple(i -> i <= cnt ? (@inbounds a[base + i]) : pad, Val(8))

# ---------- per-chunk kernels: 8 orders in, (re, im) planes out ----------
@inline function _chunk(av::V8, Lr, Li, m, rtol², maxiters)
  lg, P = vloggamma_shifted(av + 1.0)
  ex = vfma(av, Lr, -lg)
  vany_lt(V8(700.0), ex) && return nothing           # exp would overflow: caller goes scalar
  mag = vexp(ex) * P
  if m isa Real
    S = _series8(av, m, rtol², maxiters)
    Li == 0 && return (mag * S, V8(0.0))             # z > 0
    sn, cs = vsincos(av * Li)                        # z < 0: phase cis(a π)
    y = mag * S
    return (y * cs, y * sn)
  else
    Sr, Si = _series8(av, m, rtol², maxiters)
    sn, cs = vsincos(av * Li)
    return _cmul(mag * cs, mag * sn, Sr, Si)
  end
end
@inline function _chunk(ar::V8, ai::V8, Lr, Li, m, rtol², maxiters)
  (lgr, lgi), (Pr, Pi) = vloggamma_shifted(ar + 1.0, ai)
  er, ei = _cmul(ar, ai, V8(Lr), V8(Li))
  er = er - lgr; ei = ei - lgi
  vany_lt(V8(700.0), er) && return nothing
  mag = vexp(er)
  sn, cs = vsincos(ei)
  fr, fi = _cmul(mag * cs, mag * sn, Pr, Pi)
  Sr, Si = _series8(ar, ai, m, rtol², maxiters)
  return _cmul(fr, fi, Sr, Si)
end

# ---------- drivers ----------
function _besselj_v_simd!(out::Vector{ComplexF64}, a::Vector{Float64}, z, rtol, maxiters, scalar)
  h = z / 2
  Lr, Li = reim(log(complex(h)))
  m = -h^2
  rtol² = abs2(rtol)
  N = length(a)
  pa = pointer(a); po = Ptr{Float64}(pointer(out))
  GC.@preserve a out begin
    base = 0
    @inbounds while base < N
      cnt = min(8, N - base)
      raw = cnt == 8 ? _vload(pa + 8base) : V8(_pad(a, base, cnt, 0.5))
      bad = _badmask(raw)
      av = bad == 0x00 ? raw : vblend(bad, raw, 0.5)
      y = _chunk(av, Lr, Li, m, rtol², maxiters)
      if y === nothing
        for i in 1:cnt; out[base + i] = scalar(a[base + i]); end
      else
        yr, yi = y
        if cnt == 8
          _vstore(po + 16base, _ilo(yr, yi)); _vstore(po + 16base + 64, _ihi(yr, yi))
        else
          for i in 1:cnt; out[base + i] = complex(yr[i], yi[i]); end
        end
        while bad != 0x00                              # fix up flagged lanes
          i = trailing_zeros(bad) + 1
          i <= cnt && (out[base + i] = scalar(a[base + i]))
          bad &= bad - 0x01
        end
      end
      base += 8
    end
  end
  return out
end

function _besselj_v_simd!(out::Vector{ComplexF64}, a::Vector{ComplexF64}, z, rtol, maxiters, scalar)
  h = complex(z / 2)
  Lr, Li = reim(log(h))
  m = -h^2
  rtol² = abs2(rtol)
  N = length(a)
  pa = Ptr{Float64}(pointer(a)); po = Ptr{Float64}(pointer(out))
  GC.@preserve a out begin
    base = 0
    @inbounds while base < N
      cnt = min(8, N - base)
      if cnt == 8
        lo = _vload(pa + 16base); hi = _vload(pa + 16base + 64)
        ar = _even(lo, hi); ai = _odd(lo, hi)
      else
        t = _pad(a, base, cnt, 0.5 + 0.0im)
        ar = V8(map(real, t)); ai = V8(map(imag, t))
      end
      bad = _badmask(ar, ai)
      if bad != 0x00
        ar = vblend(bad, ar, 0.5); ai = vblend(bad, ai, 0.0)
      end
      y = _chunk(ar, ai, Lr, Li, m, rtol², maxiters)
      if y === nothing
        for i in 1:cnt; out[base + i] = scalar(a[base + i]); end
      else
        yr, yi = y
        if cnt == 8
          _vstore(po + 16base, _ilo(yr, yi)); _vstore(po + 16base + 64, _ihi(yr, yi))
        else
          for i in 1:cnt; out[base + i] = complex(yr[i], yi[i]); end
        end
        while bad != 0x00
          i = trailing_zeros(bad) + 1
          i <= cnt && (out[base + i] = scalar(a[base + i]))
          bad &= bad - 0x01
        end
      end
      base += 8
    end
  end
  return out
end
