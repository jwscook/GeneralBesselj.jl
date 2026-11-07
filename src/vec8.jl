# Hand-rolled 8 x Float64 SIMD via llvmcall. Maps to one zmm register on AVX-512,
# two ymm registers on AVX2 (LLVM legalises).
const _NT = NTuple{8,VecElement{Float64}}
struct V8
  v::_NT
end
@inline V8(x::Float64) = V8(ntuple(_ -> VecElement(x), Val(8)))
@inline V8(x::Real) = V8(Float64(x))
@inline V8(t::NTuple{8,Float64}) = V8(map(VecElement, t))
@inline Base.getindex(x::V8, i::Int) = x.v[i].value
@inline lanes(x::V8) = map(e -> e.value, x.v)

const _T = "<8 x double>"
for (op, ins) in ((:+, "fadd"), (:-, "fsub"), (:*, "fmul"), (:/, "fdiv"))
  @eval @inline Base.$op(a::V8, b::V8) = V8(Base.llvmcall(
    $("%r = $ins $_T %0, %1\nret $_T %r"), _NT, Tuple{_NT,_NT}, a.v, b.v))
  @eval @inline Base.$op(a::V8, b::Real) = $op(a, V8(b))
  @eval @inline Base.$op(a::Real, b::V8) = $op(V8(a), b)
end
@inline Base.:-(a::V8) = V8(Base.llvmcall(
  "%r = fneg <8 x double> %0\nret <8 x double> %r", _NT, Tuple{_NT}, a.v))

@inline vfma(a::V8, b::V8, c::V8) = V8(Base.llvmcall(("""
  declare <8 x double> @llvm.fma.v8f64(<8 x double>, <8 x double>, <8 x double>)
  define <8 x double> @entry(<8 x double> %a, <8 x double> %b, <8 x double> %c) #0 {
    %r = call <8 x double> @llvm.fma.v8f64(<8 x double> %a, <8 x double> %b, <8 x double> %c)
    ret <8 x double> %r
  }
  attributes #0 = { alwaysinline }""", "entry"), _NT, Tuple{_NT,_NT,_NT}, a.v, b.v, c.v))
@inline vfma(a, b, c) = vfma(_v(a), _v(b), _v(c))
@inline _v(x::V8) = x
@inline _v(x::Real) = V8(x)

# round to nearest (ties even) for |a| < 2^51 via the 1.5*2^52 trick: no intrinsic needed
@inline vrint(a::V8) = (a + 6755399441055744.0) - 6755399441055744.0

# x < y ? p : q, lane-wise
@inline vsel_lt(x::V8, y::V8, p::V8, q::V8) = V8(Base.llvmcall(
  "%m = fcmp olt <8 x double> %0, %1\n%r = select <8 x i1> %m, <8 x double> %2, <8 x double> %3\nret <8 x double> %r",
  _NT, Tuple{_NT,_NT,_NT,_NT}, x.v, y.v, p.v, q.v))
@inline vsel_lt(x, y, p, q) = vsel_lt(_v(x), _v(y), _v(p), _v(q))

# any lane with x < y  (NaN lanes compare false)
@inline vany_lt(x::V8, y::V8) = Base.llvmcall(
  "%m = fcmp olt <8 x double> %0, %1\n%b = bitcast <8 x i1> %m to i8\n%c = icmp ne i8 %b, 0\n%r = zext i1 %c to i8\nret i8 %r",
  UInt8, Tuple{_NT,_NT}, x.v, y.v) != 0x00

# 2^k for integer-valued k in [-1022, 1023]
@inline vpow2i(k::V8) = V8(Base.llvmcall(
  "%i = fptosi <8 x double> %0 to <8 x i64>\n%e = add <8 x i64> %i, <i64 1023, i64 1023, i64 1023, i64 1023, i64 1023, i64 1023, i64 1023, i64 1023>\n%s = shl <8 x i64> %e, <i64 52, i64 52, i64 52, i64 52, i64 52, i64 52, i64 52, i64 52>\n%r = bitcast <8 x i64> %s to <8 x double>\nret <8 x double> %r",
  _NT, Tuple{_NT}, k.v))

# for positive normal x: exponent (as Float64) and mantissa in [1,2)
@inline vexponent(x::V8) = V8(Base.llvmcall(
  "%b = bitcast <8 x double> %0 to <8 x i64>\n%s = lshr <8 x i64> %b, <i64 52, i64 52, i64 52, i64 52, i64 52, i64 52, i64 52, i64 52>\n%e = sub <8 x i64> %s, <i64 1023, i64 1023, i64 1023, i64 1023, i64 1023, i64 1023, i64 1023, i64 1023>\n%r = sitofp <8 x i64> %e to <8 x double>\nret <8 x double> %r",
  _NT, Tuple{_NT}, x.v))
@inline vmantissa(x::V8) = V8(Base.llvmcall(
  "%b = bitcast <8 x double> %0 to <8 x i64>\n%a = and <8 x i64> %b, <i64 4503599627370495, i64 4503599627370495, i64 4503599627370495, i64 4503599627370495, i64 4503599627370495, i64 4503599627370495, i64 4503599627370495, i64 4503599627370495>\n%o = or <8 x i64> %a, <i64 4607182418800017408, i64 4607182418800017408, i64 4607182418800017408, i64 4607182418800017408, i64 4607182418800017408, i64 4607182418800017408, i64 4607182418800017408, i64 4607182418800017408>\n%r = bitcast <8 x i64> %o to <8 x double>\nret <8 x double> %r",
  _NT, Tuple{_NT}, x.v))

# ---------------- vector elementary functions (branch-free) ----------------

# exp(x); flushes to 0 below -708, no overflow handling above 709 (callers stay in range)
@inline function vexp(x::V8)
  xc = vsel_lt(x, -708.0, V8(-708.0), x)
  k = vrint(xc * 1.4426950408889634)
  r = vfma(-k, 6.93147180369123816490e-01, xc)
  r = vfma(-k, 1.90821492927058770002e-10, r)
  # Taylor to r^13; |r| <= 0.347 so truncation < 4e-18
  p = vfma(r, 1/6227020800, 1/479001600)
  p = vfma(r, p, 1/39916800); p = vfma(r, p, 1/3628800); p = vfma(r, p, 1/362880)
  p = vfma(r, p, 1/40320);    p = vfma(r, p, 1/5040);    p = vfma(r, p, 1/720)
  p = vfma(r, p, 1/120);      p = vfma(r, p, 1/24);      p = vfma(r, p, 1/6)
  p = vfma(r, p, 0.5);        p = vfma(r, p, 1.0);       p = vfma(r, p, 1.0)
  res = p * vpow2i(k)
  return vsel_lt(x, -708.0, V8(0.0), res)
end

# log(x) for positive normal x (fdlibm algorithm)
@inline function vlog(x::V8)
  e = vexponent(x)
  m = vmantissa(x)
  big = vsel_lt(1.4142135623730951, m, V8(1.0), V8(0.0))
  m = vsel_lt(1.4142135623730951, m, m * 0.5, m)
  e = e + big
  f = m - 1.0
  s = f / (2.0 + f)
  z = s * s
  w = z * z
  t1 = w * vfma(w, vfma(w, 1.531383769920937332e-01, 2.222219843214978396e-01), 3.999999999940941908e-01)
  t2 = z * vfma(w, vfma(w, vfma(w, 1.479819860511658591e-01, 1.818357216161805012e-01),
                        2.857142874366239149e-01), 6.666666666666735130e-01)
  R = t2 + t1
  hfsq = 0.5 * f * f
  return vfma(e, 6.93147180369123816490e-01,
    -((hfsq - vfma(s, hfsq + R, e * 1.90821492927058770002e-10)) - f))
end

# (sin x, cos x), Cody-Waite reduction by pi/2 (fine for |x| < ~1e5)
@inline function vsincos(x::V8)
  k = vrint(x * 0.6366197723675814)
  r = vfma(-k, 1.57079632673412561417e+00, x)
  r = vfma(-k, 6.07710050630396597660e-11, r)
  r = vfma(-k, 2.02226624871116645580e-21, r)
  z = r * r
  s = vfma(z, vfma(z, vfma(z, vfma(z, vfma(z, 1.58969099521155010221e-10,
        -2.50507602534068634195e-08), 2.75573137070700676789e-06),
        -1.98412698298579493134e-04), 8.33333333332248946124e-03), -1.66666666666666324348e-01)
  s = vfma(r * z, s, r)
  c = vfma(z, vfma(z, vfma(z, vfma(z, vfma(z, -1.13596475577881948265e-11,
        2.08757232129817482790e-09), -2.75573143513906633035e-07),
        2.48015872894767294178e-05), -1.38888888888741095749e-03), 4.16666666666666019037e-02)
  c = vfma(z * z, c, vfma(z, -0.5, 1.0))
  q = k - 4.0 * vrint((k - 1.5) * 0.25)        # k mod 4 in {0,1,2,3}
  odd = vsel_lt(0.5, q, vsel_lt(q, 1.5, V8(1.0), vsel_lt(2.5, q, V8(1.0), V8(0.0))), V8(0.0))
  S = vsel_lt(0.5, odd, c, s)
  C = vsel_lt(0.5, odd, s, c)
  S = vsel_lt(1.5, q, -S, S)                                       # q in {2,3}: -sin
  C = vsel_lt(0.5, q, vsel_lt(q, 2.5, -C, C), C)                   # q in {1,2}: -cos
  return S, C
end


# Real 1/Γ(w) = P * exp(-lg), w not a pole: each lane is shifted up exactly to w >= 7
# (two steps per multiply via w(w+1), masked) and lg is a 12-term Stirling series, whose
# truncation error at w >= 7 is < 6e-18. Exact per-lane shifting keeps |lg| small, which
# matters because the caller exponentiates a·log(z/2) - lg.
const _THR = 7.0
@inline function _stirling_real(w::V8)
  iw = 1.0 / w
  iw² = iw * iw
  c = (-236364091/1506960, 77683/5796, -174611/125400, 43867/244188, -3617/122400, 1/156, -691/360360, 1/1188, -1/1680, 1/1260, -1/360, 1/12)
  ser = V8(c[1])
  for k in 2:length(c); ser = vfma(iw², ser, c[k]); end
  return vfma(w - 0.5, vlog(w), -w) + 0.9189385332046727 + iw * ser
end
@inline function vloggamma_shifted(w::V8)
  P = V8(1.0)
  while vany_lt(w, V8(_THR))
    two = vsel_lt(w, _THR - 1.0, V8(1.0), V8(0.0))            # lane needs >= 2 steps
    one = vsel_lt(w, _THR, V8(1.0), V8(0.0))                  # lane needs >= 1 step
    f = vsel_lt(w, _THR - 1.0, vfma(w, w, w), vsel_lt(w, _THR, w, V8(1.0)))
    P = P * f
    w = w + (one + two)
  end
  return _stirling_real(w), P
end
