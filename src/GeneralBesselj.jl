module GeneralBesselj

export besselj_v

using DualNumbers, SpecialFunctions
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

@inline function hypergeom_0f1(a::Number, z; atol=ATOL, rtol=RTOL, maxiters=MAXITERS)
  T = float(promote_type(typeof(a), typeof(z)))

  (!(typeof(a) <: Dual) && isinteger(a)) && return T(SpecialFunctions.besselj(Int(a), z))

  # Handle special cases
  z == 0 && return one(T)
  
  atol², rtol² = atol^2, rtol^2
  
  # Initialise variables with promoted types
  sum_val = one(T)
  term = one(T)
  # Use recurrence relation: term_{k+1} = term_k * z / ((a+k) * (k+1))
  k = 0
  while k < maxiters
    k += 1
    term *= _frac(z, muladd(a + k, k, -k)) # z / ((a + k - 1) * k)
    sum_val += term
    abs2(term) < muladd(rtol², abs2(sum_val), atol²) && return sum_val
  end
  
  throw(Methoderror("No convergence reached"))
end

@inline function hypergeom_0f1(a::AbstractVector, z; atol=ATOL, rtol=RTOL, maxiters=MAXITERS)
  T = float(promote_type(eltype(a), typeof(z)))
  n = length(a)

  (!(eltype(a) <: Dual) && all(isinteger, a)) && return T.(SpecialFunctions.besselj.(Int.(a), z))

  # Handle special cases
  sum_val = similar(a, T)
  fill!(sum_val, 1)
  z == 0 && return sum_val

  atol², rtol² = atol^2, rtol^2

  # Initialise variables with promoted types
  term = similar(a, T)
  fill!(term, 1)
  # Use recurrence relation: term_{k+1} = term_k * z / ((a+k) * (k+1))
  k = 0
  while k < maxiters
    k += 1
    converged = true
    @inbounds @simd for i in 1:n
      term[i] *= _frac(z, muladd(a[i] + k, k, -k)) # z / (a[i] + k - 1) * k)
      sum_val[i] += term[i]
      converged && (converged &= abs2(term[i]) < muladd(rtol², abs2(sum_val[i]), atol²))
    end
    converged && return sum_val
  end

  throw(ErrorException("No convergence reached"))
end
@inline _factor(a, loghalfz::Dual) = exp(a * loghalfz) / gamma(a + 1) # Can't do Complex(Dual)
@inline _factor(a, loghalfz) = exp(muladd(a, loghalfz, - loggamma(a + 1)))
function besselj(a::Number, z::Number; rtol=RTOL, atol=ATOL, maxiters=MAXITERS)
  T = float(promote_type(typeof(a), typeof(z)))
  halfz = z / 2
  loghalfz = log(Complex(halfz))
  return T(_factor(a, loghalfz) * hypergeom_0f1(a + 1, -halfz^2;
    atol=atol, rtol=rtol, maxiters=maxiters))
end
function besselj_v(a, z::Number; rtol=RTOL, atol=ATOL,
    maxiters=MAXITERS)
  T = float(promote_type(eltype(a), typeof(z)))
  halfz = z / 2
  loghalfz = log(Complex(halfz))
  output = T.(hypergeom_0f1(a .+ 1, -halfz^2;
    atol=atol, rtol=rtol, maxiters=maxiters))
  @inbounds @simd for i in 1:length(a)
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
