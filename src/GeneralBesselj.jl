module GeneralBesselj

export besselj_v

using DualNumbers, SpecialFunctions
import SpecialFunctions: besselj

const ATOL=0.0
const RTOL=1e-12
const MAXITERS=2^10

function hypergeom_0f1_fast(a::Number, z; atol=ATOL, rtol=RTOL, maxiters=MAXITERS)
  T = promote_type(typeof(a), typeof(z))

  (!(typeof(a) <: Dual) && isinteger(a)) && return T(SpecialFunctions.besselj(Int(a), z))

  # Handle special cases
  z == 0 && return one(T)
  
  # For negative integer a, function is undefined
  if isinteger(a) && a <= 0
    return iseven(a) ? T(Inf) : T(-Inf)
  end

  atol², rtol² = atol^2, rtol^2
  
  # Initialise variables with promoted types
  sum_val = one(T)
  term = one(T)
  # Use recurrence relation: term_{k+1} = term_k * z / ((a+k) * (k+1))
  k = 0
  while k < maxiters
    k += 1
    term *= z / ((a + k - 1) * k)
    sum_val += term
    abs2(term) < max(rtol² * abs2(sum_val), atol²) && return sum_val
  end
  
  throw(Methoderror("No convergence reached"))
end

function hypergeom_0f1_fast(a::AbstractVector, z; atol=ATOL, rtol=RTOL, maxiters=MAXITERS)
  T = promote_type(eltype(a), typeof(z))
  n = length(a)

  (!(eltype(a) <: Dual) && all(isinteger, a)) && return T(SpecialFunctions.besselj.(Int.(a), z))

  if any(i->isinteger(i) && i <= 0, a)
    throw(ErrorException("All orders must be positive if they are integers"))
  end

  # Handle special cases
  sum_val = similar(a, T)
  fill!(sum_val, 1)
  z == 0 && return sum_val

  # For negative integer a, function is undefined
  if all(isinteger, a) && all(i->i<=0, a)
    for (i, ai) in enumerate(a)
      sum_val[i] = iseven(ai) ? T(Inf) : T(-Inf)
    end
    return sum_val
  end

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
      term[i] *= z / ((a[i] + k - 1) * k)
      sum_val[i] += term[i]
      converged && (converged &= abs2(term[i]) < max(rtol² * abs2(sum_val[i]), atol²))
    end
    converged && return sum_val
  end

  throw(ErrorException("No convergence reached"))
end
_factor(a, z::Dual) = (z / 2)^a / gamma(a + 1) # Can't do Complex(Dual)
function _factor(a, z)
  return if abs2(a) > 28900 # gamma(±170) ∼ 10^(±308), 170^2 = 28900
    exp(a * log(Complex(z) / 2) - loggamma(a + 1))
  else
    (z / 2)^a / gamma(a + 1)
  end
end
function besselj(a, z; rtol=RTOL, atol=ATOL, maxiters=MAXITERS)
  T = promote_type(typeof(a), typeof(z))
  return T(_factor(a, z) * hypergeom_0f1_fast(a + 1, -z^2 / 4;
    atol=atol, rtol=rtol, maxiters=maxiters))
end
function besselj_v(a::AbstractVector{T}, z; rtol=RTOL, atol=ATOL, maxiters=MAXITERS
    ) where {T}
  U = promote_type(eltype(a), typeof(z))
  return U.(_factor.(a, z)) .* hypergeom_0f1_fast(a .+ 1, -z^2 / 4;
    atol=atol, rtol=rtol, maxiters=maxiters)
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
