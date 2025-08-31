module GeneralBesselj

using DualNumbers, SpecialFunctions

const ATOL=0.0
const RTOL=1e-12
const MAXITERS=2^10

function hypergeom_0f1_fast(b::Number, z; atol=ATOL, rtol=RTOL, maxiters=MAXITERS)
  T = promote_type(typeof(b), typeof(z))

  (!(typeof(b) <: Dual) && isinteger(b)) && return T(SpecialFunctions.besselj(Int(b), z))

  # Handle special cases
  z == 0 && return one(T)
  
  # For negative integer b, function is undefined
  if isinteger(b) && b <= 0
    return iseven(b) ? T(Inf) : T(-Inf)
  end
  
  # Initialise variables with promoted types
  sum_val = one(T)
  term = one(T)
  # Use recurrence relation: term_{k+1} = term_k * z / ((b+k) * (k+1))
  k = 0
  while k < maxiters
    k += 1
    term *= z / ((b + k - 1) * k)
    sum_val += term
    abs(term) < max(rtol * abs(sum_val), atol) && return sum_val
  end
  
  throw(Methoderror("No convergence reached"))
end

function hypergeom_0f1_fast(b::AbstractVector, z; atol=ATOL, rtol=RTOL, maxiters=MAXITERS)
  T = promote_type(eltype(b), typeof(z))
  n = length(b)

  (!(eltype(b) <: Dual) && all(isinteger, b)) && return T(SpecialFunctions.besselj.(Int.(b), z))

  # Handle special cases
  sum_val = similar(b, T)
  fill!(sum_val, 1)
  z == 0 && return sum_val
   

  # For negative integer b, function is undefined
  if all(isinteger, b) && all(i->i<=0, b)
    for (i, bi) in enumerate(b)
      sum_val[i] = iseven(bi) ? T(Inf) : T(-Inf)
    end
    return sum_val
  end

  if any(i->isinteger(i) && i <= 0, b)
    throw(ErrorException("All or none of b must be positive integer"))
  end

  # Initialise variables with promoted types
  term = similar(b, T)
  fill!(term, 1)
  # Use recurrence relation: term_{k+1} = term_k * z / ((b+k) * (k+1))
  k = 0
  while k < maxiters
    k += 1
    @. term *= z / ((b + k - 1) * k)
    @. sum_val += term
    all(i->abs(term[i]) < max(rtol * abs(sum_val[i]), atol), 1:n) && return sum_val
  end

  throw(ErrorException("No convergence reached"))
end
_factor(z::Dual, a) = (z / 2)^a / gamma(a + 1) # Can't do Complex(Dual)
_factor(a, z) = exp(a * log(Complex(z) / 2) - loggamma(a + 1))
function besselj(a, z; rtol=RTOL, atol=ATOL, maxiters=MAXITERS)
  T = promote_type(typeof(a), typeof(z))
  return T(_factor(a, z) * hypergeom_0f1_fast(a + 1, -z^2 / 4;
    atol=atol, rtol=rtol, maxiters=maxiters))
end
function besselj(a::AbstractVector, z; rtol=RTOL, atol=ATOL, maxiters=MAXITERS)
  T = promote_type(eltype(a), typeof(z))
  return T.(_factor.(a, z)) .* hypergeom_0f1_fast(a .+ 1, -z^2 / 4;
    atol=atol, rtol=rtol, maxiters=maxiters)
end

end # module GeneralBesselj
