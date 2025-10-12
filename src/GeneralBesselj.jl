module GeneralBesselj

export besselj_v

using DualNumbers, SpecialFunctions
import SpecialFunctions: besselj

const MAXITERS=2^10

#function _besselj_halfz(a, halfz::Number, loghalfz; maxiters=MAXITERS)
#  return _factor(a, loghalfz) * HypergeometricFunctions.pFq(
#    Tuple(()), (a + 1,), -halfz^2; kmax=maxiters)
#end

function _besselj_halfz(a, halfz::Number, loghalfz, rtol=10eps(); maxiters=MAXITERS)
  # Compute J_a(z) using 0F1 series
  # J_a(z) = (z/2)^a / Γ(a+1) * 0F1(;a+1;-z²/4)
  term = 1.0 / gamma(a + 1)
  result = term
  _z²_4 = -halfz^2
  for n in 1:maxiters
    term *= _z²_4 / (n * (a + n))
    result += term
    abs2(term) < abs2(rtol * result) && break
  end
  return exp(a * loghalfz + log(result))
end


function besselj(a::Number, z::Number, rtol=10eps(); maxiters=MAXITERS)
  T = float(promote_type(typeof(a), typeof(z)))
  (!(typeof(a) <: Dual) && isinteger(a)) && return T(besselj(round(Int, real(a)), z))
  halfz = z / 2
  loghalfz = log(halfz * (1 + 0im))
  return _besselj_halfz(a, halfz, loghalfz, rtol; maxiters=maxiters)
end
function besselj_v(a, z::Number, rtol=10eps(); maxiters=MAXITERS)
  T = float(promote_type(eltype(a), typeof(z)))
  halfz = z / 2
  loghalfz = log(halfz * (1 + 0im))
  output = similar(a, T)
  @inbounds @simd for i in 1:length(a)
    if !(eltype(a) <: Dual) && isinteger(a[i])
      output[i] = besselj(round(Int, real(a[i])), z)
    else
      output[i] = _besselj_halfz(a[i], halfz, loghalfz, rtol; maxiters=maxiters)
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
