module GeneralBesselj

export besselj_v

using DualNumbers, HypergeometricFunctions, SpecialFunctions
import SpecialFunctions: besselj

const MAXITERS=2^10

@inline _factor(a, loghalfz::Dual) = exp(a * loghalfz) / gamma(a + 1) # Can't do Complex(Dual)
@inline _factor(a, loghalfz) = exp(muladd(a, loghalfz, - loggamma(a + 1)))

function _besselj_halfz(a, halfz::Number, loghalfz; maxiters=MAXITERS)
  return _factor(a, loghalfz) * HypergeometricFunctions.pFq(
    Tuple(()), (a + 1,), -halfz^2; kmax=maxiters)
end

function besselj(a::Number, z::Number; maxiters=MAXITERS)
  T = float(promote_type(typeof(a), typeof(z)))
  (!(typeof(a) <: Dual) && isinteger(a)) && return T(besselj(round(Int, real(a)), z))
  halfz = z / 2
  loghalfz = log(halfz * (1 + 0im))
  return _besselj_halfz(a, halfz, loghalfz; maxiters=maxiters)
end
function besselj_v(a, z::Number; maxiters=MAXITERS)
  T = float(promote_type(eltype(a), typeof(z)))
  halfz = z / 2
  loghalfz = log(halfz * (1 + 0im))
  output = similar(a, T)
  @inbounds @simd for i in 1:length(a)
    if !(eltype(a) <: Dual) && isinteger(a[i])
      output[i] = besselj(round(Int, real(a[i])), z)
    else
      output[i] = _besselj_halfz(a[i], halfz, loghalfz; maxiters=maxiters)
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
