using DualNumbers, GeneralBesselj, HypergeometricFunctions, SpecialFunctions, Test

const NTESTS = 10

function isapproxinteger(z::Complex, tol=eps())
  rz, iz = reim(z)
  a = round(Int, rz)
  isapprox(a, rz, rtol=tol, atol=tol) || return false
  isapprox(0, iz, rtol=tol, atol=tol) || return false
  return true
end
function testbesselj(a::T, z) where {T<:Complex}
  #exp(a * log(z/2)) is faster and more accurate than (z/2)^a
  if !(T <: Dual) && (isinteger(a) || isapproxinteger(a, eps()))
    return promote_type(T, typeof(z))(besselj(round(Int, real(a)), z))
  else
    return GeneralBesselj._factor(a, z) * HypergeometricFunctions.pFq(
      Tuple(()), (a + 1,), -z^2 / 4)
  end
end

@testset "GeneralBesselj" begin
  @testset "Float64" begin
    for i in 1:NTESTS 
      a = rand() * 10
      z = rand() * 10
      expected = SpecialFunctions.besselj(a, z)
      result = GeneralBesselj.besselj(a, z)
      @test result ≈ expected
    end
  end
  @testset "ComplexF64" begin
    for i in 1:NTESTS 
      a = rand(ComplexF64) * 10
      z = rand(ComplexF64) * 10
      expected = testbesselj(a, z)
      result = GeneralBesselj.besselj(a, z)
      @test result ≈ expected
    end
  end
  @testset "Vector order" begin
    for i in 1:NTESTS 
      a = rand(Float64, 4) * 10
      z = rand(ComplexF64) * 10
      expected = SpecialFunctions.besselj.(a, z)
      result = GeneralBesselj.besselj(a, z)
      @test result ≈ expected
    end
  end

end
