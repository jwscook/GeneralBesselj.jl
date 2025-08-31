using DualNumbers, ForwardDiff, GeneralBesselj, HypergeometricFunctions, Random, SpecialFunctions, Test

const NTESTS = 1000
Random.seed!(0)

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
      a = randn(ComplexF64) * 10
      z = randn(ComplexF64) * 10
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
      result = GeneralBesselj.vbesselj(a, z)
      @test result ≈ expected
    end
  end

  @testset "Ensure besselj composes with complex indices and Duals" begin
    @test DualNumbers.dualpart(besselj(1.0 + im, Dual(1.0, 1))) ≈
      (besselj(0+im, 1.0) - besselj(2.0+im, 1.0)) / 2
  end

  @testset "Duals vs ForwardDiff" begin
    for n in (3, 4, -3, -4), x in (-5.0, 5.0)
      @test DualNumbers.dualpart(besselj(n, Dual(x, 1))) ≈
        ForwardDiff.derivative(z->besselj(n, z), x)
    end
    for n in (3, 4), x in (-5.0, 5.0)
      @test DualNumbers.dualpart(n^Dual(x, 1)) ≈
        ForwardDiff.derivative(z->n^z, x)
    end
  end
end
