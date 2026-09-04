// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include "NumericalAlgorithms/Spectral/Basis.hpp"
#include "NumericalAlgorithms/Spectral/Limits.hpp"
#include "NumericalAlgorithms/Spectral/Quadrature.hpp"

namespace Spectral {
namespace {
void test() {
  static_assert(limits::min_spherical_harmonic_mode == 2);
  static_assert(limits::max_spherical_harmonic_mode == 40);
  static_assert(limits::max_fourier_mode == 40);
  static_assert(limits::max_i1_polynomial_mode == 20);

  static_assert(limits::max(Basis::Chebyshev, Quadrature::GaussLobatto) == 21);
  static_assert(limits::max(Basis::Chebyshev, Quadrature::Gauss) == 21);
  static_assert(limits::max(Basis::Legendre, Quadrature::GaussLobatto) == 21);
  static_assert(limits::max(Basis::Legendre, Quadrature::Gauss) == 21);
  static_assert(
      limits::max(Basis::FiniteDifference, Quadrature::CellCentered) == 42);
  static_assert(
      limits::max(Basis::FiniteDifference, Quadrature::FaceCentered) == 42);
  static_assert(limits::max(Basis::SphericalHarmonic, Quadrature::Gauss) == 41);
  static_assert(
      limits::max(Basis::SphericalHarmonic, Quadrature::Equiangular) == 81);
  static_assert(limits::max(Basis::Fourier, Quadrature::Equiangular) == 81);
  static_assert(limits::max(Basis::ZernikeB1, Quadrature::GaussRadauUpper) ==
                21);
  static_assert(limits::max(Basis::ZernikeB2, Quadrature::GaussRadauUpper) ==
                21);
  static_assert(limits::max(Basis::ZernikeB2, Quadrature::Equiangular) == 81);
  static_assert(limits::max(Basis::ZernikeB3, Quadrature::GaussRadauUpper) ==
                21);
  static_assert(limits::max(Basis::ZernikeB3, Quadrature::Gauss) == 41);
  static_assert(limits::max(Basis::ZernikeB3, Quadrature::Equiangular) == 81);
  static_assert(limits::max(Basis::Cartoon, Quadrature::AxialSymmetry) == 1);
  static_assert(limits::max(Basis::Cartoon, Quadrature::SphericalSymmetry) ==
                1);
  // Some invalid pairs should have a limit of 0
  static_assert(
      limits::max(Basis::SphericalHarmonic, Quadrature::GaussLobatto) == 0);

  CHECK(limits::max(Basis::Chebyshev, Quadrature::GaussLobatto) == 21);
  CHECK(limits::max(Basis::Chebyshev, Quadrature::Gauss) == 21);
  CHECK(limits::max(Basis::Legendre, Quadrature::GaussLobatto) == 21);
  CHECK(limits::max(Basis::Legendre, Quadrature::Gauss) == 21);
  CHECK(limits::max(Basis::FiniteDifference, Quadrature::CellCentered) == 42);
  CHECK(limits::max(Basis::FiniteDifference, Quadrature::FaceCentered) == 42);
  CHECK(limits::max(Basis::SphericalHarmonic, Quadrature::Gauss) == 41);
  CHECK(limits::max(Basis::SphericalHarmonic, Quadrature::Equiangular) == 81);
  CHECK(limits::max(Basis::Fourier, Quadrature::Equiangular) == 81);
  CHECK(limits::max(Basis::ZernikeB1, Quadrature::GaussRadauUpper) == 21);
  CHECK(limits::max(Basis::ZernikeB2, Quadrature::GaussRadauUpper) == 21);
  CHECK(limits::max(Basis::ZernikeB2, Quadrature::Equiangular) == 81);
  CHECK(limits::max(Basis::ZernikeB3, Quadrature::GaussRadauUpper) == 21);
  CHECK(limits::max(Basis::ZernikeB3, Quadrature::Gauss) == 41);
  CHECK(limits::max(Basis::ZernikeB3, Quadrature::Equiangular) == 81);
  CHECK(limits::max(Basis::Cartoon, Quadrature::AxialSymmetry) == 1);
  CHECK(limits::max(Basis::Cartoon, Quadrature::SphericalSymmetry) == 1);
  // Some invalid pairs should have a limit of 0
  CHECK(limits::max(Basis::SphericalHarmonic, Quadrature::GaussLobatto) == 0);

  static_assert(limits::min(Basis::Chebyshev, Quadrature::GaussLobatto) == 2);
  static_assert(limits::min(Basis::Chebyshev, Quadrature::Gauss) == 1);
  static_assert(limits::min(Basis::Legendre, Quadrature::GaussLobatto) == 2);
  static_assert(limits::min(Basis::Legendre, Quadrature::Gauss) == 1);
  static_assert(
      limits::min(Basis::FiniteDifference, Quadrature::CellCentered) == 1);
  static_assert(
      limits::min(Basis::FiniteDifference, Quadrature::FaceCentered) == 2);
  static_assert(limits::min(Basis::SphericalHarmonic, Quadrature::Gauss) == 3);
  static_assert(
      limits::min(Basis::SphericalHarmonic, Quadrature::Equiangular) == 5);
  static_assert(limits::min(Basis::Fourier, Quadrature::Equiangular) == 1);
  static_assert(limits::min(Basis::ZernikeB1, Quadrature::GaussRadauUpper) ==
                1);
  static_assert(limits::min(Basis::ZernikeB2, Quadrature::GaussRadauUpper) ==
                1);
  static_assert(limits::min(Basis::ZernikeB2, Quadrature::Equiangular) == 1);
  static_assert(limits::min(Basis::ZernikeB3, Quadrature::GaussRadauUpper) ==
                2);
  static_assert(limits::min(Basis::ZernikeB3, Quadrature::Gauss) == 3);
  static_assert(limits::min(Basis::ZernikeB3, Quadrature::Equiangular) == 5);
  static_assert(limits::min(Basis::Cartoon, Quadrature::AxialSymmetry) == 1);
  static_assert(limits::min(Basis::Cartoon, Quadrature::SphericalSymmetry) ==
                1);

  CHECK(limits::min(Basis::Chebyshev, Quadrature::GaussLobatto) == 2);
  CHECK(limits::min(Basis::Chebyshev, Quadrature::Gauss) == 1);
  CHECK(limits::min(Basis::Legendre, Quadrature::GaussLobatto) == 2);
  CHECK(limits::min(Basis::Legendre, Quadrature::Gauss) == 1);
  CHECK(limits::min(Basis::FiniteDifference, Quadrature::CellCentered) == 1);
  CHECK(limits::min(Basis::FiniteDifference, Quadrature::FaceCentered) == 2);
  CHECK(limits::min(Basis::SphericalHarmonic, Quadrature::Gauss) == 3);
  CHECK(limits::min(Basis::SphericalHarmonic, Quadrature::Equiangular) == 5);
  CHECK(limits::min(Basis::Fourier, Quadrature::Equiangular) == 1);
  CHECK(limits::min(Basis::ZernikeB1, Quadrature::GaussRadauUpper) == 1);
  CHECK(limits::min(Basis::ZernikeB2, Quadrature::GaussRadauUpper) == 1);
  CHECK(limits::min(Basis::ZernikeB2, Quadrature::Equiangular) == 1);
  CHECK(limits::min(Basis::ZernikeB3, Quadrature::GaussRadauUpper) == 2);
  CHECK(limits::min(Basis::ZernikeB3, Quadrature::Gauss) == 3);
  CHECK(limits::min(Basis::ZernikeB3, Quadrature::Equiangular) == 5);
  CHECK(limits::min(Basis::Cartoon, Quadrature::AxialSymmetry) == 1);
  CHECK(limits::min(Basis::Cartoon, Quadrature::SphericalSymmetry) == 1);
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Spectral.Limits", "[NumericalAlgorithms][Unit]") {
  test();
}
}  // namespace Spectral
