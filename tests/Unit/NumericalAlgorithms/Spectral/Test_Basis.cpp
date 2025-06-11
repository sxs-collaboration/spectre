// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <vector>

#include "Framework/TestCreation.hpp"
#include "NumericalAlgorithms/Spectral/Basis.hpp"
#include "Utilities/GetOutput.hpp"
#include "Utilities/MakeString.hpp"

SPECTRE_TEST_CASE("Unit.Spectral.Basis", "[NumericalAlgorithms][Unit]") {
  CHECK(get_output(Spectral::Basis::Uninitialized) == "Uninitialized");
  CHECK(get_output(Spectral::Basis::Legendre) == "Legendre");
  CHECK(get_output(Spectral::Basis::Chebyshev) == "Chebyshev");
  CHECK(get_output(Spectral::Basis::FiniteDifference) == "FiniteDifference");
  CHECK(get_output(Spectral::Basis::SphericalHarmonic) == "SphericalHarmonic");
  CHECK(get_output(Spectral::Basis::Fourier) == "Fourier");
  CHECK(get_output(Spectral::Basis::B2Marcus) == "B2Marcus");
  CHECK(get_output(Spectral::Basis::Cartoon) == "Cartoon");

  for (const auto basis : Spectral::all_bases()) {
    CHECK(basis ==
          TestHelpers::test_creation<Spectral::Basis>(get_output(basis)));
    CHECK(basis == Spectral::to_basis(get_output(basis)));
  }

  CHECK_THROWS_WITH(
      ([]() {
        TestHelpers::test_creation<Spectral::Basis>("Bad basis name");
      }()),
      Catch::Matchers::ContainsSubstring(
          MakeString{} << "Failed to convert \"Bad basis name\" to "
                          "Spectral::Basis.\nMust be one of "
                       << Spectral::all_bases() << "."));
}
