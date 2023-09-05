// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <vector>

#include "Framework/TestCreation.hpp"
#include "NumericalAlgorithms/Spectral/Quadrature.hpp"
#include "Utilities/GetOutput.hpp"
#include "Utilities/MakeString.hpp"

SPECTRE_TEST_CASE("Unit.SpatialDiscreitization.Quadrature",
                  "[NumericalAlgorithms][Unit]") {
  CHECK(get_output(Spectral::Quadrature::Gauss) == "Gauss");
  CHECK(get_output(Spectral::Quadrature::GaussLobatto) == "GaussLobatto");
  CHECK(get_output(Spectral::Quadrature::CellCentered) == "CellCentered");
  CHECK(get_output(Spectral::Quadrature::FaceCentered) == "FaceCentered");
  CHECK(get_output(Spectral::Quadrature::Equiangular) == "Equiangular");

  for (const auto quadrature : Spectral::all_quadratures()) {
    CHECK(quadrature == TestHelpers::test_creation<Spectral::Quadrature>(
                            get_output(quadrature)));
    CHECK(quadrature == Spectral::to_quadrature(get_output(quadrature)));
  }

  CHECK_THROWS_WITH(
      ([]() {
        TestHelpers::test_creation<Spectral::Quadrature>("Bad quadrature name");
      }()),
      Catch::Matchers::ContainsSubstring(
          MakeString{} << "Failed to convert \"Bad quadrature name\" to "
                          "Spectral::Quadrature.\nMust be one of "
                       << all_quadratures() << "."));
}
