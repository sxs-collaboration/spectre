// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <vector>

#include "Framework/TestCreation.hpp"
#include "NumericalAlgorithms/SpatialDiscretization/Quadrature.hpp"
#include "Utilities/GetOutput.hpp"
#include "Utilities/MakeString.hpp"

SPECTRE_TEST_CASE("Unit.SpatialDiscreitization.Quadrature",
                  "[NumericalAlgorithms][Unit]") {
  CHECK(get_output(SpatialDiscretization::Quadrature::Gauss) == "Gauss");
  CHECK(get_output(SpatialDiscretization::Quadrature::GaussLobatto) ==
        "GaussLobatto");
  CHECK(get_output(SpatialDiscretization::Quadrature::CellCentered) ==
        "CellCentered");
  CHECK(get_output(SpatialDiscretization::Quadrature::FaceCentered) ==
        "FaceCentered");
  CHECK(get_output(SpatialDiscretization::Quadrature::Equiangular) ==
        "Equiangular");

  const std::vector known_quadratures{
      SpatialDiscretization::Quadrature::Gauss,
      SpatialDiscretization::Quadrature::GaussLobatto,
      SpatialDiscretization::Quadrature::CellCentered,
      SpatialDiscretization::Quadrature::FaceCentered,
      SpatialDiscretization::Quadrature::Equiangular};

  for (const auto quadrature : known_quadratures) {
    CHECK(quadrature ==
          TestHelpers::test_creation<SpatialDiscretization::Quadrature>(
              get_output(quadrature)));
  }

  CHECK_THROWS_WITH(
      ([]() {
        TestHelpers::test_creation<SpatialDiscretization::Quadrature>(
            "Bad quadrature name");
      }()),
      Catch::Matchers::ContainsSubstring(
          MakeString{} << "Failed to convert \"Bad quadrature name\" to "
                          "SpatialDiscretization::Quadrature.\nMust be one of "
                       << known_quadratures << "."));
}
