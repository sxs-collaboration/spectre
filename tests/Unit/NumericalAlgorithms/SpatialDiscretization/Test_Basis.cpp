// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <vector>

#include "Framework/TestCreation.hpp"
#include "NumericalAlgorithms/SpatialDiscretization/Basis.hpp"
#include "Utilities/GetOutput.hpp"
#include "Utilities/MakeString.hpp"

SPECTRE_TEST_CASE("Unit.SpatialDiscreitization.Basis",
                  "[NumericalAlgorithms][Unit]") {
  CHECK(get_output(SpatialDiscretization::Basis::Legendre) == "Legendre");
  CHECK(get_output(SpatialDiscretization::Basis::Chebyshev) == "Chebyshev");
  CHECK(get_output(SpatialDiscretization::Basis::FiniteDifference) ==
        "FiniteDifference");
  CHECK(get_output(SpatialDiscretization::Basis::SphericalHarmonic) ==
        "SphericalHarmonic");

  const std::vector known_bases{
      SpatialDiscretization::Basis::Legendre,
      SpatialDiscretization::Basis::Chebyshev,
      SpatialDiscretization::Basis::FiniteDifference,
      SpatialDiscretization::Basis::SphericalHarmonic};

  for (const auto basis : known_bases) {
    CHECK(basis == TestHelpers::test_creation<SpatialDiscretization::Basis>(
                       get_output(basis)));
  }

  CHECK_THROWS_WITH(
      ([]() {
        TestHelpers::test_creation<SpatialDiscretization::Basis>(
            "Bad basis name");
      }()),
      Catch::Matchers::ContainsSubstring(
          MakeString{} << "Failed to convert \"Bad basis name\" to "
                          "SpatialDiscretization::Basis.\nMust be one of "
                       << known_bases << "."));
}
