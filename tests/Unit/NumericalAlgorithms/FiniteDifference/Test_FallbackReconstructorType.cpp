// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <string>

#include "Framework/TestCreation.hpp"
#include "NumericalAlgorithms/FiniteDifference/FallbackReconstructorType.hpp"
#include "Utilities/GetOutput.hpp"

namespace fd::reconstruction {

SPECTRE_TEST_CASE("Unit.FiniteDifference.FallbackReconstructorType",
                  "[Unit][NumericalAlgorithms]") {
  CHECK(FallbackReconstructorType::Minmod ==
        TestHelpers::test_creation<FallbackReconstructorType>("Minmod"));
  CHECK(FallbackReconstructorType::MonotonisedCentral ==
        TestHelpers::test_creation<FallbackReconstructorType>(
            "MonotonisedCentral"));
  CHECK(FallbackReconstructorType::None ==
        TestHelpers::test_creation<FallbackReconstructorType>("None"));

  CHECK(get_output(FallbackReconstructorType::Minmod) == "Minmod");
  CHECK(get_output(FallbackReconstructorType::MonotonisedCentral) ==
        "MonotonisedCentral");
  CHECK(get_output(FallbackReconstructorType::None) == "None");

  CHECK_THROWS_WITH(
      ([]() {
        TestHelpers::test_creation<FallbackReconstructorType>("BadType");
      })(),
      Catch::Contains(
          "Failed to convert \"BadType\" to FallbackReconstructorType"));
}

}  // namespace fd::reconstruction
