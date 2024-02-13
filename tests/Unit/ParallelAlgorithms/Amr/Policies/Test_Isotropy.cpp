// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <vector>

#include "Framework/TestCreation.hpp"
#include "ParallelAlgorithms/Amr/Policies/Isotropy.hpp"
#include "Utilities/GetOutput.hpp"
#include "Utilities/MakeString.hpp"

SPECTRE_TEST_CASE("Unit.ParallelAlgorithms.Amr.Isotropy",
                  "[ParallelAlgorithms][Unit]") {
  CHECK(get_output(amr::Isotropy::Anisotropic) == "Anisotropic");
  CHECK(get_output(amr::Isotropy::Isotropic) == "Isotropic");

  const std::vector known_amr_isotropies{amr::Isotropy::Anisotropic,
                                         amr::Isotropy::Isotropic};

  for (const auto isotropy : known_amr_isotropies) {
    CHECK(isotropy ==
          TestHelpers::test_creation<amr::Isotropy>(get_output(isotropy)));
  }

  CHECK_THROWS_WITH(
      ([]() {
        TestHelpers::test_creation<amr::Isotropy>("Bad isotropy name");
      }()),
      Catch::Matchers::ContainsSubstring(
          MakeString{} << "Failed to convert \"Bad isotropy name\" to "
                          "amr::Isotropy.\nMust be one of "
                       << known_amr_isotropies << "."));
}
