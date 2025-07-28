// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <vector>

#include "Framework/TestCreation.hpp"
#include "ParallelAlgorithms/Amr/Criteria/Type.hpp"
#include "Utilities/GetOutput.hpp"
#include "Utilities/MakeString.hpp"

SPECTRE_TEST_CASE("Unit.Amr.Criteria.Type", "[Unit][ParallelAlgorithms]") {
  CHECK(get_output(amr::Criteria::Type::h) == "h");
  CHECK(get_output(amr::Criteria::Type::p) == "p");

  const std::vector known_amr_types{amr::Criteria::Type::h,
                                    amr::Criteria::Type::p};

  for (const auto type : known_amr_types) {
    CHECK(type ==
          TestHelpers::test_creation<amr::Criteria::Type>(get_output(type)));
  }

  CHECK_THROWS_WITH(
      ([]() {
        TestHelpers::test_creation<amr::Criteria::Type>("Bad type name");
      }()),
      Catch::Matchers::ContainsSubstring(
          MakeString{} << "Failed to convert \"Bad type name\" to "
                          "amr::Criteria::Type.\nMust be one of "
                       << known_amr_types << "."));
}
