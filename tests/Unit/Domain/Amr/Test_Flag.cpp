// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <vector>

#include "Domain/Amr/Flag.hpp"
#include "Framework/TestCreation.hpp"
#include "Utilities/GetOutput.hpp"
#include "Utilities/MakeString.hpp"

SPECTRE_TEST_CASE("Unit.Domain.Amr.Flag", "[Domain][Unit]") {
  CHECK(get_output(amr::domain::Flag::Undefined) == "Undefined");
  CHECK(get_output(amr::domain::Flag::Join) == "Join");
  CHECK(get_output(amr::domain::Flag::DecreaseResolution) ==
        "DecreaseResolution");
  CHECK(get_output(amr::domain::Flag::DoNothing) == "DoNothing");
  CHECK(get_output(amr::domain::Flag::IncreaseResolution) ==
        "IncreaseResolution");
  CHECK(get_output(amr::domain::Flag::Split) == "Split");

  const std::vector known_amr_flags{
      amr::domain::Flag::Undefined,          amr::domain::Flag::Join,
      amr::domain::Flag::DecreaseResolution, amr::domain::Flag::DoNothing,
      amr::domain::Flag::IncreaseResolution, amr::domain::Flag::Split};

  for (const auto flag : known_amr_flags) {
    CHECK(flag ==
          TestHelpers::test_creation<amr::domain::Flag>(get_output(flag)));
  }

  CHECK_THROWS_WITH(
      ([]() {
        TestHelpers::test_creation<amr::domain::Flag>("Bad flag name");
      }()),
      Catch::Contains(MakeString{} << "Failed to convert \"Bad flag name\" to "
                                      "amr::domain::Flag.\nMust be one of "
                                   << known_amr_flags << "."));
}
