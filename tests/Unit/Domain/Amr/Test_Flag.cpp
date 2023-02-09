// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <vector>

#include "Domain/Amr/Flag.hpp"
#include "Framework/TestCreation.hpp"
#include "Utilities/GetOutput.hpp"
#include "Utilities/MakeString.hpp"

SPECTRE_TEST_CASE("Unit.Domain.Amr.Flag", "[Domain][Unit]") {
  CHECK(get_output(amr::Flag::Undefined) == "Undefined");
  CHECK(get_output(amr::Flag::Join) == "Join");
  CHECK(get_output(amr::Flag::DecreaseResolution) == "DecreaseResolution");
  CHECK(get_output(amr::Flag::DoNothing) == "DoNothing");
  CHECK(get_output(amr::Flag::IncreaseResolution) == "IncreaseResolution");
  CHECK(get_output(amr::Flag::Split) == "Split");

  const std::vector known_amr_flags{
      amr::Flag::Undefined,          amr::Flag::Join,
      amr::Flag::DecreaseResolution, amr::Flag::DoNothing,
      amr::Flag::IncreaseResolution, amr::Flag::Split};

  for (const auto flag : known_amr_flags) {
    CHECK(flag == TestHelpers::test_creation<amr::Flag>(get_output(flag)));
  }

  CHECK_THROWS_WITH(
      ([]() { TestHelpers::test_creation<amr::Flag>("Bad flag name"); }()),
      Catch::Contains(MakeString{} << "Failed to convert \"Bad flag name\" to "
                                      "amr::Flag.\nMust be one of "
                                   << known_amr_flags << "."));
}
