// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include "Framework/TestCreation.hpp"
#include "Parallel/Phase.hpp"
#include "Utilities/GetOutput.hpp"

SPECTRE_TEST_CASE("Unit.Parallel.Phase", "[Parallel][Unit]") {
  // These two variables must correspond to the first and last
  // enum values of Parallel::Phase for the test to work properly
  const Parallel::Phase first_enum = Parallel::Phase::Cleanup;
  const Parallel::Phase last_enum = Parallel::Phase::WriteCheckpoint;

  using enum_t = std::underlying_type_t<Parallel::Phase>;
  REQUIRE(enum_t(0) == static_cast<enum_t>(first_enum));
  const auto known_phases = Parallel::known_phases();
  REQUIRE(enum_t(known_phases.size()) == static_cast<enum_t>(last_enum) + 1);

  for (const auto phase : known_phases) {
    CHECK(phase ==
          TestHelpers::test_creation<Parallel::Phase>(get_output(phase)));
  }

  CHECK_THROWS_WITH(
      ([]() {
        TestHelpers::test_creation<Parallel::Phase>("Bad phase name");
      }()),
      Catch::Contains(
          "Failed to convert \"Bad phase name\" to Parallel::Phase."));
}
