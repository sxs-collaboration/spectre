// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <string>

#include "Evolution/DiscontinuousGalerkin/Limiters/WenoType.hpp"
#include "Framework/TestCreation.hpp"
#include "Utilities/GetOutput.hpp"

SPECTRE_TEST_CASE("Unit.Evolution.DG.Limiters.WenoType", "[Limiters][Unit]") {
  CHECK(Limiters::WenoType::Hweno ==
        TestHelpers::test_creation<Limiters::WenoType>("Hweno"));
  CHECK(Limiters::WenoType::SimpleWeno ==
        TestHelpers::test_creation<Limiters::WenoType>("SimpleWeno"));

  CHECK(get_output(Limiters::WenoType::Hweno) == "Hweno");
  CHECK(get_output(Limiters::WenoType::SimpleWeno) == "SimpleWeno");
}

// [[OutputRegex, Failed to convert "BadType" to WenoType]]
SPECTRE_TEST_CASE("Unit.Evolution.DG.Limiters.WenoType.OptionParseError",
                  "[Limiters][Unit]") {
  ERROR_TEST();
  TestHelpers::test_creation<Limiters::WenoType>("BadType");
}
