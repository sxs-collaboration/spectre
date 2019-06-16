// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <string>

#include "Evolution/DiscontinuousGalerkin/SlopeLimiters/WenoType.hpp"
#include "Utilities/GetOutput.hpp"
#include "tests/Unit/TestCreation.hpp"

SPECTRE_TEST_CASE("Unit.Evolution.DG.Limiters.WenoType", "[Limiters][Unit]") {
  CHECK(Limiters::WenoType::Hweno ==
        test_enum_creation<Limiters::WenoType>("Hweno"));
  CHECK(Limiters::WenoType::SimpleWeno ==
        test_enum_creation<Limiters::WenoType>("SimpleWeno"));

  CHECK(get_output(Limiters::WenoType::Hweno) == "Hweno");
  CHECK(get_output(Limiters::WenoType::SimpleWeno) == "SimpleWeno");
}

// [[OutputRegex, Failed to convert "BadType" to WenoType]]
SPECTRE_TEST_CASE("Unit.Evolution.DG.Limiters.WenoType.OptionParseError",
                  "[Limiters][Unit]") {
  ERROR_TEST();
  test_enum_creation<Limiters::WenoType>("BadType");
}
