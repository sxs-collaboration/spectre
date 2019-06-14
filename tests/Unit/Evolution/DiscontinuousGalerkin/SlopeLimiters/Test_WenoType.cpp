// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <string>

#include "Evolution/DiscontinuousGalerkin/SlopeLimiters/WenoType.hpp"
#include "Utilities/GetOutput.hpp"
#include "tests/Unit/TestCreation.hpp"

SPECTRE_TEST_CASE("Unit.Evolution.DG.SlopeLimiters.WenoType",
                  "[SlopeLimiters][Unit]") {
  CHECK(SlopeLimiters::WenoType::Hweno ==
        test_enum_creation<SlopeLimiters::WenoType>("Hweno"));
  CHECK(SlopeLimiters::WenoType::SimpleWeno ==
        test_enum_creation<SlopeLimiters::WenoType>("SimpleWeno"));

  CHECK(get_output(SlopeLimiters::WenoType::Hweno) == "Hweno");
  CHECK(get_output(SlopeLimiters::WenoType::SimpleWeno) == "SimpleWeno");
}

// [[OutputRegex, Failed to convert "BadType" to WenoType]]
SPECTRE_TEST_CASE("Unit.Evolution.DG.SlopeLimiters.WenoType.OptionParseError",
                  "[SlopeLimiters][Unit]") {
  ERROR_TEST();
  test_enum_creation<SlopeLimiters::WenoType>("BadType");
}
