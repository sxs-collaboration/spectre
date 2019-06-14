// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <string>

#include "Evolution/DiscontinuousGalerkin/SlopeLimiters/MinmodType.hpp"
#include "Utilities/GetOutput.hpp"
#include "tests/Unit/TestCreation.hpp"

SPECTRE_TEST_CASE("Unit.Evolution.DG.SlopeLimiters.MinmodType",
                  "[SlopeLimiters][Unit]") {
  CHECK(SlopeLimiters::MinmodType::LambdaPi1 ==
        test_enum_creation<SlopeLimiters::MinmodType>("LambdaPi1"));
  CHECK(SlopeLimiters::MinmodType::LambdaPiN ==
        test_enum_creation<SlopeLimiters::MinmodType>("LambdaPiN"));
  CHECK(SlopeLimiters::MinmodType::Muscl ==
        test_enum_creation<SlopeLimiters::MinmodType>("Muscl"));

  CHECK(get_output(SlopeLimiters::MinmodType::LambdaPi1) == "LambdaPi1");
  CHECK(get_output(SlopeLimiters::MinmodType::LambdaPiN) == "LambdaPiN");
  CHECK(get_output(SlopeLimiters::MinmodType::Muscl) == "Muscl");
}

// [[OutputRegex, Failed to convert "BadType" to MinmodType]]
SPECTRE_TEST_CASE("Unit.Evolution.DG.SlopeLimiters.MinmodType.OptionParseError",
                  "[SlopeLimiters][Unit]") {
  ERROR_TEST();
  test_enum_creation<SlopeLimiters::MinmodType>("BadType");
}
