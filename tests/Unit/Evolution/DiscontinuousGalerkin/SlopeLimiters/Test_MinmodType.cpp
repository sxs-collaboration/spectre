// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <string>

#include "Evolution/DiscontinuousGalerkin/SlopeLimiters/MinmodType.hpp"
#include "Utilities/GetOutput.hpp"
#include "tests/Unit/TestCreation.hpp"

SPECTRE_TEST_CASE("Unit.Evolution.DG.SlopeLimiters.MinmodType",
                  "[SlopeLimiters][Unit]") {
  CHECK(Limiters::MinmodType::LambdaPi1 ==
        test_enum_creation<Limiters::MinmodType>("LambdaPi1"));
  CHECK(Limiters::MinmodType::LambdaPiN ==
        test_enum_creation<Limiters::MinmodType>("LambdaPiN"));
  CHECK(Limiters::MinmodType::Muscl ==
        test_enum_creation<Limiters::MinmodType>("Muscl"));

  CHECK(get_output(Limiters::MinmodType::LambdaPi1) == "LambdaPi1");
  CHECK(get_output(Limiters::MinmodType::LambdaPiN) == "LambdaPiN");
  CHECK(get_output(Limiters::MinmodType::Muscl) == "Muscl");
}

// [[OutputRegex, Failed to convert "BadType" to MinmodType]]
SPECTRE_TEST_CASE("Unit.Evolution.DG.SlopeLimiters.MinmodType.OptionParseError",
                  "[SlopeLimiters][Unit]") {
  ERROR_TEST();
  test_enum_creation<Limiters::MinmodType>("BadType");
}
