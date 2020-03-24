// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <string>

#include "Evolution/DiscontinuousGalerkin/Limiters/MinmodType.hpp"
#include "Framework/TestCreation.hpp"
#include "Utilities/GetOutput.hpp"

SPECTRE_TEST_CASE("Unit.Evolution.DG.Limiters.MinmodType", "[Limiters][Unit]") {
  CHECK(Limiters::MinmodType::LambdaPi1 ==
        TestHelpers::test_creation<Limiters::MinmodType>("LambdaPi1"));
  CHECK(Limiters::MinmodType::LambdaPiN ==
        TestHelpers::test_creation<Limiters::MinmodType>("LambdaPiN"));
  CHECK(Limiters::MinmodType::Muscl ==
        TestHelpers::test_creation<Limiters::MinmodType>("Muscl"));

  CHECK(get_output(Limiters::MinmodType::LambdaPi1) == "LambdaPi1");
  CHECK(get_output(Limiters::MinmodType::LambdaPiN) == "LambdaPiN");
  CHECK(get_output(Limiters::MinmodType::Muscl) == "Muscl");
}

// [[OutputRegex, Failed to convert "BadType" to MinmodType]]
SPECTRE_TEST_CASE("Unit.Evolution.DG.Limiters.MinmodType.OptionParseError",
                  "[Limiters][Unit]") {
  ERROR_TEST();
  TestHelpers::test_creation<Limiters::MinmodType>("BadType");
}
