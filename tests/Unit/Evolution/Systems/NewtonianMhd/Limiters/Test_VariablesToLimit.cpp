// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <string>

#include "Evolution/Systems/NewtonianMhd/Limiters/VariablesToLimit.hpp"
#include "Framework/TestCreation.hpp"
#include "Utilities/GetOutput.hpp"

SPECTRE_TEST_CASE(
    "Unit.Evolution.Systems.NewtonianMhd.Limiters.VariablesToLimit",
    "[Limiters][Unit]") {
  CHECK(NewtonianMhd::Limiters::VariablesToLimit::Conserved ==
        TestHelpers::test_creation<NewtonianMhd::Limiters::VariablesToLimit>(
            "Conserved"));
  CHECK(NewtonianMhd::Limiters::VariablesToLimit::Characteristic ==
        TestHelpers::test_creation<NewtonianMhd::Limiters::VariablesToLimit>(
            "Characteristic"));
  CHECK(NewtonianMhd::Limiters::VariablesToLimit::NumericalCharacteristic ==
        TestHelpers::test_creation<NewtonianMhd::Limiters::VariablesToLimit>(
            "NumericalCharacteristic"));

  CHECK(get_output(NewtonianMhd::Limiters::VariablesToLimit::Conserved) ==
        "Conserved");
  CHECK(get_output(NewtonianMhd::Limiters::VariablesToLimit::Characteristic) ==
        "Characteristic");
  CHECK(
      get_output(
          NewtonianMhd::Limiters::VariablesToLimit::NumericalCharacteristic) ==
      "NumericalCharacteristic");

  CHECK_THROWS_WITH(
      (TestHelpers::test_creation<NewtonianMhd::Limiters::VariablesToLimit>(
          "BadVars")),
      Catch::Matchers::ContainsSubstring(
          "Failed to convert \"BadVars\" to VariablesToLimit"));
}
