// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <string>

#include "Evolution/Systems/NewtonianEuler/Limiters/VariablesToLimit.hpp"
#include "Framework/TestCreation.hpp"
#include "Utilities/GetOutput.hpp"

SPECTRE_TEST_CASE(
    "Unit.Evolution.Systems.NewtonianEuler.Limiters.VariablesToLimit",
    "[Limiters][Unit]") {
  CHECK(NewtonianEuler::Limiters::VariablesToLimit::Conserved ==
        TestHelpers::test_creation<NewtonianEuler::Limiters::VariablesToLimit>(
            "Conserved"));
  CHECK(NewtonianEuler::Limiters::VariablesToLimit::Characteristic ==
        TestHelpers::test_creation<NewtonianEuler::Limiters::VariablesToLimit>(
            "Characteristic"));

  CHECK(get_output(NewtonianEuler::Limiters::VariablesToLimit::Conserved) ==
        "Conserved");
  CHECK(
      get_output(NewtonianEuler::Limiters::VariablesToLimit::Characteristic) ==
      "Characteristic");
}

// [[OutputRegex, Failed to convert "BadVars" to VariablesToLimit]]
SPECTRE_TEST_CASE(
    "Unit.Evolution.Systems.NewtonianEuler.Limiters.VariablesToLimit.ParseErr",
    "[Limiters][Unit]") {
  ERROR_TEST();
  TestHelpers::test_creation<NewtonianEuler::Limiters::VariablesToLimit>(
      "BadVars");
}
