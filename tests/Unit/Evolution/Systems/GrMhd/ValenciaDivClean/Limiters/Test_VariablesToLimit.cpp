// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <string>

#include "Evolution/Systems/GrMhd/ValenciaDivClean/Limiters/VariablesToLimit.hpp"
#include "Framework/TestCreation.hpp"
#include "Utilities/GetOutput.hpp"

SPECTRE_TEST_CASE("Unit.GrMhd.ValenciaDivClean.Limiters.VariablesToLimit",
                  "[Limiters][Unit]") {
  CHECK(grmhd::ValenciaDivClean::Limiters::VariablesToLimit::Conserved ==
        TestHelpers::test_creation<
            grmhd::ValenciaDivClean::Limiters::VariablesToLimit>("Conserved"));
  CHECK(grmhd::ValenciaDivClean::Limiters::VariablesToLimit::
            NumericalCharacteristic ==
        TestHelpers::test_creation<
            grmhd::ValenciaDivClean::Limiters::VariablesToLimit>(
            "NumericalCharacteristic"));

  CHECK(get_output(
            grmhd::ValenciaDivClean::Limiters::VariablesToLimit::Conserved) ==
        "Conserved");
  CHECK(get_output(grmhd::ValenciaDivClean::Limiters::VariablesToLimit::
                       NumericalCharacteristic) == "NumericalCharacteristic");
}

// [[OutputRegex, Failed to convert "BadVars" to VariablesToLimit]]
SPECTRE_TEST_CASE(
    "Unit.GrMhd.ValenciaDivClean.Limiters.VariablesToLimit.ParseErr",
    "[Limiters][Unit]") {
  ERROR_TEST();
  TestHelpers::test_creation<
      grmhd::ValenciaDivClean::Limiters::VariablesToLimit>("BadVars");
}
