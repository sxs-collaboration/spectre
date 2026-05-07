// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <string>

#include "Evolution/Systems/GrMhd/ValenciaDivClean/System.hpp"

SPECTRE_TEST_CASE("Unit.grmhd.ValenciaDivClean.System.Name",
                  "[Unit][Evolution]") {
  CHECK(grmhd::ValenciaDivClean::System::name() == "ValenciaDivClean");
}
