// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <string>

#include "Domain/Amr/Flag.hpp"
#include "Utilities/GetOutput.hpp"

SPECTRE_TEST_CASE("Unit.Domain.Amr.Flag", "[Domain][Unit]") {
  CHECK(get_output(amr::domain::Flag::Undefined) == "Undefined");
  CHECK(get_output(amr::domain::Flag::Join) == "Join");
  CHECK(get_output(amr::domain::Flag::DecreaseResolution) ==
        "DecreaseResolution");
  CHECK(get_output(amr::domain::Flag::DoNothing) == "DoNothing");
  CHECK(get_output(amr::domain::Flag::IncreaseResolution) ==
        "IncreaseResolution");
  CHECK(get_output(amr::domain::Flag::Split) == "Split");
}
