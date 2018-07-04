// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <string>

#include "Domain/Amr/Flag.hpp"
#include "Utilities/GetOutput.hpp"

SPECTRE_TEST_CASE("Unit.Domain.Amr.Flag", "[Domain][Unit]") {
  CHECK(get_output(domain::amr::Flag::Undefined) == "Undefined");
  CHECK(get_output(domain::amr::Flag::Join) == "Join");
  CHECK(get_output(domain::amr::Flag::DecreaseResolution) ==
        "DecreaseResolution");
  CHECK(get_output(domain::amr::Flag::DoNothing) == "DoNothing");
  CHECK(get_output(domain::amr::Flag::IncreaseResolution) ==
        "IncreaseResolution");
  CHECK(get_output(domain::amr::Flag::Split) == "Split");
}
