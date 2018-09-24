// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <string>

#include "DataStructures/Tensor/IndexType.hpp"  // IWYU pragma: keep
#include "Evolution/Systems/RelativisticEuler/Valencia/Tags.hpp"

SPECTRE_TEST_CASE("Unit.RelativisticEuler.Valencia.Tags", "[Unit][Evolution]") {
  CHECK(RelativisticEuler::Valencia::Tags::TildeD::name() == "TildeD");
  CHECK(RelativisticEuler::Valencia::Tags::TildeTau::name() == "TildeTau");
  CHECK(RelativisticEuler::Valencia::Tags::TildeS<3, Frame::Inertial>::name() ==
        "TildeS");
  CHECK(RelativisticEuler::Valencia::Tags::TildeS<3, Frame::Logical>::name() ==
        "Logical_TildeS");
}
