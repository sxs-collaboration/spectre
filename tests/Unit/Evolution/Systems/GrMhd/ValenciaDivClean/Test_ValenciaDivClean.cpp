// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <string>

#include "DataStructures/Tensor/IndexType.hpp"  // IWYU pragma: keep
#include "Evolution/Systems/GrMhd/ValenciaDivClean/Tags.hpp"

SPECTRE_TEST_CASE("Unit.Evolution.Systems.GrMhd.ValenciaDivClean.Tags",
                  "[Unit][GrMhd]") {
  CHECK(grmhd::ValenciaDivClean::Tags::TildeD::name() == "TildeD");
  CHECK(grmhd::ValenciaDivClean::Tags::TildeTau::name() == "TildeTau");
  CHECK(grmhd::ValenciaDivClean::Tags::TildeS<Frame::System>::name() ==
        "System_TildeS");
  CHECK(grmhd::ValenciaDivClean::Tags::TildeS<
            Frame::GlobalTimeIndependent>::name() ==
        "GlobalTimeIndependent_TildeS");
  CHECK(grmhd::ValenciaDivClean::Tags::TildeB<Frame::System>::name() ==
        "System_TildeB");
  CHECK(grmhd::ValenciaDivClean::Tags::TildeB<
            Frame::GlobalTimeIndependent>::name() ==
        "GlobalTimeIndependent_TildeB");
  CHECK(grmhd::ValenciaDivClean::Tags::TildePhi::name() == "TildePhi");
}
