// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <string>

#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/Tags/SurfaceJacobian.hpp"
#include "Helpers/DataStructures/DataBox/TestHelpers.hpp"

namespace domain {

SPECTRE_TEST_CASE("Unit.Domain.Tags.SurfaceJacobian", "[Unit][Domain]") {
  TestHelpers::db::test_simple_tag<
      Tags::DetSurfaceJacobian<Frame::ElementLogical, Frame::Inertial>>(
      "DetSurfaceJacobian");
}

}  // namespace domain
