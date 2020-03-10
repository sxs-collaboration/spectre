// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <string>

#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "DataStructures/Tensor/IndexType.hpp"
#include "Domain/CoordinateMaps/Tags.hpp"
#include "tests/Unit/DataStructures/DataBox/TestHelpers.hpp"

namespace domain {
namespace {
template <size_t Dim>
void test() noexcept {
  TestHelpers::db::test_simple_tag<
      CoordinateMaps::Tags::CoordinateMap<Dim, Frame::Logical, Frame::Grid>>(
      "CoordinateMap(Logical,Grid)");
  TestHelpers::db::test_simple_tag<
      CoordinateMaps::Tags::CoordinateMap<Dim, Frame::Grid, Frame::Inertial>>(
      "CoordinateMap(Grid,Inertial)");
}

SPECTRE_TEST_CASE("Unit.Domain.CoordinateMaps.Tags", "[Unit][Domain]") {
  test<1>();
  test<2>();
  test<3>();
}
}  // namespace
}  // namespace domain
