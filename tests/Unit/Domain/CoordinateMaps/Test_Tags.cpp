// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <string>

#include "DataStructures/Tensor/IndexType.hpp"
#include "Domain/CoordinateMaps/Tags.hpp"
#include "Helpers/DataStructures/DataBox/TestHelpers.hpp"

namespace domain {
namespace {
template <size_t Dim>
void test() {
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
