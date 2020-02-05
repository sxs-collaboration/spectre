// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <string>

#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "Domain/Tags.hpp"
#include "tests/Unit/DataStructures/DataBox/TestHelpers.hpp"

namespace {
template <size_t Dim>
void test() noexcept {
  TestHelpers::db::test_simple_tag<Tags::Domain<Dim>>("Domain");
  TestHelpers::db::test_simple_tag<Tags::InitialExtents<Dim>>("InitialExtents");
  TestHelpers::db::test_simple_tag<Tags::InitialRefinementLevels<Dim>>(
      "InitialRefinementLevels");
  TestHelpers::db::test_simple_tag<Tags::Element<Dim>>("Element");
  TestHelpers::db::test_simple_tag<Tags::Mesh<Dim>>("Mesh");
  TestHelpers::db::test_simple_tag<Tags::ElementMap<Dim>>(
      "ElementMap(Inertial)");
  TestHelpers::db::test_simple_tag<Tags::ElementMap<Dim, Frame::Grid>>(
      "ElementMap(Grid)");
  TestHelpers::db::test_simple_tag<Tags::Coordinates<Dim, Frame::Grid>>(
      "GridCoordinates");
  TestHelpers::db::test_simple_tag<Tags::Coordinates<Dim, Frame::Logical>>(
      "LogicalCoordinates");
  TestHelpers::db::test_simple_tag<Tags::Coordinates<Dim, Frame::Inertial>>(
      "InertialCoordinates");
}

SPECTRE_TEST_CASE("Unit.Domain.Tags", "[Unit][Domain]") {
  test<1>();
  test<2>();
  test<3>();
}
}  // namespace
