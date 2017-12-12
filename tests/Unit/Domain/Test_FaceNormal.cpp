// Distributed under the MIT License.
// See LICENSE.txt for details.

#include <catch.hpp>
#include <cmath>

#include "DataStructures/Index.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/CoordinateMaps/AffineMap.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.hpp"
#include "Domain/CoordinateMaps/ProductMaps.hpp"
#include "Domain/CoordinateMaps/Rotation.hpp"
#include "Domain/Direction.hpp"
#include "Domain/FaceNormal.hpp"
#include "Utilities/Gsl.hpp"
#include "tests/Unit/TestHelpers.hpp"

namespace {
template <typename Map>
void check(const Map& map,
           const std::array<std::array<double, Map::dim>, Map::dim>& expected) {
  const Index<Map::dim - 1> extents(3);
  for (size_t d = 0; d < Map::dim; ++d) {
    const auto upper_normal = unnormalized_grid_normal(
        extents, map, Direction<Map::dim>(d, Side::Upper));
    const auto lower_normal = unnormalized_grid_normal(
        extents, map, Direction<Map::dim>(d, Side::Lower));
    for (size_t i = 0; i < 2; ++i) {
      CHECK_ITERABLE_APPROX(upper_normal.get(i),
                            DataVector(extents.product(),
                                       gsl::at(gsl::at(expected, d), i)));
      CHECK_ITERABLE_APPROX(lower_normal.get(i),
                            DataVector(extents.product(),
                                       -gsl::at(gsl::at(expected, d), i)));
    }
  }
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Domain.GridNormal", "[Unit][Domain]") {
  /// [grid_normal_example]
  const Index<0> extents_0d;
  const auto map_1d = make_coordinate_map<Frame::Logical, Frame::Grid>(
      CoordinateMaps::AffineMap(-1.0, 1.0, -3.0, 7.0));
  const auto normal_1d_lower =
      unnormalized_grid_normal(extents_0d, map_1d, Direction<1>::lower_xi());
  /// [grid_normal_example]

  CHECK(normal_1d_lower.get(0) == DataVector(1, -0.2));

  const auto normal_1d_upper =
      unnormalized_grid_normal(extents_0d, map_1d, Direction<1>::upper_xi());

  CHECK(normal_1d_upper.get(0) == DataVector(1, 0.2));

  check(make_coordinate_map<Frame::Logical, Frame::Grid>(
            CoordinateMaps::Rotation<2>(atan2(4., 3.))),
        {{{{0.6, 0.8}}, {{-0.8, 0.6}}}});

  check(make_coordinate_map<Frame::Logical, Frame::Grid>(
            CoordinateMaps::ProductOf2Maps<CoordinateMaps::AffineMap,
                                           CoordinateMaps::Rotation<2>>(
                {-1., 1., 2., 7.}, CoordinateMaps::Rotation<2>(atan2(4., 3.)))),
        {{{{0.4, 0., 0.}}, {{0., 0.6, 0.8}}, {{0., -0.8, 0.6}}}});
}
