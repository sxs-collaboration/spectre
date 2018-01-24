// Distributed under the MIT License.
// See LICENSE.txt for details.

#include <cstddef>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/Tensor/IndexType.hpp"
#include "Domain/CoordinateMaps/AffineMap.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.hpp"
#include "Domain/CoordinateMaps/ProductMaps.hpp"
#include "Domain/ElementId.hpp"
#include "Domain/ElementMap.hpp"
#include "Domain/Tags.hpp"
#include "tests/Unit/TestHelpers.hpp"

namespace {
template <size_t Dim, typename T>
void test_coordinates_compute_item(Index<Dim> extents, T map) noexcept {
  using map_tag = Tags::ElementMap<Dim>;
  const auto box = db::create<
      db::AddTags<Tags::Extents<Dim>, map_tag>,
      db::AddComputeItemsTags<
          Tags::LogicalCoordinates<Dim>,
          Tags::Coordinates<map_tag, Tags::LogicalCoordinates<Dim>>>>(
      extents, ElementMap<Dim, Frame::Grid>(
                   ElementId<Dim>(0),
                   make_coordinate_map_base<Frame::Logical, Frame::Grid>(map)));
  CHECK_ITERABLE_APPROX(
      (db::get<Tags::Coordinates<map_tag, Tags::LogicalCoordinates<Dim>>>(box)),
      (make_coordinate_map<Frame::Logical, Frame::Grid>(map)(
          db::get<Tags::LogicalCoordinates<Dim>>(box))));
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Domain.CoordinatesTag", "[Unit][Domain]") {
  using AffineMap = CoordinateMaps::AffineMap;
  using AffineMap2d = CoordinateMaps::ProductOf2Maps<AffineMap, AffineMap>;
  using AffineMap3d =
      CoordinateMaps::ProductOf3Maps<AffineMap, AffineMap, AffineMap>;

  test_coordinates_compute_item(Index<1>{5}, AffineMap{-1.0, 1.0, -0.3, 0.7});
  test_coordinates_compute_item(Index<2>{5, 7},
                                AffineMap2d{AffineMap{-1.0, 1.0, -0.3, 0.7},
                                            AffineMap{-1.0, 1.0, 0.3, 0.55}});
  test_coordinates_compute_item(Index<3>{5, 6, 9},
                                AffineMap3d{AffineMap{-1.0, 1.0, -0.3, 0.7},
                                            AffineMap{-1.0, 1.0, 0.3, 0.55},
                                            AffineMap{-1.0, 1.0, 2.3, 2.8}});
}
